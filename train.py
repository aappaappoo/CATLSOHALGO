#!/usr/bin/env python3
"""
Train a regression model to predict battery SOH.

Usage:
    python train.py --train_path input/训练集_特征_标签.csv \
                    --test_path input/测试集_特征_标签.csv \
                    --mapping_path input/window_range_maping.json \
                    --output_dir output \
                    --n_trials 50
"""

import argparse
import json
import os
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

FILL_ZERO_COLS = [
    "pack_v_kurtosis", "pack_v_skew", "pack_p_kurtosis", "pack_p_skew",
]


def build_row_features(df: pd.DataFrame, wr_mapping: dict) -> pd.DataFrame:
    """Create per-row model features from raw data."""
    out = df.copy()
    rv = out["rated_voltage"]
    re = out["rated_energy"]

    out["v_ratio"] = out["pack_v_max"] / rv
    out["v_ptp_ratio"] = out["pack_v_ptp"] / rv
    out["v_meandiff_ratio"] = out["pack_v_meandiff"] / rv
    out["p_ptp_ratio"] = out["pack_p_ptp"] / re
    out["p_meandiff_ratio"] = out["pack_p_meandiff"] / re

    for c in FILL_ZERO_COLS:
        out[c] = out[c].fillna(0.0)

    out["wr_encoded"] = (
        out["window_range"].map(wr_mapping).fillna(-1).astype(float)
    )

    feature_cols = [
        "v_ratio", "v_ptp_ratio",
        "pack_v_kurtosis", "pack_v_skew", "v_meandiff_ratio",
        "p_ptp_ratio", "pack_p_kurtosis", "pack_p_skew", "p_meandiff_ratio",
        "dqdv_peak_05", "dqdv_05_ptp", "dqdv_peak_10", "dqdv_10_ptp",
        "wr_encoded", "window_range_idx", "window_median",
        "rated_capacity", "rated_voltage", "rated_energy",
    ]
    return out[feature_cols]


def build_order_features(
    df: pd.DataFrame, wr_mapping: dict
) -> pd.DataFrame:
    """Aggregate row-level features to one row per order_id."""
    feat_df = build_row_features(df, wr_mapping)
    feat_df["order_id"] = df["order_id"].values

    label_col = "soh" if "soh" in df.columns else "y_t"
    feat_df["label"] = df[label_col].values if label_col in df.columns else 0.0

    agg_cols = [c for c in feat_df.columns if c not in ("order_id", "label")]
    grouped = feat_df.groupby("order_id", sort=False)

    agg = grouped[agg_cols + ["order_id"]].agg(
        {**{c: ["mean", "std", "min", "max"] for c in agg_cols},
         "order_id": "count"},
    )
    agg.columns = ["_".join(col).strip("_") for col in agg.columns]
    agg = agg.rename(columns={"order_id_count": "n_windows"})

    agg["label"] = grouped["label"].first()
    agg = agg.fillna(0).replace([np.inf, -np.inf], 0)
    return agg.reset_index()


# ---------------------------------------------------------------------------
# Optuna helpers
# ---------------------------------------------------------------------------

def _lgb_objective(trial, X_tr, y_tr, X_val, y_val):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "learning_rate": trial.suggest_float("lr", 0.005, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "verbose": -1, "random_state": 42,
    }
    m = lgb.LGBMRegressor(**params)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
    return mean_absolute_error(y_val, m.predict(X_val))


def _mlp_objective(trial, X_tr, y_tr, X_val, y_val):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layers = tuple(trial.suggest_int(f"l{i}", 32, 256) for i in range(n_layers))
    alpha = trial.suggest_float("alpha", 0.01, 10.0, log=True)
    lr = trial.suggest_float("lr", 1e-4, 0.01, log=True)
    m = MLPRegressor(
        hidden_layer_sizes=layers, max_iter=3000, alpha=alpha,
        learning_rate_init=lr, learning_rate="adaptive", random_state=42,
        early_stopping=True, validation_fraction=0.15, n_iter_no_change=30,
    )
    m.fit(X_tr, y_tr)
    return mean_absolute_error(y_val, m.predict(X_val))


def _ridge_objective(trial, X_tr, y_tr, X_val, y_val):
    alpha = trial.suggest_float("alpha", 1e-3, 1e4, log=True)
    m = Ridge(alpha=alpha)
    m.fit(X_tr, y_tr)
    return mean_absolute_error(y_val, m.predict(X_val))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train SOH regression model")
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--test_path", default=None,
                        help="Optional test CSV for evaluation")
    parser.add_argument("--mapping_path", required=True)
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--n_trials", type=int, default=50,
                        help="Optuna trials per model type")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)

    # ---- load -------------------------------------------------------------
    print("[1/5] Loading data …")
    train_df = pd.read_csv(args.train_path)
    with open(args.mapping_path) as fh:
        wr_mapping = json.load(fh)

    test_df = None
    if args.test_path and os.path.exists(args.test_path):
        test_df = pd.read_csv(args.test_path)

    # ---- row-level features (for LightGBM) --------------------------------
    print("[2/5] Building features …")
    X_row_all = build_row_features(train_df, wr_mapping)
    y_all = train_df["soh"]
    row_feat_names = list(X_row_all.columns)

    X_row_tr, X_row_val, y_row_tr, y_row_val = train_test_split(
        X_row_all, y_all, test_size=0.15, random_state=args.seed,
    )

    # ---- order-level features (for MLP / Ridge) ---------------------------
    order_df = build_order_features(train_df, wr_mapping)
    ord_feat = [c for c in order_df.columns if c not in ("order_id", "label")]
    X_ord = order_df[ord_feat].values
    y_ord = order_df["label"].values

    X_ord_tr, X_ord_val, y_ord_tr, y_ord_val = train_test_split(
        X_ord, y_ord, test_size=0.15, random_state=args.seed,
    )
    ord_scaler = StandardScaler()
    X_ord_tr_s = ord_scaler.fit_transform(X_ord_tr)
    X_ord_val_s = ord_scaler.transform(X_ord_val)

    # optional PCA on order-level features
    pca = PCA(n_components=min(50, X_ord_tr_s.shape[1]), random_state=args.seed)
    X_ord_tr_pca = pca.fit_transform(X_ord_tr_s)
    X_ord_val_pca = pca.transform(X_ord_val_s)

    # ---- 1. LightGBM (row-level) -----------------------------------------
    print("[3/5] Optimising LightGBM …")
    lgb_study = optuna.create_study(direction="minimize")
    lgb_study.optimize(
        lambda t: _lgb_objective(t, X_row_tr, y_row_tr, X_row_val, y_row_val),
        n_trials=args.n_trials,
    )
    best_lgb = {**lgb_study.best_params, "verbose": -1, "random_state": 42}
    best_lgb["learning_rate"] = best_lgb.pop("lr")
    lgb_model = lgb.LGBMRegressor(**best_lgb)
    lgb_model.fit(X_row_all, y_all)
    print(f"  LGB val MAE = {lgb_study.best_value:.6f}")

    # ---- 2. MLP (order-level + PCA) --------------------------------------
    print("[4/5] Optimising MLP & Ridge (order-level) …")
    mlp_study = optuna.create_study(direction="minimize")
    mlp_study.optimize(
        lambda t: _mlp_objective(t, X_ord_tr_pca, y_ord_tr, X_ord_val_pca, y_ord_val),
        n_trials=args.n_trials,
    )
    bp = mlp_study.best_params
    n_layers = bp.pop("n_layers")
    layers = tuple(bp.pop(f"l{i}") for i in range(n_layers))
    best_alpha = bp.pop("alpha")
    best_lr = bp.pop("lr")

    # Retrain on full data
    full_ord_scaler = StandardScaler()
    X_ord_all_s = full_ord_scaler.fit_transform(X_ord)
    full_pca = PCA(n_components=min(50, X_ord_all_s.shape[1]),
                   random_state=args.seed)
    X_ord_all_pca = full_pca.fit_transform(X_ord_all_s)

    mlp_model = MLPRegressor(
        hidden_layer_sizes=layers, max_iter=3000, alpha=best_alpha,
        learning_rate_init=best_lr, learning_rate="adaptive",
        random_state=42, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=30,
    )
    mlp_model.fit(X_ord_all_pca, y_ord)
    print(f"  MLP val MAE = {mlp_study.best_value:.6f}")

    # ---- 3. Ridge (order-level + PCA) ------------------------------------
    ridge_study = optuna.create_study(direction="minimize")
    ridge_study.optimize(
        lambda t: _ridge_objective(t, X_ord_tr_pca, y_ord_tr, X_ord_val_pca, y_ord_val),
        n_trials=args.n_trials,
    )
    ridge_model = Ridge(alpha=ridge_study.best_params["alpha"])
    ridge_model.fit(X_ord_all_pca, y_ord)
    print(f"  Ridge val MAE = {ridge_study.best_value:.6f}")

    # ---- ensemble weights (optimised on order-level val split) -----------
    print("[5/5] Optimising ensemble weights …")
    # Rebuild per-order validation predictions
    val_ids = set(
        train_df.loc[X_row_val.index, "order_id"].unique()
    )
    ord_val_mask = order_df["order_id"].isin(val_ids)
    X_ov = order_df.loc[ord_val_mask, ord_feat].values
    y_ov = order_df.loc[ord_val_mask, "label"].values
    oid_ov = order_df.loc[ord_val_mask, "order_id"].values

    X_ov_s = full_ord_scaler.transform(X_ov)
    X_ov_pca = full_pca.transform(X_ov_s)

    p_mlp = mlp_model.predict(X_ov_pca)
    p_ridge = ridge_model.predict(X_ov_pca)

    # LGB row -> order mean
    val_df = train_df.loc[X_row_val.index].copy()
    val_df["pred_lgb"] = lgb_model.predict(X_row_val)
    lgb_ord = val_df.groupby("order_id")["pred_lgb"].mean()

    merged = pd.DataFrame({
        "order_id": oid_ov, "label": y_ov,
        "p_mlp": p_mlp, "p_ridge": p_ridge,
    })
    merged["p_lgb"] = merged["order_id"].map(lgb_ord)
    merged = merged.dropna()

    if len(merged) > 5:
        def ens_obj(trial):
            w1 = trial.suggest_float("w_lgb", 0, 1)
            w2 = trial.suggest_float("w_mlp", 0, 1)
            w3 = trial.suggest_float("w_ridge", 0, 1)
            s = w1 + w2 + w3 + 1e-10
            pred = (w1 * merged["p_lgb"] + w2 * merged["p_mlp"]
                    + w3 * merged["p_ridge"]) / s
            return mean_absolute_error(merged["label"], pred)

        ens_study = optuna.create_study(direction="minimize")
        ens_study.optimize(ens_obj, n_trials=300)
        weights = ens_study.best_params
    else:
        weights = {"w_lgb": 1 / 3, "w_mlp": 1 / 3, "w_ridge": 1 / 3}

    print(f"  Weights: {weights}")

    # ---- save -------------------------------------------------------------
    artifacts = {
        "lgb_model": lgb_model,
        "mlp_model": mlp_model,
        "ridge_model": ridge_model,
        "ord_scaler": full_ord_scaler,
        "pca": full_pca,
        "wr_mapping": wr_mapping,
        "row_feat_names": row_feat_names,
        "ord_feat_names": ord_feat,
        "weights": weights,
    }
    path = os.path.join(args.output_dir, "model_artifacts.pkl")
    joblib.dump(artifacts, path)
    print(f"\nSaved → {path}")

    # ---- optional test evaluation -----------------------------------------
    if test_df is not None:
        _evaluate_test(test_df, artifacts)


# ---------------------------------------------------------------------------
# Test evaluation helper
# ---------------------------------------------------------------------------

def _evaluate_test(test_df, arts):
    print("\n--- Test-set evaluation ---")
    wr_mapping = arts["wr_mapping"]
    lgb_model = arts["lgb_model"]
    mlp_model = arts["mlp_model"]
    ridge_model = arts["ridge_model"]
    ord_scaler = arts["ord_scaler"]
    pca = arts["pca"]
    ord_feat = arts["ord_feat_names"]
    w = arts["weights"]

    X_test_row = build_row_features(test_df, wr_mapping)
    pred_lgb = lgb_model.predict(X_test_row)

    test_copy = test_df.copy()
    test_copy["pred_lgb"] = pred_lgb
    lgb_ord = test_copy.groupby("order_id").agg(
        y_t=("y_t", "first"), pred_lgb=("pred_lgb", "mean"),
    ).reset_index()

    ord_test = build_order_features(test_df, wr_mapping)
    for c in ord_feat:
        if c not in ord_test.columns:
            ord_test[c] = 0.0
    X_ot = ord_test[ord_feat].values
    X_ot_s = ord_scaler.transform(X_ot)
    X_ot_pca = pca.transform(X_ot_s)

    pred_mlp = mlp_model.predict(X_ot_pca)
    pred_ridge = ridge_model.predict(X_ot_pca)

    final = lgb_ord.copy()
    mlp_map = dict(zip(ord_test["order_id"], pred_mlp))
    ridge_map = dict(zip(ord_test["order_id"], pred_ridge))
    final["pred_mlp"] = final["order_id"].map(mlp_map)
    final["pred_ridge"] = final["order_id"].map(ridge_map)

    ws = w["w_lgb"] + w["w_mlp"] + w["w_ridge"] + 1e-10
    final["pred_ens"] = (
        w["w_lgb"] * final["pred_lgb"]
        + w["w_mlp"] * final["pred_mlp"]
        + w["w_ridge"] * final["pred_ridge"]
    ) / ws

    for col in ["pred_lgb", "pred_mlp", "pred_ridge", "pred_ens"]:
        mae = mean_absolute_error(final["y_t"], final[col])
        print(f"  {col:15s} order-MAE = {mae:.6f}")

    # row-level ensemble
    test_copy["pred_mlp"] = test_copy["order_id"].map(mlp_map)
    test_copy["pred_ridge"] = test_copy["order_id"].map(ridge_map)
    test_copy["pred_ens"] = (
        w["w_lgb"] * test_copy["pred_lgb"]
        + w["w_mlp"] * test_copy["pred_mlp"]
        + w["w_ridge"] * test_copy["pred_ridge"]
    ) / ws
    row_mae = mean_absolute_error(test_copy["y_t"], test_copy["pred_ens"])
    print(f"\n  Row-level ensemble MAE = {row_mae:.6f}")


if __name__ == "__main__":
    main()
