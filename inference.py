#!/usr/bin/env python3
"""
Run inference on test data using a trained SOH regression model.

Usage:
    python inference.py --test_path input/测试集_特征_标签.csv \
                        --model_path output/model_artifacts.pkl \
                        --output_path output/predictions.csv
"""

import argparse
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Feature engineering (mirrors train.py)
# ---------------------------------------------------------------------------

FILL_ZERO_COLS = [
    "pack_v_kurtosis", "pack_v_skew", "pack_p_kurtosis", "pack_p_skew",
]


def build_row_features(df: pd.DataFrame, wr_mapping: dict) -> pd.DataFrame:
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
    df: pd.DataFrame, wr_mapping: dict, order_feat_names: list
) -> pd.DataFrame:
    feat_df = build_row_features(df, wr_mapping)
    feat_df["order_id"] = df["order_id"].values

    label_col = "soh" if "soh" in df.columns else "y_t"
    if label_col in df.columns:
        feat_df["label"] = df[label_col].values
    else:
        feat_df["label"] = 0.0

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
    agg = agg.reset_index()

    for c in order_feat_names:
        if c not in agg.columns:
            agg[c] = 0.0
    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Inference for SOH regression")
    parser.add_argument("--test_path", required=True, help="Path to test CSV")
    parser.add_argument("--model_path", required=True,
                        help="Path to model_artifacts.pkl")
    parser.add_argument("--output_path", default="output/predictions.csv",
                        help="Path to save predictions CSV")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    # ---- load -------------------------------------------------------------
    print("Loading model artifacts …")
    arts = joblib.load(args.model_path)
    lgb_model = arts["lgb_model"]
    mlp_model = arts["mlp_model"]
    ridge_model = arts["ridge_model"]
    ord_scaler = arts["ord_scaler"]
    pca = arts["pca"]
    wr_mapping = arts["wr_mapping"]
    ord_feat_names = arts["ord_feat_names"]
    w = arts["weights"]

    print("Loading test data …")
    test_df = pd.read_csv(args.test_path)

    # ---- row-level predictions (LightGBM) ---------------------------------
    X_test_row = build_row_features(test_df, wr_mapping)
    pred_lgb = lgb_model.predict(X_test_row)

    test_df["pred_lgb"] = pred_lgb
    lgb_ord = (
        test_df.groupby("order_id")["pred_lgb"].mean().reset_index()
    )

    # ---- order-level predictions (MLP, Ridge) -----------------------------
    ord_test = build_order_features(test_df, wr_mapping, ord_feat_names)
    X_ot = ord_test[ord_feat_names].values
    X_ot_pca = pca.transform(ord_scaler.transform(X_ot))

    pred_mlp = mlp_model.predict(X_ot_pca)
    pred_ridge = ridge_model.predict(X_ot_pca)

    ord_preds = pd.DataFrame({
        "order_id": ord_test["order_id"].values,
        "pred_mlp": pred_mlp,
        "pred_ridge": pred_ridge,
    })

    # ---- ensemble ---------------------------------------------------------
    final = lgb_ord.merge(ord_preds, on="order_id", how="left")
    final["pred_mlp"] = final["pred_mlp"].fillna(final["pred_lgb"])
    final["pred_ridge"] = final["pred_ridge"].fillna(final["pred_lgb"])

    ws = w["w_lgb"] + w["w_mlp"] + w["w_ridge"] + 1e-10
    final["prediction"] = (
        w["w_lgb"] * final["pred_lgb"]
        + w["w_mlp"] * final["pred_mlp"]
        + w["w_ridge"] * final["pred_ridge"]
    ) / ws

    # ---- row-level predictions --------------------------------------------
    mlp_map = dict(zip(ord_preds["order_id"], pred_mlp))
    ridge_map = dict(zip(ord_preds["order_id"], pred_ridge))
    test_df["pred_mlp"] = test_df["order_id"].map(mlp_map)
    test_df["pred_ridge"] = test_df["order_id"].map(ridge_map)
    test_df["pred_mlp"] = test_df["pred_mlp"].fillna(test_df["pred_lgb"])
    test_df["pred_ridge"] = test_df["pred_ridge"].fillna(test_df["pred_lgb"])
    test_df["prediction"] = (
        w["w_lgb"] * test_df["pred_lgb"]
        + w["w_mlp"] * test_df["pred_mlp"]
        + w["w_ridge"] * test_df["pred_ridge"]
    ) / ws

    # ---- evaluate if labels exist -----------------------------------------
    if "y_t" in test_df.columns:
        row_mae = mean_absolute_error(test_df["y_t"], test_df["prediction"])
        print(f"\nRow-level MAE: {row_mae:.6f}")

        order_true = test_df.groupby("order_id")["y_t"].first()
        order_pred = final.set_index("order_id")["prediction"]
        common = order_true.index.intersection(order_pred.index)
        order_mae = mean_absolute_error(
            order_true.loc[common], order_pred.loc[common]
        )
        print(f"Order-level MAE: {order_mae:.6f}")

    # ---- save -------------------------------------------------------------
    out = final[["order_id", "prediction"]].copy()
    if "y_t" in test_df.columns:
        true_map = test_df.groupby("order_id")["y_t"].first()
        out["y_t"] = out["order_id"].map(true_map)
    out.to_csv(args.output_path, index=False)
    print(f"\nPredictions saved to {args.output_path}")

    row_out = args.output_path.replace(".csv", "_row_level.csv")
    keep = ["order_id", "window_range", "prediction"]
    if "y_t" in test_df.columns:
        keep.append("y_t")
    test_df[keep].to_csv(row_out, index=False)
    print(f"Row-level predictions saved to {row_out}")


if __name__ == "__main__":
    main()
