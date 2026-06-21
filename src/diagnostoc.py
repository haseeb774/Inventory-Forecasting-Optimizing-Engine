# # diagnostic.py — run this first
def diagnostic():


    import pandas as pd

    # Check 1 — what does the model actually see as "recent demand" per SKU?
    features_df = pd.read_csv("data/features/features.csv", parse_dates=["order_date"])

    print("Real average daily demand per SKU (last 90 days of actual data):")
    recent = features_df.sort_values("order_date").groupby("sku").tail(90)
    print(recent.groupby("sku")["units_sold"].mean().round(2))

    print("\nOverall average daily demand per SKU (all 3 years):")
    print(features_df.groupby("sku")["units_sold"].mean().round(2))

    print("\nLast known stock per SKU in products file:")
    products = pd.read_csv("data/raw/products_raw.csv")
    print(products[["sku", "current_stock", "price"]])
    import joblib
    saved = joblib.load("models/xgboost_model.pkl")
    model = saved["model"]
    cols = saved["feature_cols"]

    import pandas as pd
    imp = pd.DataFrame({"feature": cols, "importance": model.feature_importances_})
    print(imp.sort_values("importance", ascending=False).head(10))

    import pickle
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta

    with open("models/xgboost_model.pkl", "rb") as f:
        saved = pickle.load(f)
    MODEL_FEATURES = saved["feature_cols"]

    features_df = pd.read_csv("data/features/features.csv", parse_dates=["order_date"])

    history_df = (
        features_df.sort_values("order_date").groupby("sku").tail(95)
        [["sku", "order_date", "units_sold"]]
    )

    print("Last 7 actual units_sold per SKU (sanity check on raw history):")
    for sku in history_df["sku"].unique():
        vals = history_df[history_df["sku"]==sku]["units_sold"].values[-7:]
        print(f"{sku}: {vals}")

    print("\nsku_mean_demand computed from last 95 days per SKU:")
    for sku in history_df["sku"].unique():
        vals = history_df[history_df["sku"]==sku]["units_sold"].values
        print(f"{sku}: mean={np.mean(vals):.2f}  std={np.std(vals):.2f}")
if __name__ == "__main__":
    diagnostic()

