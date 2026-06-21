# src/data_pipeline.py

import pandas as pd
import numpy as np
import os
from datetime import datetime

class DataPipeline:
    """
    Takes raw Shopify orders + products data and
    produces a clean, feature-rich dataset ready
    for XGBoost training.
    """

    def __init__(self):
        self.orders_df   = None
        self.products_df = None
        self.daily_df    = None  # aggregated daily sales per SKU
        self.features_df = None  # final ML-ready features

    # ─────────────────────────────────────────
    # STEP 1 — Load raw data
    # ─────────────────────────────────────────
    def load_data(self, orders_path, products_path):
        print("Loading raw data...")
        self.orders_df   = pd.read_csv(orders_path, parse_dates=["order_date"])
        self.products_df = pd.read_csv(products_path)
        print(f"Orders loaded:   {self.orders_df.shape}")
        print(f"Products loaded: {self.products_df.shape}")
        return self

    # ─────────────────────────────────────────
    # STEP 2 — Clean orders
    # ─────────────────────────────────────────
    def clean_orders(self):
        print("\nCleaning orders...")
        df = self.orders_df.copy()
        df["order_date"] = pd.to_datetime(df["order_date"])

        # Drop refunded/cancelled orders — they are not real demand
        before = len(df)
        df = df[df["financial_status"] == "paid"]
        print(f"Removed {before - len(df)} unpaid/refunded orders")

        # Drop rows with missing SKU or zero quantity
        df = df.dropna(subset=["sku", "order_date"])
        df = df[df["quantity"] > 0]

        # Remove obvious data entry errors
        # Quantity > 500 in one order line is likely a data error
        df = df[df["quantity"] <= 500]

        # Remove negative prices
        df = df[df["unit_price"] >= 0]

        # Standardize SKU format — uppercase, strip whitespace
        df["sku"] = df["sku"].str.upper().str.strip()

        # Remove unknown SKUs
        df = df[df["sku"] != "UNKNOWN"]

        print(f"Clean orders shape: {df.shape}")
        self.orders_df = df
        return self

    # ─────────────────────────────────────────
    # STEP 3 — Aggregate to daily sales per SKU
    # This is the foundation of your time series
    # ─────────────────────────────────────────
    def aggregate_daily(self):
        print("\nAggregating to daily SKU sales...")
        df = self.orders_df.copy()

        # Group by SKU + date
        daily = df.groupby(["sku", "product_title", "order_date"]).agg(
            units_sold   = ("quantity",    "sum"),
            revenue      = ("total_price", "sum"),
            num_orders   = ("order_id",    "nunique"),
            avg_price    = ("unit_price",  "mean")
        ).reset_index()

        # Create a complete date range for every SKU
        # (fill gaps where no sales happened with 0)
        all_skus  = daily["sku"].unique()
        date_range = pd.date_range(
            start = daily["order_date"].min(),
            end   = daily["order_date"].max(),
            freq  = "D"
        )

        # Build full grid — every SKU × every date
        idx = pd.MultiIndex.from_product(
            [all_skus, date_range],
            names=["sku", "order_date"]
        )
        daily = daily.set_index(["sku", "order_date"])
        daily = daily.reindex(idx, fill_value=0).reset_index()

        # Re-attach product title (lost in reindex)
        title_map = self.orders_df.groupby("sku")["product_title"].first()
        daily["product_title"] = daily["sku"].map(title_map)

        print(f"Daily aggregated shape: {daily.shape}")
        self.daily_df = daily
        return self

    # ─────────────────────────────────────────
    # STEP 4 — Feature Engineering
    # This is where the ML magic comes from
    # ─────────────────────────────────────────
    def build_features(self):
        print("\nBuilding features...")
        df = self.daily_df.copy()
        df = df.sort_values(["sku", "order_date"])

        # ── DATE FEATURES ──────────────────────────────────
        df["day_of_week"]   = df["order_date"].dt.dayofweek    # 0=Mon 6=Sun
        df["day_of_month"]  = df["order_date"].dt.day
        df["week_of_year"]  = df["order_date"].dt.isocalendar().week.astype(int)
        df["month"]         = df["order_date"].dt.month
        df["quarter"]       = df["order_date"].dt.quarter
        df["year"]          = df["order_date"].dt.year
        df["is_weekend"]    = df["day_of_week"].isin([4, 5, 6]).astype(int)
        # In Pakistan, Friday is the big day (Jumma)
        df["is_friday"]     = (df["day_of_week"] == 4).astype(int)
        # Month-end salary effect (25th-31st)
        df["is_month_end"]  = (df["day_of_month"] >= 25).astype(int)

        # ── PAKISTANI SEASONAL FEATURES ────────────────────
        # These are the most important features for NestHaven

        # Nov-Dec holiday season (peak)
        df["is_holiday_season"] = df["month"].isin([11, 12]).astype(int)

        # Eid ul-Fitr — moves each year (lunar calendar)
        # Approximate Gregorian dates for 2022-2025
        eid_fitr_dates = [
            "2022-05-02", "2022-05-03",
            "2023-04-21", "2023-04-22",
            "2024-04-10", "2024-04-11",
            "2025-03-30", "2025-03-31",
        ]
        # 14-day pre-Eid window (shopping peaks before Eid)
        eid_fitr_windows = []
        for d in eid_fitr_dates:
            dt = pd.to_datetime(d)
            for i in range(-14, 3):
                eid_fitr_windows.append(dt + pd.Timedelta(days=i))

        # Eid ul-Adha — approximate dates
        eid_adha_dates = [
            "2022-07-09", "2022-07-10",
            "2023-06-28", "2023-06-29",
            "2024-06-17", "2024-06-18",
            "2025-06-06", "2025-06-07",
        ]
        eid_adha_windows = []
        for d in eid_adha_dates:
            dt = pd.to_datetime(d)
            for i in range(-14, 3):
                eid_adha_windows.append(dt + pd.Timedelta(days=i))

        df["is_eid_fitr_window"] = df["order_date"].isin(eid_fitr_windows).astype(int)
        df["is_eid_adha_window"] = df["order_date"].isin(eid_adha_windows).astype(int)
        df["is_eid_season"]      = ((df["is_eid_fitr_window"] == 1) | (df["is_eid_adha_window"] == 1)).astype(int)

        # Combined peak flag — any major event
        df["is_peak_season"] = ((df["is_holiday_season"] == 1) | (df["is_eid_season"] == 1)).astype(int)

        # ── LAG FEATURES ───────────────────────────────────
        # "What did this SKU sell yesterday / last week?"
        # XGBoost needs these to understand momentum
        for lag in [1, 3, 7, 14, 21, 30]:
            df[f"lag_{lag}d"] = df.groupby("sku")["units_sold"].shift(lag)

        # ── ROLLING WINDOW FEATURES ────────────────────────
        # Smoothed averages — removes daily noise
        for window in [7, 14, 30, 60, 90]:
            df[f"rolling_mean_{window}d"] = (
                df.groupby("sku")["units_sold"]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            )
            df[f"rolling_std_{window}d"] = (
                df.groupby("sku")["units_sold"]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std().fillna(0))
            )
        sku_mean = df.groupby("sku")["units_sold"].transform("mean")
        sku_std  = df.groupby("sku")["units_sold"].transform("std")
        df["sku_mean_demand"] = sku_mean
        df["sku_std_demand"]  = sku_std
        df["sku_demand_cv"]   = sku_std / (sku_mean + 1e-5)  # coefficient of variation
        # ── TREND FEATURES ─────────────────────────────────
        # Is this SKU growing or declining?
        df["trend_7_vs_30"] = (
            df["rolling_mean_7d"] / (df["rolling_mean_30d"] + 1e-5)
        )  # >1 means accelerating, <1 means slowing

        # ── CUMULATIVE FEATURES ────────────────────────────
        df["cumulative_units"] = df.groupby("sku")["units_sold"].cumsum()
        # ── INTERACTION FEATURES ───────────────────────────────
        # These combine SKU behavior WITH seasonal context
        # Critical for "Linen Duvet during Eid" pattern

        df["sku_x_eid"]         = df["sku_mean_demand"] * df["is_eid_season"]
        df["sku_x_holiday"]     = df["sku_mean_demand"] * df["is_holiday_season"]
        df["sku_x_peak"]        = df["sku_mean_demand"] * df["is_peak_season"]
        df["sku_x_weekend"]     = df["sku_mean_demand"] * df["is_weekend"]
        df["demand_x_lag"] = df["rolling_mean_7d"] * df["lag_1d"]

        # ── PRODUCT FEATURES ───────────────────────────────
        # Merge product metadata
        self.products_df["tags"] = self.products_df["tags"].fillna("No tags")
        self.products_df["product_type"] = self.products_df["product_type"].fillna("snowboard")
        product_meta = self.products_df[["sku", "product_type", "price", "current_stock"]].copy()
        product_meta["sku"] = product_meta["sku"].str.upper().str.strip()
        df = df.merge(product_meta, on="sku", how="left")

        # Encode product type as category code
        df["product_type_code"] = df["product_type"].astype("category").cat.codes

        # Price tier — budget / mid / premium
        df["price_tier"] = pd.cut(
            df["price"],
            bins   = [0, 25, 60, float("inf")],
            labels = [0, 1, 2]  # 0=budget, 1=mid, 2=premium
        ).astype(float)

        # Drop rows with NaN lags (first few days per SKU have no lag data)
        df = df.dropna(subset=["lag_7d", "rolling_mean_7d"])

        print(f"Final features shape: {df.shape}")
        print(f"Features created: {[c for c in df.columns if c not in ['sku','product_title','order_date','units_sold']]}")

        self.features_df = df
        return self

    # ─────────────────────────────────────────
    # STEP 5 — Save
    # ─────────────────────────────────────────
    def save(self):
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("data/features",  exist_ok=True)

        self.daily_df.to_csv("data/processed/daily_sales.csv", index=False)
        self.features_df.to_csv("data/features/features.csv", index=False)

        print("\n✅ Saved:")
        print("   data/processed/daily_sales.csv")
        print("   data/features/features.csv")
        return self

    def run(self, orders_path, products_path):
        """Run full pipeline in one call."""
        return (self
            .load_data(orders_path, products_path)
            .clean_orders()
            .aggregate_daily()
            .build_features()
            .save()
        )


# ── Run directly ───────────────────────────────────────
if __name__ == "__main__":
    # Find latest raw files
    orders_path   = sorted([f for f in os.listdir("data/raw") if "orders"   in f])[-1]
    products_path = sorted([f for f in os.listdir("data/raw") if "products" in f])[-1]

    pipeline = DataPipeline()
    pipeline.run(
        orders_path   = f"data/raw/orders_raw.csv",
        products_path = f"data/raw/products_raw.csv"
    )

    print("\nSample of final features:")
    print(pipeline.features_df[["sku","order_date","units_sold",
                                  "lag_7d","rolling_mean_30d",
                                  "is_eid_season","is_peak_season"]].head(10))