# src/reorder_logic.py

import pickle
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta


class Reorder:

    MODEL_PATH      = "models/xgboost_model.pkl"
    STOCK_DATA_PATH = "data/raw/products_raw.csv"
    OUTPUT_PATH     = "data/outputs/reorder_alerts.csv"
    LEAD_TIME_DAYS  = 7
    SERVICE_LEVEL_Z = 1.645  # 95% service level

    def __init__(self):
        self.forecast_summary = None

    # ─────────────────────────────────────────────
    # STEP 1 — Forecast (your method, lightly cleaned)
    # ─────────────────────────────────────────────
    def prediction(self, forecast_days: int = 30):
        with open(self.MODEL_PATH, 'rb') as f:
            saved = pickle.load(f)
        model = saved['model']
        MODEL_FEATURES = saved['feature_cols']

        # ✅ Use REAL historical data instead of fabricated seed
        features_df = pd.read_csv("data/features/features.csv", parse_dates=["order_date"])
        input_df = pd.read_csv(self.STOCK_DATA_PATH)

        type_mapping = {"Bedding": 0, "Decor": 1, "Fragrance": 2, "Kitchen": 3}
        input_df['product_type_code'] = input_df['product_type'].map(type_mapping).fillna(0).astype(int)
        input_df['price_tier'] = pd.qcut(input_df['price'], q=3, labels=[1,2,3], duplicates='drop').astype(float)

        # Build history_df from REAL data — last 95 days per SKU
        history_df = (
            features_df
            .sort_values("order_date")
            .groupby("sku")
            .tail(95)[["sku", "order_date", "units_sold", "price", "current_stock",
                    "product_type_code", "price_tier"]]
            .copy()
        )
        history_df["order_date"] = history_df["order_date"].dt.date

        start_date = features_df["order_date"].max()  # continue from real last date, not today

        forecast_results = []

        for day in range(1, forecast_days + 1):
            target_date = start_date + timedelta(days=day)
            

            day_features = {
                'day_of_week':        target_date.weekday(),
                'day_of_month':       target_date.day,
                'week_of_year':       target_date.isocalendar()[1],
                'month':              target_date.month,
                'quarter':            (target_date.month - 1) // 3 + 1,
                'year':               target_date.year,
                'is_weekend':         1 if target_date.weekday() >= 5 else 0,
                'is_friday':          1 if target_date.weekday() == 4 else 0,
                'is_month_end':       1 if (target_date + timedelta(days=1)).day == 1 else 0,
                'is_holiday_season':  1 if target_date.month in [11, 12] else 0,
                'is_eid_fitr_window': 0,
                'is_eid_adha_window': 0,
                'is_eid_season':      0,
                'is_peak_season':     0,
            }

            for _, sku_row in input_df.iterrows():
                sku = sku_row['sku']
                sku_hist = history_df[history_df['sku'] == sku].sort_values('order_date')
                past_sales = sku_hist['units_sold'].values

                lags = {
                    'lag_1d':  past_sales[-1]  if len(past_sales) >= 1  else 0,
                    'lag_3d':  past_sales[-3]  if len(past_sales) >= 3  else 0,
                    'lag_7d':  past_sales[-7]  if len(past_sales) >= 7  else 0,
                    'lag_14d': past_sales[-14] if len(past_sales) >= 14 else 0,
                    'lag_21d': past_sales[-21] if len(past_sales) >= 21 else 0,
                    'lag_30d': past_sales[-30] if len(past_sales) >= 30 else 0,
                }
                rolling = {
                    'rolling_mean_7d':  np.mean(past_sales[-7:]),
                    'rolling_std_7d':   np.std(past_sales[-7:]),
                    'rolling_mean_14d': np.mean(past_sales[-14:]),
                    'rolling_std_14d':  np.std(past_sales[-14:]),
                    'rolling_mean_30d': np.mean(past_sales[-30:]),
                    'rolling_std_30d':  np.std(past_sales[-30:]),
                    'rolling_mean_60d': np.mean(past_sales[-60:]),
                    'rolling_std_60d':  np.std(past_sales[-60:]),
                    'rolling_mean_90d': np.mean(past_sales[-90:]),
                    'rolling_std_90d':  np.std(past_sales[-90:]),
                }
                profile = {
                    'sku_mean_demand': np.mean(past_sales),
                    'sku_std_demand':  np.std(past_sales),
                    'sku_demand_cv':   np.std(past_sales) / (np.mean(past_sales) + 1e-5),
                    'trend_7_vs_30':   np.mean(past_sales[-7:]) / (np.mean(past_sales[-30:]) + 1e-5),
                    'cumulative_units': int(np.sum(past_sales)),
                }
                interactions = {
                    'sku_x_eid':     profile['sku_mean_demand'] * day_features['is_eid_season'],
                    'sku_x_holiday': profile['sku_mean_demand'] * day_features['is_holiday_season'],
                    'sku_x_peak':    profile['sku_mean_demand'] * day_features['is_peak_season'],
                    'sku_x_weekend': profile['sku_mean_demand'] * day_features['is_weekend'],
                    'demand_x_lag':  profile['sku_mean_demand'] * lags['lag_1d'],
                }

                full_row = {
                    **day_features, **lags, **rolling, **profile, **interactions,
                    'price':             sku_row['price'],
                    'current_stock':     sku_row['current_stock'],
                    'product_type_code': sku_row['product_type_code'],
                    'price_tier':        sku_row['price_tier'],
                }

                X_test = pd.DataFrame([full_row])[MODEL_FEATURES]
                predicted_units = max(0, model.predict(X_test)[0])

                history_df = pd.concat([history_df, pd.DataFrame([{
                    'sku': sku, 'order_date': target_date.date(),
                    'units_sold': predicted_units, 'price': sku_row['price'],
                    'current_stock': sku_row['current_stock'],
                    'product_type_code': sku_row['product_type_code'],
                    'price_tier': sku_row['price_tier']
                }])], ignore_index=True)

                forecast_results.append({
                    'sku': sku, 'date': target_date.date(), 'predicted_sales': predicted_units
                })

        forecast_output = pd.DataFrame(forecast_results)

        summary = forecast_output.groupby('sku').agg(
            total_30d_demand = ('predicted_sales', 'sum'),
            avg_daily_demand = ('predicted_sales', 'mean'),
            std_daily_demand = ('predicted_sales', 'std'),
        ).reset_index()

        self.forecast_summary = summary.merge(
            input_df[['sku', 'product_title', 'current_stock', 'price']],
            on='sku'
        )
        return self.forecast_summary

    # ─────────────────────────────────────────────
    # STEP 2 — Safety stock (fixed: uses real per-SKU std, not hardcoded)
    # ─────────────────────────────────────────────
    def calculate_safety_stock(self, std_daily_demand):
        return self.SERVICE_LEVEL_Z * std_daily_demand * np.sqrt(self.LEAD_TIME_DAYS)

    # ─────────────────────────────────────────────
    # STEP 3 — Reorder point (fixed: returns the value!)
    # ─────────────────────────────────────────────
    def calculate_reorder_point(self, avg_daily_demand, std_daily_demand):
        safety_stock = self.calculate_safety_stock(std_daily_demand)
        rop = (avg_daily_demand * self.LEAD_TIME_DAYS) + safety_stock
        return rop, safety_stock

    # ─────────────────────────────────────────────
    # STEP 4 — Generate alerts (fixed: row-by-row, real comparisons)
    # ─────────────────────────────────────────────
    def generate_alerts(self):
        if self.forecast_summary is None:
            raise ValueError("Run .prediction() first.")

        df = self.forecast_summary.copy()
        rows = []

        for _, r in df.iterrows():
            avg_demand = r['avg_daily_demand']
            std_demand = r['std_daily_demand'] if not np.isnan(r['std_daily_demand']) else avg_demand * 0.3
            current_stock = r['current_stock']

            rop, safety_stock = self.calculate_reorder_point(avg_demand, std_demand)

            days_until_stockout = current_stock / (avg_demand + 1e-5)

            # EOQ — assumes order cost $20, holding cost = 15% of price annually
            annual_demand = avg_demand * 365
            order_cost    = 20
            holding_cost  = r['price'] * 0.15
            eoq = np.sqrt((2 * annual_demand * order_cost) / (holding_cost + 1e-5))

            # ── Status logic ──
            if current_stock < rop:
                status = "CRITICAL"
                urgency = f"Order now — stockout in {days_until_stockout:.0f} days"
                recommended_qty = round(eoq)
            elif current_stock < rop * 1.5:
                status = "WARNING"
                urgency = f"Order within 7 days — {days_until_stockout:.0f} days of stock left"
                recommended_qty = round(eoq)
            elif current_stock > 3 * r['total_30d_demand']:
                status = "OVERSTOCK"
                urgency = f"Reduce next order — {days_until_stockout:.0f} days of stock on hand"
                recommended_qty = 0
            else:
                status = "HEALTHY"
                urgency = "Stock levels are sufficient"
                recommended_qty = 0

            rows.append({
                "sku":                    r["sku"],
                "product_title":          r["product_title"],
                "current_stock":          int(current_stock),
                "daily_demand":           round(avg_demand, 2),
                "reorder_point":          round(rop, 1),
                "safety_stock":           round(safety_stock, 1),
                "recommended_order_qty":  recommended_qty,
                "days_until_stockout":    round(days_until_stockout, 1),
                "status":                 status,
                "urgency_message":        urgency
            })

        result = pd.DataFrame(rows).sort_values(
            by="status",
            key=lambda x: x.map({"CRITICAL": 0, "WARNING": 1, "HEALTHY": 2, "OVERSTOCK": 3})
        )

        os.makedirs("data/outputs", exist_ok=True)
        result.to_csv(self.OUTPUT_PATH, index=False)

        print("\n" + "="*90)
        print("REORDER ALERTS")
        print("="*90)
        print(result.to_string(index=False))
        print(f"\n✅ Saved to {self.OUTPUT_PATH}")

        return result


if __name__ == "__main__":
    obj = Reorder()
    obj.prediction(forecast_days=30)
    obj.generate_alerts()