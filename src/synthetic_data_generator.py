# scripts/generate_realistic_csv.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)
class RealisticDataGenerator:
    PRODUCTS = [
        {"sku": "LDS-001", "title": "Linen Duvet Set",     "price": 89.99, "base": 8,  "type": "Bedding",   "season": "eid_high"},
        {"sku": "CVB-002", "title": "Ceramic Vase Bundle", "price": 45.00, "base": 5,  "type": "Decor",     "season": "holiday_high"},
        {"sku": "SCP-003", "title": "Scented Candle Pack", "price": 29.99, "base": 12, "type": "Fragrance", "season": "holiday_high"},
        {"sku": "WAB-004", "title": "Wall Art Boho",       "price": 65.00, "base": 4,  "type": "Decor",     "season": "eid_high"},
        {"sku": "TRS-005", "title": "Table Runner Set",    "price": 22.50, "base": 3,  "type": "Kitchen",   "season": "flat"},
        {"sku": "TPC-006", "title": "Throw Pillow Cover",  "price": 18.99, "base": 7,  "type": "Bedding",   "season": "eid_high"},
    ]

    EID_WINDOWS = [
        ("2022-04-18", "2022-05-04"),
        ("2022-06-25", "2022-07-11"),
        ("2023-04-07", "2023-04-23"),
        ("2023-06-15", "2023-07-01"),
        ("2024-03-27", "2024-04-12"),
        ("2024-06-04", "2024-06-20"),
    ]

    HOLIDAY_WINDOWS = [
        ("2022-11-01", "2022-12-31"),
        ("2023-11-01", "2023-12-31"),
        ("2024-11-01", "2024-12-31"),
    ]
    def __init__(self,seed: int = 42,
        start_date: str = "2022-01-01",
        end_date: str = "2025-01-01"):
         np.random.seed(seed)
         self.date_range = pd.date_range(start_date, end_date, freq="D")
         self.orders_df = None
         self.products_df = None

    def is_in_windows(self,date:pd.Timestamp,windows:list) -> bool:
        for start, end in windows:
            if pd.Timestamp(start) <= date <= pd.Timestamp(end):
                return True
        return False

    def get_multiplier(self,date:pd.Timestamp, season:str):
        m = 1.0

        # Weekly patterns
        if date.dayofweek == 4:        # Friday — big shopping day Pakistan
            m *= 1.8
        elif date.dayofweek in [5, 6]: # Weekend
            m *= 1.4

        # Month-end salary effect
        if date.day >= 25:
            m *= 1.5

        # Dead season — Jan/Feb post-holiday slump
        if date.month in [1, 2]:
            m *= 0.4   # stronger drop

        # Pre-Eid buildup — people shop 2 weeks before
        if self.is_in_windows(date, self.EID_WINDOWS):
            if season == "eid_high":
                m *= 6.0   # strong Eid spike
            elif season == "holiday_high":
                m *= 2.5
            else:
                m *= 1.8   # even flat products spike slightly

        # Nov-Dec holiday season
        if self.is_in_windows(date, self.HOLIDAY_WINDOWS):
            if season == "holiday_high":
                m *= 5.0   # strong holiday spike
            elif season == "eid_high":
                m *= 2.0
            else:
                m *= 1.5

        return m

    def generate_orders(self):
        date_range = pd.date_range("2022-01-01", "2025-01-01", freq="D")
        rows = []
        order_id = 10000

        for date in date_range:
            for p in self.PRODUCTS:
                multiplier   = self.get_multiplier(date, p["season"])
                expected_qty = p["base"] * multiplier

                # Poisson noise — realistic for retail demand
                qty = np.random.poisson(expected_qty)
                if qty == 0:
                    continue

                rows.append({
                    "order_id":         order_id,
                    "order_date":       date.strftime("%Y-%m-%d"),
                    "financial_status": "paid",
                    "product_id":       abs(hash(p["sku"])) % 100000,
                    "variant_id":       abs(hash(p["sku"] + "v")) % 100000,
                    "sku":              p["sku"],
                    "product_title":    p["title"],
                    "quantity":         qty,
                    "unit_price":       p["price"],
                    "total_price":      round(p["price"] * qty, 2)
                })
                order_id += 1

        return pd.DataFrame(rows)

    def generate_products(self):
        rows = []
        for p in self.PRODUCTS:
            rows.append({
                "product_id":    abs(hash(p["sku"])) % 100000,
                "product_title": p["title"],
                "product_type":  p["type"],
                "tags":          p["season"],
                "variant_id":    abs(hash(p["sku"] + "v")) % 100000,
                "sku":           p["sku"],
                "current_stock": np.random.randint(10, 200),
                "price":         p["price"]
            })
        return pd.DataFrame(rows)

    def save_data(self, output_dir: str = "data/raw"):
        """Saves generated DataFrames into structural physical CSV storage locations."""
        os.makedirs(output_dir, exist_ok=True)

        if self.orders_df is not None:
            path = os.path.join(output_dir, "orders_raw.csv")
            self.orders_df.to_csv(path, index=False)
            print(f"✅ {len(self.orders_df):,} orders saved to {path}")
        

        if self.products_df is not None:
            path = os.path.join(output_dir, "products_raw.csv")
            self.products_df.to_csv(path, index=False)
            print(f"✅ {len(self.products_df)} products saved to {path}")

    def run_diagnostic(self):
        """Displays data summary logs validating seasonality trends."""
        if self.orders_df is None:
            print("❌ No data available. Run generation functions first.")
            return

        df = self.orders_df.copy()
        df["order_date"] = pd.to_datetime(df["order_date"])
        df["month"] = df["order_date"].dt.month
        monthly = df.groupby("month")["quantity"].mean().round(2)

        print("\nSample — peak vs off-peak check:")
        print(monthly)
        print(
            "\nExpected: Nov/Dec/Eid months should be 2-3x higher than Jan/Feb"
        )


# Operational Pipeline Execution Block
if __name__ == "__main__":
    # Create generator instance with customized timelines
    generator = RealisticDataGenerator(
        start_date="2022-01-01", end_date="2025-01-01"
    )
    
    print("Generating realistic orders...")
    # FIX: Assign the returned DataFrame to self.orders_df
    generator.orders_df = generator.generate_orders()
    
    print("\nGenerating products catalog...")
    # FIX: Assign the returned DataFrame to self.products_df
    generator.products_df = generator.generate_products()
    
    # Save to disk and evaluate trends
    generator.save_data()
    generator.run_diagnostic()