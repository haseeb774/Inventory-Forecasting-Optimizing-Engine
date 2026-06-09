import pandas as pd
import numpy as np
from datetime import datetime

def generate_products():
    products = [
        ("SKU001", "Modern Sofa", 799),
        ("SKU002", "Coffee Table", 199),
        ("SKU003", "Dining Chair", 89),
        ("SKU004", "Bookshelf", 249),
        ("SKU005", "Bed Frame", 499),
        ("SKU006", "Nightstand", 129),
        ("SKU007", "Desk Lamp", 49),
        ("SKU008", "Office Desk", 349),
        ("SKU009", "TV Stand", 279),
        ("SKU010", "Accent Chair", 159),
    ]

    rows = []

    for idx, (sku, title, price) in enumerate(products, start=1):
        rows.append({
            "product_id": idx,
            "variant_id": idx * 100,
            "sku": sku,
            "product_title": title,
            "price": price,
            "current_stock": np.random.randint(50, 300)
        })

    return pd.DataFrame(rows)

def generate_orders(products_df):
    np.random.seed(42)

    dates = pd.date_range(
        start="2022-01-01",
        end="2025-01-01",
        freq="D"
    )

    rows = []

    order_id = 100000

    for date in dates:

        for _, product in products_df.iterrows():

            base_demand = np.random.poisson(3)

            month = date.month

            seasonal_multiplier = 1

            if month in [11, 12]:
                seasonal_multiplier = 2.0

            elif month in [6, 7]:
                seasonal_multiplier = 1.4

            quantity = int(
                max(
                    0,
                    base_demand * seasonal_multiplier
                )
            )

            if quantity == 0:
                continue

            rows.append({
                "order_id": order_id,
                "order_date": date,
                "financial_status": "paid",
                "product_id": product["product_id"],
                "variant_id": product["variant_id"],
                "sku": product["sku"],
                "product_title": product["product_title"],
                "quantity": quantity,
                "unit_price": product["price"],
                "total_price": quantity * product["price"]
            })

            order_id += 1

    return pd.DataFrame(rows)

if __name__ == "__main__":

    products_df = generate_products()

    orders_df = generate_orders(products_df)

    products_df.to_csv(
        "data/raw/products_raw.csv",
        index=False
    )

    orders_df.to_csv(
        "data/raw/orders_raw.csv",
        index=False
    )

    print(products_df.shape)
    print(orders_df.shape)