# src/shopify_connector.py
from src.exception import CustomException
from src.logger import logging
import sys

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv

load_dotenv()

class ShopifyConnector:
    """
    Connects to Shopify Admin API and pulls
    historical orders + product data for NestHaven.
    """

    def __init__(self):
        self.store_url  = os.getenv("SHOPIFY_STORE_URL")
        self.token      = os.getenv("SHOPIFY_ACCESS_TOKEN")
        self.api_version = "2024-01"
        self.base_url   = f"https://{self.store_url}/admin/api/{self.api_version}"
        self.headers    = {
            "X-Shopify-Access-Token": self.token,
            "Content-Type": "application/json"
        }

    def _get(self, endpoint, params={}):
        """
        Safe API call with rate limit handling.
        Shopify allows 2 requests/second — we respect that.
        """
        try:
            url = f"{self.base_url}/{endpoint}"
            all_data = []
            
            while url:
                response = requests.get(url, headers=self.headers, params=params)

                # Handle rate limiting (429)
                if response.status_code == 429:
                    print("Rate limited — waiting 2 seconds...")
                    time.sleep(2)
                    continue

                response.raise_for_status()
                data = response.json()

                # Collect results
                key = list(data.keys())[0]
                all_data.extend(data[key])

                # Handle pagination (Shopify uses Link header)
                link = response.headers.get("Link", "")
                if 'rel="next"' in link:
                    next_url = [
                        part.strip().split(";")[0].strip("<>")
                        for part in link.split(",")
                        if 'rel="next"' in part
                    ]
                    url = next_url[0] if next_url else None
                    params = {}  # params are in the next URL already
                else:
                    url = None

                time.sleep(0.5)  # be polite to Shopify

            return all_data
        except Exception as e:
            logging.error(f"Error during API call: {str(e)}")
            raise CustomException(str(e), sys)

    def get_orders(self, days_back=1095):  # 1095 = 3 years
        """
        Pull all orders from the last N days.
        Returns a clean DataFrame ready for the pipeline.
        """
        since_date = (datetime.now() - timedelta(days=days_back))
        since_str  = since_date.strftime("%Y-%m-%dT%H:%M:%S")

        print(f"Fetching orders since {since_str}...")
        try:
            raw_orders = self._get("orders.json", params={
                "status":        "any",
                "created_at_min": since_str,
                "limit":         250,  # max per page
                "fields":        "id,created_at,financial_status,line_items,total_price"
            })
            print(f"Orders returned: {len(raw_orders)}")

            if raw_orders:
                print("First order:")
                print(raw_orders[0])

            print(f"Fetched {len(raw_orders)} orders. Parsing...")

            # Flatten line items — one row per SKU per order
            rows = []
            for order in raw_orders:
                for item in order.get("line_items", []):
                    rows.append({
                        "order_id":        order["id"],
                        "order_date":      order["created_at"][:10],  # YYYY-MM-DD
                        "financial_status": order.get("financial_status"),
                        "product_id":      item.get("product_id"),
                        "variant_id":      item.get("variant_id"),
                        "sku":             item.get("sku", "UNKNOWN"),
                        "product_title":   item.get("title"),
                        "quantity":        item.get("quantity", 0),
                        "unit_price":      float(item.get("price", 0)),
                        "total_price":     float(item.get("price", 0)) * item.get("quantity", 0)
                    })

            df = pd.DataFrame(rows)
            if df.empty:
                print("No orders found in the specified date range.")
                return df
            df["order_date"] = pd.to_datetime(df["order_date"])
            print(f"Parsed orders into DataFrame with shape {df.shape}.")
            return df
        except Exception as e:
            logging.error(f"Error fetching orders: {str(e)}")
            raise CustomException(str(e), sys)
    def get_products(self):
        """
        Pull product catalog — titles, SKUs, inventory levels.
        """
        print("Fetching product catalog...")
        try:
            raw_products = self._get("products.json", params={
                "limit":  250,
                "fields": "id,title,variants,product_type,tags"
            })

            rows = []
            for product in raw_products:
                for variant in product.get("variants", []):
                    rows.append({
                        "product_id":       product["id"],
                        "product_title":    product["title"],
                        "product_type":     product.get("product_type", ""),
                        "tags":             product.get("tags", ""),
                        "variant_id":       variant["id"],
                        "sku":              variant.get("sku", "UNKNOWN"),
                        "current_stock":    variant.get("inventory_quantity", 0),
                        "price":            float(variant.get("price", 0))
                    })

            df = pd.DataFrame(rows)
            print(f"Fetched {len(df)} variants across {len(raw_products)} products.")
            return df
        except Exception as e:
            logging.error(f"Error fetching products: {str(e)}")
            raise CustomException(str(e), sys)

    def save_to_csv(self, df, filename):
        """Save raw pull to data/raw/ folder."""
        try:
            os.makedirs("data/raw", exist_ok=True)
            path = f"data/raw/{filename}_{datetime.now().strftime('%Y%m%d')}.csv"
            df.to_csv(path, index=False)
            print(f"Saved to {path}")
            return path
        except Exception as e:
            logging.error(f"Error saving to CSV: {str(e)}")
            raise CustomException(str(e), sys)


# ── Run this file directly to test the connection ──────────────────
if __name__ == "__main__":
    connector = ShopifyConnector()

    products_df = connector.get_products()

    print(products_df.head())
    print(products_df.shape)