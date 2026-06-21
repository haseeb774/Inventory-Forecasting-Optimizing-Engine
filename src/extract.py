from src.exception import CustomException
from src.logger import logging
import sys

import requests
import pandas as pd
import time
import os
from dotenv import load_dotenv

load_dotenv()


class ShopifyConnector:

    def __init__(self):

        self.store_url = os.getenv("SHOPIFY_STORE_URL")
        self.token = os.getenv("SHOPIFY_ACCESS_TOKEN")

        self.api_version = "2024-01"

        self.base_url = (
            f"https://{self.store_url}/admin/api/{self.api_version}"
        )

        self.headers = {
            "X-Shopify-Access-Token": self.token,
            "Content-Type": "application/json"
        }

    def _get(self, endpoint, params=None):

        if params is None:
            params = {}

        try:

            url = f"{self.base_url}/{endpoint}"

            all_data = []

            while url:

                response = requests.get(
                    url,
                    headers=self.headers,
                    params=params
                )

                if response.status_code == 429:
                    print("Rate limited. Waiting...")
                    time.sleep(2)
                    continue

                response.raise_for_status()

                data = response.json()

                key = list(data.keys())[0]

                all_data.extend(data[key])

                link = response.headers.get("Link", "")

                if 'rel="next"' in link:

                    next_url = [
                        part.strip().split(";")[0].strip("<>")
                        for part in link.split(",")
                        if 'rel="next"' in part
                    ]

                    url = next_url[0] if next_url else None

                    params = {}

                else:
                    url = None

                time.sleep(0.5)

            return all_data

        except Exception as e:

            logging.error(str(e))

            raise CustomException(str(e), sys)

    def get_products(self):

        try:

            print("Fetching products from Shopify...")

            raw_products = self._get(
                "products.json",
                params={
                    "limit": 250,
                    "fields":
                    "id,title,variants,product_type,tags"
                }
            )

            rows = []

            for product in raw_products:

                for variant in product.get("variants", []):

                    sku = variant.get("sku")

                    if not sku:
                        sku = f"VARIANT_{variant['id']}"

                    rows.append({
                        "product_id": product["id"],
                        "product_title": product["title"],
                        "product_type": product.get(
                            "product_type", ""
                        ),
                        "tags": product.get("tags", ""),
                        "variant_id": variant["id"],
                        "sku": sku,
                        "current_stock":
                        variant.get(
                            "inventory_quantity", 0
                        ),
                        "price":
                        float(
                            variant.get("price", 0)
                        )
                    })

            df = pd.DataFrame(rows)

            print(
                f"Fetched {len(df)} variants "
                f"across {len(raw_products)} products."
            )

            return df

        except Exception as e:

            logging.error(str(e))

            raise CustomException(str(e), sys)