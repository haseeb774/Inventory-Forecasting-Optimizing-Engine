from src.shopify_connector import ShopifyConnector
from src.exception import CustomException
from src.logger import logging
import sys


def run_shopify_connector():
    try:
        logging.info("Starting Shopify Connector...")

        connector = ShopifyConnector()

        orders_df = connector.get_orders(days_back=1095)
        products_df = connector.get_products()

        connector.save_to_csv(orders_df, "orders_raw")
        connector.save_to_csv(products_df, "products_raw")

        logging.info(
            f"Fetched {len(orders_df)} order rows and "
            f"{len(products_df)} product rows."
        )

        print("Orders:", orders_df.shape)
        print("Products:", products_df.shape)

    except Exception as e:
        logging.error(f"Error in Shopify Connector: {e}")
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_shopify_connector()