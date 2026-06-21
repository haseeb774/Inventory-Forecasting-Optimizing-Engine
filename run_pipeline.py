# run pipeline
from src.diagnostoc import diagnostic
from src.synthetic_data_generator import RealisticDataGenerator
from src.model_train import ModelTrainer
from src.forecasting_feature import Reorder
from src.transform import DataPipeline
def run_pipeline():
    # data generate/ extract
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

    #data transform
    pipeline = DataPipeline()
    pipeline.run(
        orders_path   = f"data/raw/orders_raw.csv",
        products_path = f"data/raw/products_raw.csv"
    )


    print("\nSample of final features:")
    print(pipeline.features_df[["sku","order_date","units_sold",
                                  "lag_7d","rolling_mean_30d",
                                  "is_eid_season","is_peak_season"]].head(10))
    
    # model train predictions
    trainer = ModelTrainer()

    # Load
    df = trainer.load_data("data/features/features.csv")

    # Split
    X_train, y_train, X_test, y_test, test_df = trainer.split(df)

    # Train
    trainer.train(X_train, y_train, X_test, y_test)

    # Evaluate
    trainer.evaluate(X_test, y_test, test_df)

    # Feature importance
    trainer.show_feature_importance(top_n=15)

    # Test forecast for one SKU
    sample_sku = df["sku"].iloc[0]
    print(f"\nSample 30-day forecast for {sample_sku}:")
    forecast_df = trainer.forecast(df, sku=sample_sku, days=30)
    print(forecast_df.head(10))

    # Save
    trainer.save()
    reorder = Reorder()
    reorder.prediction(forecast_days=30)
    reorder.generate_alerts()
    diagnostic()
if __name__=="__main__":
    run_pipeline()