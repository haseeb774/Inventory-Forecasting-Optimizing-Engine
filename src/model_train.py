import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error
)

NON_FEATURE_COLS = [
    "sku", "order_date", "product_title",
    "product_type", "units_sold",
    "revenue", "num_orders", "avg_price",
    "current_stock"   # ← add this, it's an outcome not a predictor
]
FEATURES_COLS = None

class ModelTrainer:

    def __init__(self):
        self.model       = None
        self.feature_cols = None
        self.cutoff_date  = None
        self.results      = {}

    # ─────────────────────────────────────────
    # STEP 1 — Load & Validate
    # ─────────────────────────────────────────
    def load_data(self, path="data/features/features.csv"):
        print("Loading features...")
        df = pd.read_csv(path, parse_dates=["order_date"])
        df = df.sort_values(["sku", "order_date"]).reset_index(drop=True)

        print(f"Shape: {df.shape}")
        print(f"Date range: {df['order_date'].min()} → {df['order_date'].max()}")
        print(f"SKUs: {df['sku'].nunique()}")
        print(f"Total rows: {len(df):,}")

        # fix Issue 3 from Haseeb's code — smart NaN fill before dropping
        df["lag_14d"] = df["lag_14d"].fillna(df["rolling_mean_7d"])
        df["lag_21d"] = df["lag_21d"].fillna(df["rolling_mean_14d"])
        df["lag_30d"] = df["lag_30d"].fillna(df["rolling_mean_30d"])
        df = df.dropna(subset=["lag_7d"])

        return df

    # ─────────────────────────────────────────
    # STEP 2 — Time-based Train/Test Split
    # ─────────────────────────────────────────
    def split(self, df, test_ratio=0.2):
        """
        Split by date — not random.
        Last 20% of dates = test set.
        This simulates real forecasting conditions.
        """
        self.cutoff_date = df["order_date"].quantile(1 - test_ratio)
        print(f"\nTrain/test cutoff: {self.cutoff_date.date()}")

        train = df[df["order_date"] <= self.cutoff_date].copy()
        test  = df[df["order_date"] >  self.cutoff_date].copy()

        print(f"Train rows: {len(train):,} | Test rows: {len(test):,}")

        # Feature columns = everything except non-features
        self.feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

        X_train = train[self.feature_cols]
        y_train = train["units_sold"]
        X_test  = test[self.feature_cols]
        y_test  = test["units_sold"]

        return X_train, y_train, X_test, y_test, test

    def train(self, X_train, y_train, X_test, y_test):
        print("\nTraining XGBoost model...")

        self.model = xgb.XGBRegressor(
            n_estimators          = 1000,   # more trees
            max_depth             = 4,      # shallower — prevents overfitting on small data
            learning_rate         = 0.02,   # slower learning = better generalization
            subsample             = 0.7,
            colsample_bytree      = 0.7,
            min_child_weight      = 3,      # lower — allows learning from fewer samples
            early_stopping_rounds = 100,    # more patience
            reg_alpha             = 0.1,    # L1 regularization
            reg_lambda            = 1.0,    # L2 regularization
            eval_metric           = "mae",
            random_state          = 42)
        self.model.fit(X_train,y_train,eval_set= [(X_train,y_train),(X_test,y_test)],verbose= 100)
        print(f"Best iteration: {self.model.best_iteration}")
        return self
    
    
    
    def evaluate(self, X_test, y_test, test_df):
        def smape(y_true, y_pred):
            """
            Symmetric MAPE — handles zero actual values gracefully.
            Returns percentage between 0-100.
            Use this instead of MAPE for inventory data
            where many days have zero sales.
            """
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            
            denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
            
            # Where both actual and predicted are 0 — perfect prediction, error = 0
            mask = denominator != 0
            
            result = np.zeros_like(y_true, dtype=float)
            result[mask] = np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]
            
            return np.mean(result) * 100

        print("\n" + "="*55)
        print("MODEL EVALUATION")
        print("="*55)

        y_pred = self.model.predict(X_test)
        y_pred = np.clip(y_pred, 0, None)

        mae        = mean_absolute_error(y_test, y_pred)
        rmse       = root_mean_squared_error(y_test, y_pred)
        smape_score = smape(y_test.values, y_pred)  # ✅ use smape now

        print(f"\n{'Global MAE':<25} {mae:.2f} units")
        print(f"{'Global RMSE':<25} {rmse:.2f} units")
        print(f"{'Global SMAPE':<25} {smape_score:.2f}%")
        print(f"{'Global Accuracy':<25} {100 - smape_score:.2f}%")

        if smape_score <= 15:
            print(f"\n✅ CONTRACT GUARANTEE MET — SMAPE {smape_score:.2f}% is below 15%")
        else:
            print(f"\n⚠️  NOT MET — SMAPE {smape_score:.2f}% exceeds 15%")

        # Per-SKU
        print(f"\n{'─'*60}")
        print(f"{'SKU':<25} {'MAE':>8} {'SMAPE':>10} {'Accuracy':>10} {'Status':>8}")
        print(f"{'─'*60}")

        test_copy = test_df.copy()
        test_copy["y_pred"] = y_pred
        test_copy["y_test"] = y_test.values

        sku_results = {}
        for sku, group in test_copy.groupby("sku"):
            sku_mae   = mean_absolute_error(group["y_test"], group["y_pred"])
            sku_smape = smape(group["y_test"].values, group["y_pred"].values)
            sku_acc   = 100 - sku_smape
            status    = "✅" if sku_smape <= 15 else "⚠️ "

            sku_results[sku] = {
                "mae":      round(sku_mae,   2),
                "smape":    round(sku_smape, 2),
                "accuracy": round(sku_acc,   2)
            }
            print(f"{sku:<25} {sku_mae:>8.2f} {sku_smape:>9.2f}% {sku_acc:>9.2f}% {status:>8}")

        self.results = {
            "global":  {"mae": mae, "rmse": rmse, "smape": smape_score},
            "per_sku": sku_results
        }
        return y_pred
    def evaluate_baseline(self, test_df):
            """
            Naive baseline: predict 'same as 7 days ago' for each SKU.
            If XGBoost can't beat this, it's not earning its complexity.
            """
            def smape(y_true, y_pred):
                y_true = np.array(y_true)
                y_pred = np.array(y_pred)
                denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
                mask = denominator != 0
                result = np.zeros_like(y_true, dtype=float)
                result[mask] = np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]
                return np.mean(result) * 100

            baseline_pred = test_df["lag_7d"].fillna(0).values
            y_true = test_df["units_sold"].values

            baseline_mae   = mean_absolute_error(y_true, baseline_pred)
            baseline_smape = smape(y_true, baseline_pred)

            xgb_smape = self.results["global"]["smape"]
            improvement = ((baseline_smape - xgb_smape) / baseline_smape) * 100

            print("\n" + "="*55)
            print("BASELINE COMPARISON (naive: same as 7 days ago)")
            print("="*55)
            print(f"{'Baseline MAE':<25} {baseline_mae:.2f} units")
            print(f"{'Baseline SMAPE':<25} {baseline_smape:.2f}%")
            print(f"{'XGBoost SMAPE':<25} {xgb_smape:.2f}%")
            print(f"{'Improvement over baseline':<25} {improvement:.1f}%")

            self.results["baseline"] = {
                "mae": baseline_mae,
                "smape": baseline_smape,
                "improvement_pct": improvement
            }
            return self.results["baseline"]
        
    def show_feature_importance(self, top_n=15):
        importance = pd.DataFrame({
            "feature":    self.feature_cols,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False).head(top_n)

        print(f"\nTop {top_n} Most Important Features:")
        print(f"{'─'*40}")
        for _, row in importance.iterrows():
            bar   = "█" * int(row["importance"] * 300)
            print(f"{row['feature']:<25} {bar} {row['importance']:.4f}")

        return importance

    def forecast(self, df, sku, days=30):
        """
        Predict next N days for a specific SKU.
        Uses last known data as starting point,
        then rolls forward day by day.
        """
        sku_df = df[df["sku"] == sku].copy()
        sku_df = sku_df.sort_values("order_date")

        if sku_df.empty:
            print(f"SKU {sku} not found.")
            return None

        last_row   = sku_df.iloc[-1].copy()
        last_date  = sku_df["order_date"].max()
        recent_sales = sku_df["units_sold"].tail(30).values

        forecasts = []
        for i in range(1, days + 1):
            forecast_date = last_date + pd.Timedelta(days=i)

            # Build feature row for this future date
            row = last_row.copy()
            row["order_date"]    = forecast_date
            row["day_of_week"]   = forecast_date.dayofweek
            row["day_of_month"]  = forecast_date.day
            row["week_of_year"]  = forecast_date.isocalendar()[1]
            row["month"]         = forecast_date.month
            row["quarter"]       = (forecast_date.month - 1) // 3 + 1
            row["year"]          = forecast_date.year
            row["is_weekend"]    = int(forecast_date.dayofweek in [4, 5, 6])
            row["is_friday"]     = int(forecast_date.dayofweek == 4)
            row["is_month_end"]  = int(forecast_date.day >= 25)

            # Use rolling average as lag proxy for future dates
            avg = np.mean(recent_sales[-7:])
            row["lag_1d"]  = recent_sales[-1] if len(recent_sales) >= 1 else avg
            row["lag_7d"]  = recent_sales[-7] if len(recent_sales) >= 7 else avg
            row["lag_14d"] = recent_sales[-14] if len(recent_sales) >= 14 else avg
            row["lag_30d"] = recent_sales[-30] if len(recent_sales) >= 30 else avg

            row["rolling_mean_7d"]  = np.mean(recent_sales[-7:])
            row["rolling_mean_30d"] = np.mean(recent_sales[-30:])
            row["rolling_mean_90d"] = np.mean(recent_sales)
            row["trend_7_vs_30"]    = (
                row["rolling_mean_7d"] / (row["rolling_mean_30d"] + 1e-5)
            )

            X_future = pd.DataFrame([row])[self.feature_cols]
            pred = float(self.model.predict(X_future)[0])
            pred = max(0, round(pred, 1))  # no negative units

            forecasts.append({
                "date":         forecast_date.strftime("%Y-%m-%d"),
                "sku":          sku,
                "forecast_units": pred
            })

            # Roll the recent sales window forward
            recent_sales = np.append(recent_sales, pred)

        return pd.DataFrame(forecasts)

    # ─────────────────────────────────────────
    # STEP 7 — Save Model + Metadata
    # ─────────────────────────────────────────
    def save(self):
        os.makedirs("models", exist_ok=True)

        joblib.dump({
            "model":        self.model,
            "feature_cols": self.feature_cols,
            "cutoff_date":  self.cutoff_date,
            "results":      self.results
        }, "models/xgboost_model.pkl")

        print("\n✅ Model saved to models/xgboost_model.pkl")


# ── Run directly ───────────────────────────────────────────────
if __name__ == "__main__":
    trainer = ModelTrainer()

    # Load
    df = trainer.load_data("data/features/features.csv")

    # Split
    X_train, y_train, X_test, y_test, test_df = trainer.split(df)

    # Train
    trainer.train(X_train, y_train, X_test, y_test)

    # Evaluate
    trainer.evaluate(X_test, y_test, test_df)
    # Baseline comparison
    trainer.evaluate_baseline(test_df)

    # Feature importance
    trainer.show_feature_importance(top_n=15)

    # Test forecast for one SKU
    sample_sku = df["sku"].iloc[0]
    print(f"\nSample 30-day forecast for {sample_sku}:")
    forecast_df = trainer.forecast(df, sku=sample_sku, days=30)
    print(forecast_df.head(10))

    # Save
    trainer.save()