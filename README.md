![Tests](https://github.com/haseeb774/Inventory-Forecasting-Optimizing-Engine/actions/workflows/tests.yml/badge.svg)
# NestHaven Inventory Intelligence

**Smart inventory & demand forecasting system for e-commerce — built to catch stockouts before they happen and free up cash trapped in slow-moving stock.**

[Live demo](https://inventory-forecasting-optimizing-engine-fz6hc6egappoqzivhceixg.streamlit.app/) · [NestHaven_Case_Study.pdf](https://github.com/user-attachments/files/29355360/NestHaven_Case_Study.pdf) · Built by Haseeb U Rehman

---

https://github.com/user-attachments/assets/facc0fcd-a989-434a-b4a6-7f8413a321fd

## The problem

NestHaven is a home goods e-commerce store carrying 400+ SKUs. Like most growing stores, inventory decisions were made on gut feel and spreadsheets — and it was costing real money in two directions at once:

- **Stockouts on bestsellers.** Last Eid season, a duvet set sold out 6 days before the holiday peak. 23 orders had to be refunded. Each refund is a lost sale and a damaged customer relationship.
- **Cash trapped in dead stock.** Meanwhile, slower-moving SKUs sat in the warehouse for months, tying up capital that could have funded better-selling inventory.

Neither problem is visible until it's too late — by the time a human notices low stock, the stockout has often already happened.

## The approach

A forecasting and alert system that watches every SKU continuously and tells the operator exactly what to do, in plain language, before the problem hits.

**Three components:**

1. **Demand forecasting** — an XGBoost model trained on historical sales, engineered with features specific to this market: day-of-week effects, month-end salary cycles, and the Eid ul-Fitr / Eid ul-Adha / Nov–Dec holiday windows that drive outsized demand swings in this region.
2. **Reorder logic** — classical supply-chain math (reorder point, safety stock via service-level z-score, economic order quantity) applied to the model's forecasts, producing a daily-refreshed status per SKU: **Critical**, **Warning**, **Healthy**, or **Overstock**.
3. **MLOps automation** — a nightly GitHub Actions workflow that regenerates data, retrains the model, and recalculates every alert without manual intervention, committing results straight back to the live dashboard.

## Architecture


```mermaid
flowchart LR
    A[Synthetic / Shopify<br/>order data] --> B[Data pipeline<br/>clean + aggregate]
    B --> C[Feature engineering<br/>lags, rolling stats,<br/>holiday windows]
    C --> D[XGBoost model<br/>train + evaluate]
    D --> E[30-day forecast<br/>per SKU]
    E --> F[Reorder engine<br/>ROP, safety stock, EOQ]
    F --> G[Alert classification<br/>Critical/Warning/Healthy/Overstock]
    G --> H[Streamlit dashboard]

    I[GitHub Actions<br/>nightly cron] -.triggers retrain.-> B
    I -.triggers retrain.-> D
    I -.triggers retrain.-> F
```

## The product

A Streamlit dashboard an operations manager can use without any technical background:

- Color-coded reorder alerts table, sorted by urgency
- Per-SKU forecast chart — actual sales bridging directly into a 30-day forward forecast, with the forecast window clearly shaded
- One-click manual override, so a planned promotion or known external factor can adjust the system's recommendation
- Live "last retrained" timestamp, backed by a real automated pipeline — not a static number

![Reorder alerts dashboard](./screenshots/alerts.png)
![SKU forecast detail](./screenshots/detail.png)

## The result

On synthetic data modeling NestHaven's real seasonal patterns, the system correctly:

- Flags **Scented Candle Pack** as critical 4 days before a projected stockout, with a specific recommended reorder quantity
- Identifies **$4,275** in capital currently tied up in overstocked SKUs — money that's invisible without this system
- Forecasts the post-holiday demand drop accurately, showing the model has learned real seasonal structure rather than just averaging history

The reorder engine differentiates correctly between products — a low-volume, flat-demand item is never treated the same as a high-velocity bestseller, which is what makes the alerts trustworthy rather than generic.

## Why it's built this way

A few decisions worth calling out, because they're the difference between a model that works in a notebook and a system a business can actually run on:

- **One global XGBoost model, not per-SKU models.** With a limited number of SKUs, cross-product learning (how *any* product behaves during Eid) outperforms isolated per-SKU models that don't have enough individual history.
- **SMAPE over MAPE.** Standard MAPE breaks (returns infinite error) on the very common case of a day with zero sales. SMAPE handles it correctly — important for any retail dataset with intermittent demand.
- **Time-based train/test split, never random.** A forecasting model must be validated the way it will actually be used — predicting forward from a point in time, never peeking at the future.
- **No data leakage features.** Early iterations included `revenue` and `current_stock` as model inputs — both quietly leak the answer back into the prediction. Removing them dropped a misleadingly high accuracy number to an honest one, and fixed per-SKU forecasts that had been collapsing toward a meaningless average.

## Model performance vs. baseline


It's easy to claim a model "works" without showing what it's being compared against. Here, the XGBoost model is benchmarked against the simplest reasonable baseline — predicting that demand on any given day equals demand exactly 7 days prior (captures weekly seasonality with zero modeling effort):

| Model                            | MAE (units) | SMAPE  |
|-----------------------------------|------------:|-------:|
| Naive baseline (same as 7d ago)  | 6.87        | 49.19% |
| XGBoost (this project)           | 3.44        | 29.76% |

**The XGBoost model cuts forecast error by 39.5% relative to the naive baseline** — roughly half the mean absolute error (6.87 → 3.44 units) and a 19-point drop in SMAPE. This is the test that actually matters: a model that doesn't clearly beat "look at last week" isn't worth the added complexity, infrastructure, and retraining cost. This one does.

## Tech stack

`Python` · `XGBoost` · `pandas` / `NumPy` · `Streamlit` · `Plotly` · `GitHub Actions` (scheduled retraining) · `Streamlit Community Cloud` (deployment)

## Project structure

```
├── src/
│   ├── extract.py                    # data extraction
│   ├── transform.py                  # cleaning, aggregation, feature engineering
│   ├── synthetic_data_generator.py   # seasonally-realistic demand simulation
│   ├── model_train.py                # XGBoost training, evaluation, forecasting
│   └── forecasting_feature.py        # reorder point / safety stock / alert engine
├── tests/
│   └── test_reorder_logic.py         # unit tests for reorder math
├── app.py                            # Streamlit dashboard
├── .github/workflows/
│   ├── nightly_retrain.yml           # automated MLOps pipeline
│   └── tests.yml                     # CI test runner
├── data/
└── models/
```

## Limitations


Worth being upfront about, since these are exactly the things a real deployment would need to validate first:

- **Synthetic data, not live transactions.** The seasonal patterns (Eid windows, month-end pay cycles) are engineered to be realistic, but they're authored assumptions, not observed behavior. A real Shopify connection would likely surface demand drivers (marketing pushes, competitor stockouts, weather) that aren't in this simulation at all.
- **No exogenous signals.** The model forecasts from sales history alone — it doesn't see price changes, promotions, or marketing spend as inputs. A real deployment would need those wired in, since a promo can spike demand in a way no amount of historical lag features can predict.
- **Cold-start SKUs are unsolved.** New products with little or no sales history would get poor forecasts from this model — it leans on lag and rolling-window features that don't exist yet for a brand-new item. A production version would need a separate cold-start strategy (category averages, manual overrides until enough history accumulates).
- **EOQ assumes fixed cost inputs.** Order cost ($20) and holding cost (15% of price) are reasonable defaults, not values pulled from NestHaven's actual accounting — a real client would need to supply their true costs for the recommended order quantities to be trustworthy.
- **SMAPE of ~30% is good relative to baseline, not perfect in absolute terms.** It's a meaningful improvement, but it's not a number you'd want to over-promise on a sales contract without first validating on a full real season of data.

## What this would look like for a real client

This project is built on realistic synthetic data engineered to match the seasonal patterns of an actual e-commerce business. Connected to a live Shopify store, the same pipeline runs unchanged — the `ShopifyConnector` module already handles paginated order/product extraction via the Admin API, ready to swap in real credentials.

---

*Interested in a forecasting and inventory system for your own store? [Get in touch](mailto:haseeb@email.com).*
