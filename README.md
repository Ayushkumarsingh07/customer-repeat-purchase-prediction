# Customer Repeat-Purchase Predictor

An end-to-end machine learning project predicting whether a first-time
e-commerce customer will make a second purchase — built on 56,000+ real
orders, from raw multi-table data through EDA, modeling, and a deployed,
interpretable web app.

**[Live App](#)** · **[Demo Screenshot](#)** *(add links once deployed)*

---

## Problem

Most e-commerce churn projects assume a "churned customer" label already
exists. This dataset doesn't have one — like most real businesses. So the
first job here wasn't modeling, it was **defining the right problem**:
EDA revealed that ~97% of customers in this marketplace are one-time
buyers, which reframed the entire project from generic "churn prediction"
into a more honest and business-relevant question:

> **Will this customer make a second purchase, based only on what we know
> right after their first order?**

## Why Brazilian data for an India-relevant project?

This project targets Indian e-commerce roles, but uses the Olist Brazilian
E-Commerce dataset rather than an Indian one. That's a deliberate,
documented choice: no public Indian e-commerce dataset matches Olist's
transactional depth (real delivery timestamps, installment payments,
review scores, multi-table relational structure). Available Indian
alternatives were either synthetic (generator-created, not real behavior)
or small attitude surveys, not order-level transactional data.

The feature engineering and modeling approach here transfer directly to
Indian platforms (Flipkart, Myntra, Meesho), which show comparable
patterns — COD/installment-heavy payments, regional logistics variance,
and low first-time repeat rates. The deployed app displays values in ₹
for readability, with an explicit note on this substitution rather than
silently relabeling Brazilian states as Indian ones.

## Key Findings

- **3.97% repeat-purchase rate** (within a fair 180-day comparison
  window) — a ~24:1 class imbalance that shaped every modeling decision
- **Product category is the strongest driver**: ~5x spread between best
  (`home_appliances`, 9.0%) and worst (`consoles_games`, 1.9%) categories,
  reflecting natural differences in repurchase cycles
- **Delivery delay matters**, but only after correcting for a real
  survivorship-bias confound found during EDA (see *Methodology Notes*)
- **Price has a weak negative relationship** with repeat purchase —
  cheaper first orders repeat slightly more often
- **Review score is a weaker signal than expected** — only 5-star reviews
  show a clear edge; 1–4 star reviewers repeat at similar rates

## Methodology Notes (the part that matters most)

A few decisions worth knowing about, because they're the actual signal of
analytical maturity in this project, not the charts themselves:

1. **Customer ID grain mismatch**: the dataset's `customer_id` is unique
   *per order*, not per person — `customer_unique_id` is the real
   person-level key. Using the wrong one would have made every repeat
   customer look like a stranger each time.
2. **Survivorship bias in delivery-delay analysis**: an early pass showed
   *early* deliveries had the *highest* repeat rate — counter to
   intuition. Investigation revealed this was confounded by purchase
   timing: "early delivery" orders happened to cluster earlier in the
   dataset's timeline, giving those customers more elapsed time to
   repeat. Restricting to a fair 180-day comparison window reversed the
   result to the expected direction.
3. **Data quality artifacts**: non-integer review scores (e.g., 4.5) were
   identified as averaging artifacts from multi-item orders with
   differing review values, based on tiny (3–21 order) subgroups — these
   were rounded rather than treated as real signal.
4. **Modeling ceiling**: XGBoost barely outperformed Logistic Regression
   (ROC-AUC 0.602 vs. 0.605), and target encoding vs. one-hot encoding for
   category, plus adding customer state, moved performance only
   marginally (final ROC-AUC ≈ 0.61). This was treated as a genuine
   finding, not a failure: **first-order data alone has a real,
   structural ceiling** for predicting repeat-purchase behavior, since
   the strongest drivers (life circumstances, competing offers, ongoing
   need) aren't captured by a single transaction's metadata.

## Project Structure

```
churn-prediction/
├── data/
│   ├── raw/                  # Original Olist CSVs (not committed — see Data Source)
│   └── processed/
│       ├── features_phase1.csv
│       └── features_fair.csv # EDA-ready, time-window-filtered dataset
├── notebooks/
│   └── 01_eda_and_modeling.ipynb
├── app/
│   ├── streamlit_app.py      # Deployed prediction app
│   ├── requirements.txt
│   ├── churn_model.pkl
│   ├── category_encoding.json
│   ├── state_encoding.json
│   ├── global_mean.json
│   └── feature_columns.json
└── README.md
```

## Tech Stack

Python · Pandas · Scikit-learn · XGBoost · SHAP · Streamlit

## Model

- **Algorithm**: XGBoost Classifier with `scale_pos_weight` for class
  imbalance
- **Features**: order economics (price, freight, installments), delivery
  performance (delay, speed, approval time), behavioral (purchase
  timing), satisfaction (review score), and target-encoded product
  category + customer state
- **Performance**: ROC-AUC ≈ 0.61, PR-AUC ≈ 0.066 (vs. ~0.04 baseline
  given 3.97% positive rate)
- **Interpretability**: SHAP values explain individual predictions in
  the deployed app

## Data Source

[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle) —
100,000+ real orders, 2016–2018.

## Running Locally

```bash
# Clone and set up
git clone <your-repo-url>
cd churn-prediction/app
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

## Future Work

- Incorporate customer browsing/session data, if available, to push past
  the first-order-only feature ceiling
- Multi-touch attribution: marketing emails, retargeting, and promo
  exposure between orders
- Re-validate the approach on real Indian e-commerce data if/when a
  comparable public dataset becomes available
