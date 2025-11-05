# Retail Stock Insights

**Retail Stock Insights** is a data-driven exploration of customer purchasing patterns and retail market behavior using the **UCI Online Retail Dataset (December 2010 – December 2011)**.
The project aims to uncover what drives revenue, when customers buy, who the key customer groups are, and where growth opportunities exist through advanced data mining, machine learning, and visual analytics.

---

## Project Overview

This project analyzes transactional data from a UK-based online retail store to discover patterns in customer behavior, product performance, and market trends.

It explores how factors such as product category, pricing, time, customer type, and geography influence revenue and purchasing decisions.
By applying association rule mining, clustering, and predictive modeling, the project generates actionable insights to support business growth and customer engagement.

---

## Objectives

### 1. Identify Key Revenue Drivers

Determine which product characteristics (price, category, or description patterns) most strongly correlate with higher sales, order values, and repeat purchases.

### 2. Analyze Temporal Market Dynamics

Understand how seasonality, day-of-week, and time-of-day patterns influence sales volume and revenue.

### 3. Develop Customer Segmentation

Use clustering techniques to group customers based on purchasing frequency, basket size, and spending behavior.

### 4. Apply Association Rule Mining

Identify frequently co-purchased products to uncover cross-selling and bundling opportunities.

### 5. Visualize Trends

Build interactive dashboards to explore country-wise, time-based, and customer-segmented sales patterns.

### 6. Predict Future Behavior

Use machine learning models to forecast sales, customer lifetime value, and churn risk.

---

## Research Questions

### A. Product Popularity and Sales Dynamics – "What Drives Revenue?"

1. **Product Category Performance:** Which product categories generate the most revenue, and how do these vary across seasons, countries, and segments?
2. **Price Sensitivity:** How does price affect purchase quantity and demand? Are bulk purchases or premium products distributed differently?
3. **High-Value Transactions:** Which product or pricing features predict high-value orders (for example, orders greater than £100)?

---

### B. Temporal Patterns and Seasonality – "When Do People Buy?"

4. **Seasonal Purchase Trends:** How do sales vary across seasons or holidays, and which products are most seasonal?
5. **Day and Hour Effects:** Do certain days or times of day show higher transaction counts or order values?
6. **Forecasting Trends:** Can models such as ARIMA or Prophet accurately predict future sales patterns?

---

### C. Customer Segmentation and Behavior – "Who Are the Customers?"

7. **Customer Clustering:** Can we identify distinct buyer types such as frequent buyers, seasonal shoppers, or bargain hunters?
8. **Lifetime Value Prediction:** Which factors best predict customer lifetime value or inactivity risk?
9. **Geographic Variations:** How do purchasing behaviors differ across countries?

---

### D. Association Rule Mining and Cross-Selling – "What Gets Bought Together?"

10. **Frequent Product Pairs:** Which products are most often purchased together?
11. **Cross-Selling Opportunities:** Which complementary or substitutable items can be bundled to increase sales?
12. **Rule Lift and Impact:** Which association rules show the strongest lift and highest potential business value?

---

### E. Predictive Modeling and Interpretability – "How to Predict?"

13. **Feature Importance:** Which features most influence transaction value and purchase frequency?
14. **Model Comparison:** How do tree-based models (such as Random Forest or XGBoost) compare to time-series models for forecasting?
15. **Explainability:** How can SHAP or LIME explain individual predictions or uncover actionable insights?

---

### F. Geographic and Market Expansion – "Where Is Growth?"

16. **Country-Level Trends:** Which countries contribute most to revenue, and where is growth accelerating?
17. **Regional Segmentation:** Can we cluster countries into groups with similar purchasing characteristics for improved targeting?

---

## Scope and Limitations

### Scope

- Covers a 12-month retail period (December 2010 – December 2011)
- Includes all countries represented in the dataset
- Focuses on transactional and behavioral analysis
- Utilizes clustering, association rules, and forecasting models

### Limitations

- Limited to one retailer; findings may not generalize across industries
- No demographic data such as age or gender
- Profit margins and cost data unavailable (revenue = Quantity × UnitPrice)
- Excludes external market factors like promotions or economic conditions
- Findings are associative, not causal
- One-year timeframe may not capture long-term trends

---

## Tech Stack (Planned or In Use)

- **Languages:** Python (Pandas, NumPy, Scikit-learn)
- **Visualization:** Matplotlib, Seaborn
- **Modeling:** ARIMA, Prophet, XGBoost, Random Forest
- **Clustering:** K-Means, DBSCAN, Hierarchical Clustering
- **Association Rules:** Apriori, FP-Growth
- **Dashboard:** Streamlit

---

## Project Goals Summary


| Focus Area            | Core Question            | Method                         | Outcome                               |
| --------------------- | ------------------------ | ------------------------------ | ------------------------------------- |
| Product Analysis      | What drives revenue?     | Correlation and Regression     | Identification of key product drivers |
| Time Analysis         | When do people buy?      | Seasonal Decomposition         | Insights for inventory planning       |
| Customer Segmentation | Who buys what?           | Clustering (K-Means/DBSCAN)    | Targeted marketing strategies         |
| Basket Analysis       | What’s bought together? | Apriori / FP-Growth            | Cross-selling insights                |
| Predictive Modeling   | How to forecast?         | Machine Learning / Time-Series | Demand forecasting                    |

---

## Expected Deliverables

- Cleaned and well-documented dataset
- Analytical reports and visualizations
- Predictive and clustering models
- Association rule mining results
- Business recommendations and insights
- Interactive dashboard for exploration

---

## Acknowledgments

Dataset sourced from the **UCI Machine Learning Repository**.
Special thanks to the open-source community for providing the tools and frameworks used in this analysis.
