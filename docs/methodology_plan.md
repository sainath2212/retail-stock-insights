### Methodology Plan  
### Retail Stock Market Behavior Project

This document presents the comprehensive methodology framework for the *Retail Stock Market Behavior* project.  
It integrates the literature foundation, data preprocessing, exploratory analysis, visualization, and modeling strategies — aligned with the three project phases and respective team leadership responsibilities.

---

## 1. Literature Review Summary (Phase 1)
The literature review establishes the theoretical basis and industry best practices guiding retail analytics and predictive modeling.

- **Market Basket Analysis:**  
  Explore **Apriori** and **FP-Growth** algorithms for association rule mining (H9). Determine optimal thresholds for *support*, *confidence*, and *lift* to balance precision and coverage.

- **Customer Segmentation:**  
  Adopt the **RFM (Recency, Frequency, Monetary)** framework (H7) for segmentation. Review the effectiveness of **K-Means Clustering** on RFM scores and methods to evaluate segmentation quality.

- **Predictive Modeling in Retail:**  
  Examine feature engineering techniques for sales and value prediction (H2, H11). Compare **time-series models (ARIMA)** with **ensemble learning models (XGBoost, Random Forest)** for forecasting (H12).

- **Retail Trend Validation:**  
  Review empirical evidence for the **Pareto Principle (80/20 rule)** in sales contribution (H3) and recurring **seasonality patterns** (H4).
---

## 2. Data Preprocessing Plan (Phase 1)

All preprocessing tasks will be implemented in the `data_preprocessing.ipynb` notebook to ensure clean, structured, and analysis-ready data.

### 2.1 Handling Missing Values
- **CustomerID:**  
  Rows without customer IDs (~25%) will be **excluded** from customer-level analyses but **retained** for aggregated product/sales insights.
- **Description:**  
  Rows with missing descriptions will be dropped after verifying minimal impact.

### 2.2 Outlier & Anomaly Treatment
- **Negative Quantity:**  
  Identify returns/cancellations; process separately for net sales and exclude from Association Rule Mining.
- **Zero/Extreme UnitPrice:**  
  Remove zero or implausible `UnitPrice` values indicating data entry errors.

### 2.3 Feature Engineering
- Create `TotalSales` = `Quantity × UnitPrice`.  
- Extract temporal features (`Year`, `Month`, `DayOfWeek`, `HourOfDay`) from `InvoiceDate` (H4–H6).  
- Compute RFM metrics — Recency, Frequency, and Monetary — per `CustomerID` (H7).

---

## 3. Exploratory Data Analysis (Phase 2)
**Lead:** A. Jithendranath  

The EDA phase aims to uncover early insights and validate hypotheses through descriptive and visual exploration.

| Focus Area | Supported Hypotheses | Analytical Techniques / Expected Output |
| :---------- | :------------------- | :-------------------------------------- |
| Product Performance | H1, H3 | Analyze `UnitPrice`–`Quantity` trends. Compute cumulative revenue to validate 80/20 rule. |
| Temporal Patterns | H4, H5, H6 | Aggregate sales by Month, Day, Hour to reveal seasonality and peak hours. |
| Geographical Insights | H8, H14 | Rank countries by sales and order value. Compare purchase behavior across markets. |
| Customer Distribution (RFM) | H7 | Visualize R, F, and M score distributions before clustering. |

---

## 4. Visualization Plan (Phase 2 & 3)

Interactive and analytical visualizations will be developed using **Plotly**, later integrated into a **Streamlit/Dash dashboard** for the final phase.

| Phase | Visualization Type | Purpose / Insight |
| :----- | :----------------- | :---------------- |
| Phase 2 | Time Series (Sales by Month) | Show seasonal peaks and long-term sales trends (H4). |
| Phase 2 | Scatter (`UnitPrice` vs. `Quantity`) | Examine inverse relation between price and units sold (H1). |
| Phase 2 | Choropleth / Ranked Bar Chart | Highlight sales distribution across countries (H14). |
| Phase 3 | RFM Scatter Plot | Visualize customer clusters formed via K-Means (H7). |
| Phase 3 | Network Graph / Heatmap | Display top association rules (H9). |
| Phase 3 | Actual vs. Predicted Plot | Evaluate forecasting accuracy (H12). |
| Phase 3 | Feature Importance / SHAP Plot | Identify key drivers influencing order value (H11). |

---

## 5. Model Training & Evaluation Plan (Phase 3)

**Lead:** M. Sree Sai Nath  

This phase focuses on segmentation, association mining, and predictive modeling to generate actionable insights.

### A. Segmentation & Association Mining

1. **Customer Segmentation (H7):**  
   - Apply **K-Means** on normalized RFM scores.  
   - Determine optimal *k* using the **Elbow Method** and **Silhouette Score**.

2. **Market Basket Analysis (H9, H10):**  
   - Use **Apriori** algorithm on cleaned transactions (excluding returns).  
   - Prioritize rules with high **Lift** and **Confidence** for recommendation systems.

3. **Market Classification (H15):**  
   - Apply **K-Means** to country-level aggregates (Avg. Order Value, Frequency, Preferences) to define market types.

---

### B. Predictive Modeling

| Prediction Task | Hypothesis | Algorithm(s) | Evaluation Metrics |
| :--------------- | :----------- | :------------- | :------------------ |
| Big Order Classification | H2 | XGBoost Classifier | F1-Score, ROC AUC, Confusion Matrix |
| Sales Forecasting | H12 | SARIMA (baseline), Gradient Boosting Regressor | RMSE |
| Feature Influence (Order Value Drivers) | H11 | Random Forest Regressor | SHAP, Feature Importance |

---

### C. Validation & Explainability

- **Cross-Validation:**  
  Use *k*-fold cross-validation for supervised models to ensure generalization.  
- **Hyperparameter Tuning:**  
  Employ **Grid Search** or **Randomized Search** for model optimization.  
- **Interpretability:**  
  Utilize **SHAP values** and **feature importance** rankings to explain model behavior in business terms.

---

## 6. Deliverables by Phase Summary

| Phase | Focus Area | Primary Output |
| :----- | :----------- | :-------------- |
| Phase 1 | Literature Review & Preprocessing | Clean dataset, RFM features, theoretical foundation |
| Phase 2 | EDA & Visualization | Validated hypotheses, trend insights |
| Phase 3 | Modeling & Evaluation | Customer segments, association rules, predictive models |

---