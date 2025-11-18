# Literature Review Summary

---

## A. Market Basket Analysis and Association Rule Mining (H9, H10)

Market Basket Analysis (MBA) is the foundational technique for identifying product relationships and understanding co-purchase behavior.

* **Apriori Algorithm:** A classic method for discovering frequent itemsets and generating association rules using **Support**, **Confidence**, and **Lift**. Lift is especially critical as it measures how much more likely item Y is purchased when item X is bought, compared to independent probabilities.
* **FP-Growth:** A more memory-efficient and time-efficient alternative to Apriori, enabling scalable frequent pattern mining.
* **Relevance to Hypotheses:** These techniques directly support **H9** (identifying frequently co-purchased products) and **H10** (cross-selling revenue uplift estimation).

### Key Research Papers

* **Agrawal, R., & Srikant, R. (1994).** *Fast algorithms for mining association rules* — foundational work introducing the Apriori algorithm.
* **Han, J., Pei, J., & Yin, Y. (2000).** *Mining frequent patterns without candidate generation* — introduced the FP-Growth algorithm.

---

## B. Customer Segmentation and RFM Modeling (H7)

The RFM (Recency, Frequency, Monetary) framework is the established standard for quantifying customer value and purchasing behavior.

* **RFM Model:**  
  - **Recency:** Time since last purchase  
  - **Frequency:** Number of purchases  
  - **Monetary:** Total spending  
* **Clustering Approach:** **K-Means Clustering** is the primary technique used to segment customers based on normalized RFM scores.  
  - The **Elbow Method** and **Silhouette Score** are standard for determining the optimal number of clusters.
* **Relevance to Hypotheses:** This approach directly supports **H7**, which focuses on identifying distinct customer groups with differing behavioral characteristics.

### Key Research Papers

* **Hughes, A. M. (1994).** *Strategic database marketing* — popularized the RFM model.  
* **Bhattacharjee, M., & Varghese, K. (2018).** *Customer segmentation using RFM model and K-Means clustering* — modern application of clustering on RFM features.

---

## C. Predictive Modeling and Time-Series Analysis (H4, H11, H12)

Predictive analytics literature provides evidence for selecting robust baseline and advanced models for forecasting sales and understanding key drivers of purchasing behavior.

* **Time-Series Baselines:** Classical models like **ARIMA** and **SARIMA** consistently perform well as baseline predictors for sales and demand trends.
* **Ensemble Models:** **Random Forest** and **XGBoost** often outperform traditional time-series models in forecasting and are widely used for predictive tasks.
* **Interpretability:**  
  - **Feature Importance** explains which variables influence predictions most.  
  - **SHAP values** provide granular, business-friendly interpretability for tree-based models.
* **Relevance to Hypotheses:**  
  - **H4:** Seasonality effects and temporal forecasting  
  - **H11:** Key predictors influencing order value  
  - **H12:** Comparing ML models with classical forecasting models

### Key Research Papers

* **Box, G. E., Jenkins, G. M., & Reinsel, G. C. (2008).** *Time series analysis: Forecasting and control* — definitive reference on ARIMA models.  
* **Lundberg, S. M., & Lee, S.-I. (2017).** *A unified approach to interpreting model predictions* — introduces the SHAP interpretability method.

---

## D. Retail Benchmarks and Empirical Principles (H3, H4)

Retail analytics literature frequently validates several empirical rules and behavioral trends relevant to this project.

* **Pareto Principle (80/20 Rule):** Commonly observed in retail datasets — around **20% of products generate 80% of revenue**, directly informing **H3**.
* **Seasonality & Holiday Effects:** Temporal cycles such as the **Christmas shopping peak** are well-documented, supporting analyses for **H4**.
* **Behavioral Economics:** Customer recency and frequency patterns play a major role in predicting future purchases.

### Key Research Papers

* **Dixon, C., & Wilkinson, N. (2011).** *Marketing analytics: Data mining techniques for better marketing decisions* — discusses Pareto behavior in retail data.  
* **Gönül, F., Carter, F., & Fader, P. (2000).** *Forecasting advertising effectiveness using household-level data* — validates the importance of recency and frequency in customer behavior.

---
