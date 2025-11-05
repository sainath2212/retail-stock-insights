#  Literature Review Summary

---

## 1. Market Basket Analysis and Association Rule Mining (H9, H10)

The foundational literature for identifying product relationships lies in **Market Basket Analysis (MBA)**.

* **Core Concepts:** The **Apriori algorithm** is the classic approach for discovering frequent itemsets and generating association rules, using metrics like **Support**, **Confidence**, and **Lift**. Lift is the critical measure, indicating how much more likely item Y is purchased given item X, relative to their independent probabilities.
* **Relevance:** The project will implement **Apriori** or **FP-Growth** to identify strong product co-occurrences (**H9**) and form the basis for simulated product recommendations (**H10**). We will prioritize filtering rules based on high Lift.

### Illustrative Research Papers:

* **Agrawal, R., & Srikant, R. (1994).** *Fast algorithms for mining association rules*. This seminal work established the Apriori algorithm.
* **Han, J., Pei, J., & Yin, Y. (2000).** *Mining frequent patterns without candidate generation*. This paper introduced the FP-Growth algorithm as a more memory and time-efficient alternative to Apriori.

---

## 2. Customer Segmentation and RFM Modeling (H7, H15)

Understanding customer heterogeneity is crucial, primarily achieved through clustering and behavioral metrics.

* **RFM Model:** The **Recency, Frequency, and Monetary (RFM)** model is the widely accepted standard for quantifying customer value and behavior.
* **Clustering:** Unsupervised learning, specifically **K-Means Clustering**, is the dominant technique for partitioning customers based on their normalized RFM scores (**H7**). Methods like the **Elbow Method** and **Silhouette Score** are standard practices for objectively determining the optimal number of segments ($k$).
* **Relevance:** The methodology will center on creating distinct customer personas using RFM and K-Means, and applying a similar clustering approach to aggregate country-level metrics for market classification (**H15**).

### Illustrative Research Papers:

* **Hughes, A. M. (1994).** *Strategic database marketing*. This work popularized the RFM technique in direct marketing applications.
* **Bhattacharjee, M., & Varghese, K. (2018).** *Customer segmentation using RFM model and K-Means clustering*. This paper highlights the modern application of K-Means to RFM scores in retail analytics.

---

## 3. Predictive and Time-Series Analysis (H4, H11, H12)

Literature guides the selection and comparison of models for forecasting and determining influential factors.

* **Sales Forecasting:** Traditional **Time-Series Models** like **ARIMA** and **SARIMA** serve as robust baselines for sales prediction (H4, H12). Ensemble Machine Learning models (e.g., XGBoost, Random Forest) are often shown to yield lower error rates, justifying the model comparison in **H12**.
* **Feature Importance & Explainability:** Regression models are essential for determining the factors that drive transactional value (**H11**). For model interpretability, methods like **SHAP (SHapley Additive exPlanations)** are preferred for tree-based models.

### Illustrative Research Papers:

* **Box, G. E., Jenkins, G. M., & Reinsel, G. C. (2008).** *Time series analysis: Forecasting and control*. This remains the definitive work on the ARIMA family of models.
* **Lundberg, S. M., & Lee, S.-I. (2017).** *A unified approach to interpreting model predictions*. This highly cited paper introduced the SHAP method for model interpretability, which will be used for feature influence analysis.

---

## 4. Retail Benchmarks and Behavioral Economics (H3)

Certain empirical rules are frequently validated in retail datasets, informing analytical expectations.

* **Pareto Principle (80/20 Rule):** Numerous studies confirm the application of the Pareto Principle in inventory and sales data, suggesting approximately **20% of products account for 80% of total revenue** (**H3**). This principle guides inventory and marketing focus.
* **Behavioral Trends:** The well-documented "holiday shopping effect" and other temporal cycles are standard expectations for seasonality (**H4**), supporting the need for time-based feature engineering.

### Illustrative Research Papers:

* **Dixon, C., & Wilkinson, N. (2011).** *Marketing analytics: Data mining techniques for better marketing decisions*. This book applies many classic data mining concepts, including the validation of the Pareto Principle, to retail data.
* **Gönül, F., Carter, F., & Fader, P. (2000).** *Forecasting advertising effectiveness using household-level data*. This paper validates the critical importance of purchase recency and frequency in predicting future behavior, concepts central to RFM.

---