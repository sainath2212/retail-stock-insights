# Project Hypotheses & Innovation

This document outlines the testable hypotheses for the **Retail Stock Insights** project, derived from the core research questions. Each hypothesis is presented with a corresponding null hypothesis and a proposed methodology for testing.

---

## A. Product Performance & Revenue Dynamics — "What Drives Sales?"

### H1: Price Elasticity and Purchase Behavior

**Description:** To what extent does unit price affect purchase quantity across different product categories?

**Hypothesis (H1):** There is a **statistically significant negative correlation** between `UnitPrice` and `Quantity`. Specifically, higher-priced items will show lower average purchase quantities, with elasticity varying by product category.

**Null Hypothesis (H0):** The relationship between `UnitPrice` and `Quantity` is **non-existent or not statistically significant** (p > 0.05).

**Methodology:**

* Join `online_retail.csv` data by product category (derived from Description).
* Calculate correlation coefficient (Pearson or Spearman) between `UnitPrice` and `Quantity` for the full dataset and by category.
* Use a **multiple linear regression model**: `Quantity ~ UnitPrice + Category + UnitPrice:Category` to test for interaction effects.
* Visualize the relationship using scatter plots with trend lines per category.
* Compare elasticity estimates across categories using confidence intervals.

---

### H2: High-Value Transaction Predictability

**Description:** Which combination of product attributes (unit price, category, quantity) best predicts whether a transaction will exceed a specified revenue threshold (e.g., £500)?

**Hypothesis (H1):** A **classification model (e.g., Random Forest or Logistic Regression)** can predict high-value transactions (>£500 total value) with high accuracy (**AUC > 0.75**). We hypothesize that `UnitPrice` and `Quantity` will be the strongest predictive features.

**Null Hypothesis (H0):** The model will perform **no better than a random baseline** (AUC ≈ 0.50), indicating that transaction attributes are not predictive of value.

**Methodology:**

* Create a binary target variable: `high_value_transaction` (1 if `Quantity × UnitPrice > £500`, 0 otherwise).
* Engineer features: `UnitPrice`, `Quantity`, product category (from Description), day-of-week, month, season.
* Train a **Random Forest classifier** and **Logistic Regression model** using k-fold cross-validation.
* Evaluate using **AUC-ROC**, **Precision**, **Recall**, and **F1-score** (to handle potential class imbalance).
* Extract and visualize **feature importances** to identify strongest predictors.

---

### H3: Product Category Revenue Concentration

**Description:** Are certain product categories responsible for a disproportionate share of total revenue?

**Hypothesis (H1):** Approximately **20% of product categories** will generate **80% of total revenue** (Pareto principle), indicating strong revenue concentration.

**Null Hypothesis (H0):** Revenue is **evenly distributed** across product categories, with each category contributing proportionally to its product count.

**Methodology:**

* Extract product category from item descriptions using keyword matching or topic modeling.
* Calculate total revenue per category: `Revenue = SUM(Quantity × UnitPrice)` by category.
* Rank categories by revenue contribution and calculate cumulative percentage.
* Plot a **Pareto curve** showing the percentage of categories vs. cumulative revenue percentage.
* Identify and document the top 20% revenue-generating categories.

---

## B. Temporal Patterns & Seasonality — "When Do People Buy?"

### H4: Seasonal Purchase Behavior

**Description:** Is there a statistically significant seasonal effect on purchase volume and revenue across quarters or months?

**Hypothesis (H1):** There is a **statistically significant seasonal pattern** in both purchase volume and revenue. Specifically, we hypothesize that **Q4 (Oct-Dec) will show peak sales** due to holiday shopping, while **Feb-Mar will show lower sales** (post-holiday slump).

**Null Hypothesis (H0):** Monthly and quarterly sales volumes are **random or show no consistent pattern**; seasonality is negligible.

**Methodology:**

* Extract month and quarter from `InvoiceDate`.
* Aggregate `Quantity` and revenue (`Quantity × UnitPrice`) by month and quarter.
* Perform a **seasonal decomposition** (e.g., STL decomposition) to isolate seasonal, trend, and residual components.
* Use **ANOVA** to test for significant differences in mean sales across months or quarters.
* Visualize using **time-series plots** and **heatmaps** showing purchase patterns by month and day-of-week.
* Fit a **SARIMA or Prophet model** to confirm and forecast the seasonal pattern.

---

### H5: Day-of-Week Effects

**Description:** Do certain days of the week (e.g., weekends vs. weekdays) exhibit significantly different purchasing patterns?

**Hypothesis (H1):** There is a **statistically significant difference** in mean transaction value and purchase frequency between weekdays and weekends. We hypothesize that **weekdays will show higher average transaction value**, possibly reflecting business purchases, while **weekends will show higher transaction frequency**.

**Null Hypothesis (H0):** Day-of-week has **no statistically significant effect** on transaction characteristics; patterns are random or uniform.

**Methodology:**

* Extract day-of-week (0=Monday, 6=Sunday) from `InvoiceDate`.
* Create binary variable: `is_weekend` (1 if Saturday or Sunday, 0 otherwise).
* Aggregate metrics by day-of-week: count of transactions, average `Quantity`, average transaction value (`Quantity × UnitPrice`).
* Perform **ANOVA** or **Kruskal-Wallis test** to compare mean values across days.
* Visualize using **bar charts** and **box plots** showing distributions by day-of-week.
* Use **contrast coding** in regression to compare weekdays vs. weekends directly.

---

### H6: Intra-Day Purchase Timing

**Description:** Is there a detectable intra-day effect (time of day) on transaction frequency or average order value?

**Hypothesis (H1):** There is a **non-uniform distribution of purchases across hours of the day**. We hypothesize that **business hours (9 AM - 5 PM) will show higher transaction volume**, with a possible **peak during lunch hours (12 PM - 2 PM)** and **secondary peak in early evening**.

**Null Hypothesis (H0):** Transactions are **uniformly distributed** across all hours, or no statistically significant hourly pattern exists.

**Methodology:**

* Extract hour from `InvoiceDate` (if available in the dataset; if not, aggregate by day and note limitation).
* Count transactions and calculate average transaction value by hour.
* Perform **ANOVA** to test for significant differences in volume/value across hours.
* Visualize using **line plots** showing transaction frequency and average order value by hour.
* Use **Fourier analysis** or **spectral methods** to identify periodic patterns in intra-day purchasing.

---

## C. Customer Segmentation & Behavior — "Who Are the Customers?"

### H7: Customer Clustering by Purchase Behavior (RFM Segmentation)

**Description:** Can customers be meaningfully segmented into distinct behavioral groups based on purchasing metrics (RFM: Recency, Frequency, Monetary)?

**Hypothesis (H1):** Using **RFM-based clustering (K-means or hierarchical clustering)**, we can identify **at least 3-4 distinct customer segments** with significantly different characteristics. We hypothesize that a "High-Value Loyal" segment will have high frequency and monetary value, while a "One-Time Buyer" segment will have low frequency and moderate monetary value.

**Null Hypothesis (H0):** Customer purchase behaviors are **random or uniformly distributed**; no meaningful clustering structure exists, or the optimal number of clusters is 1.

**Methodology:**

* Calculate **RFM metrics** per customer:
  * **Recency:** Days since last purchase (relative to dataset end date).
  * **Frequency:** Number of distinct transactions by customer.
  * **Monetary:** Total spending (`SUM(Quantity × UnitPrice)`).
* Normalize RFM metrics (e.g., min-max scaling) for fair clustering.
* Apply **K-means clustering** with **elbow method** to identify optimal number of clusters (k).
* Validate clustering using **silhouette score** and **Davies-Bouldin index**.
* Characterize each cluster by mean RFM values and other metrics (e.g., product preferences, seasonality).
* Visualize clusters using **2D scatter plots** (PCA or t-SNE) and **3D plots** (R, F, M).

---

### H8: Geographic Customer Differences

**Description:** Do customers from different countries exhibit statistically significant differences in purchasing behavior?

**Hypothesis (H1):** There is a **statistically significant difference** in mean transaction value, purchase frequency, and product preferences across the top 5 countries by transaction volume. We hypothesize that **Western European customers will show higher average order value**, while **Eastern European or Asian customers will show higher transaction frequency**.

**Null Hypothesis (H0):** Country has **no statistically significant effect** on purchasing metrics; behavior is uniform across geographies.

**Methodology:**

* Identify the **top 5-10 countries** by transaction volume or distinct customers.
* Calculate per-country metrics: mean transaction value, median frequency, average `Quantity`, category preferences.
* Perform **ANOVA** or **Kruskal-Wallis test** to compare mean transaction values across countries.
* Use **chi-square test** to compare product category distributions across countries.
* Visualize using **violin plots**, **heatmaps** (country × category), and **bar charts** (country metrics).
* Perform **post-hoc pairwise comparisons** (Tukey HSD) if overall ANOVA is significant.

---

## D. Association Rule Mining — "What Gets Bought Together?"

### H9: Frequent Product Co-Purchases

**Description:** Which product pairs have the highest support, confidence, and lift in the dataset?

**Hypothesis (H1):****Strong association rules exist** between certain product categories (e.g., candles + matches, decorations + packaging materials) with **confidence > 0.3 and lift > 1.5**, indicating non-random co-purchase patterns.

**Null Hypothesis (H0):** Product associations are **random or uniform**; no rules with lift > 1 exist, indicating independent purchasing.

**Methodology:**

* Extract product categories from descriptions using **keyword extraction** or **topic modeling**.
* Create a **transaction-item matrix** where each row is an invoice and columns are product categories (binary: purchased or not).
* Apply **Apriori algorithm** with minimum support (e.g., 5%) and confidence (e.g., 20%) thresholds.
* Calculate **lift** for each rule: `Lift = P(A and B) / (P(A) × P(B))`.
* Visualize top rules using **network diagrams** (nodes = categories, edges = association rules) or **bar charts** (ranked by lift).
* Segment rules by customer segment or season to identify context-specific associations.

---

### H10: Cross-Selling Revenue Impact

**Description:** Can strategic product bundling based on association rules increase average transaction value?

**Hypothesis (H1):** Recommending **high-lift product bundles** to customers will increase average transaction value by **at least 10-15%** compared to baseline transactions without recommendations.

**Null Hypothesis (H0):** Product recommendations have **no effect** on transaction value; bundling strategies do not influence purchasing.

**Methodology:**

* Identify top association rules (high confidence and lift) from H9.
* Simulate a recommendation system: flag transactions containing one item in the rule and assume the second item is added (at average quantity and price).
* Calculate the **uplift in transaction value** for recommended bundles.
* Perform a **sensitivity analysis** varying recommendation thresholds (confidence, lift) and observing revenue impact.
* Estimate potential revenue increase if recommendations are implemented.
* Document **top 20 bundling recommendations** with expected ROI.

---

## E. Predictive Modeling & Interpretability — "How to Predict?"

### H11: Feature Importance for Revenue Prediction

**Description:** Which features (product category, temporal, customer) most strongly contribute to predicting transaction revenue?

**Hypothesis (H1):** In a Random Forest regression model predicting transaction revenue, **`Quantity` and `UnitPrice` will be the top 2 features**, accounting for > 50% of feature importance. Temporal features (day-of-week, season) will account for 10-20%.

**Null Hypothesis (H0):** All features contribute **equally or negligibly** to the model; feature importances are not meaningfully different.

**Methodology:**

* Engineer features: `Quantity`, `UnitPrice`, product category, day-of-week, month, season, customer segment (from RFM clustering).
* Train a **Random Forest Regressor** on transaction revenue using k-fold cross-validation.
* Extract **feature importances** using Gini/impurity-based importance.
* Create a **SHAP summary plot** and **dependence plots** to show marginal contributions.
* Visualize top 10-15 features with importance scores.
* Validate using a **permutation importance** method for robustness.

---

### H12: Model Comparison for Demand Forecasting

**Description:** Which machine learning model (time-series vs. machine learning) best predicts future purchase volume by product category?

**Hypothesis (H1):** An **ensemble method (XGBoost or LightGBM)** with engineered temporal and categorical features will **outperform a traditional time-series model (ARIMA)** in Mean Absolute Percentage Error (MAPE), achieving **< 10% MAPE** vs. ARIMA's **> 15% MAPE**.

**Null Hypothesis (H0):** All models have **statistically similar performance**; MAPE differences are not significant.

**Methodology:**

* Aggregate purchase volume by product category and week or month.
* Split data into **training (80%)** and **test (20%)** sets.
* Train three models:
  * **ARIMA** with optimal (p,d,q) parameters.
  * **Prophet** (Facebook's time-series forecasting library).
  * **XGBoost** with lagged features, seasonal dummies, and category embeddings.
* Evaluate using **MAPE**, **RMSE**, **MAE** on the test set.
* Perform **Diebold-Mariano test** to compare forecast accuracy statistically.
* Visualize **actual vs. predicted** for each model.

---

### H13: Customer Lifetime Value Prediction

**Description:** Can we predict customer lifetime value (CLV) using transactional features and improve targeting of high-value customers?

**Hypothesis (H1):** A **machine learning model (e.g., Gradient Boosting)** trained on early customer purchase history (first 3-6 months) can predict **future CLV (next 6 months) with R² > 0.65**, enabling actionable customer targeting.

**Null Hypothesis (H0):** Historical purchase patterns are **not predictive** of future spending; R² ≈ 0 or model performs near baseline.

**Methodology:**

* Define **CLV** as total spending in the final 6 months of the dataset.
* For each customer, calculate features from the first 6 months: RFM metrics, purchase frequency by category, average basket size, seasonal purchase patterns.
* Train a **Gradient Boosting Regressor** using k-fold cross-validation.
* Evaluate using **R²**, **RMSE**, and **Mean Absolute Percentage Error (MAPE)**.
* Identify **top 10% of predicted high-CLV customers** and characterize their traits.
* Visualize **actual vs. predicted CLV** using scatter plots and decile analysis.

---

## F. Geographic & Market Expansion — "Where Is Growth?"

### H14: Country-Level Sales Trends

**Description:** Are there statistically significant differences in total revenue and growth trends across major country markets?

**Hypothesis (H1):** The **top 5 countries by revenue will account for > 70% of total revenue**. Additionally, at least **2-3 countries will show statistically significant positive growth trends** over the 12-month period, while others remain flat or decline.

**Null Hypothesis (H0):** Revenue is **evenly distributed** across countries, and no significant growth trends exist; country variations are random.

**Methodology:**

* Aggregate revenue by country and month: `Revenue_month_country = SUM(Quantity × UnitPrice)`.
* Rank countries by total revenue and calculate cumulative percentage.
* For the top 5-10 countries, fit a **linear regression** model: `Revenue_t ~ time + Country` to detect trends.
* Perform **Theil-Sen slope estimation** (robust to outliers) for each country.
* Use **Kruskal-Wallis test** to compare median revenues across countries.
* Visualize using **bar charts** (total revenue), **time-series plots** (monthly trends per country), and **growth rate comparisons**.

---

### H15: Market Segmentation by Regional Characteristics

**Description:** Can countries be clustered into distinct market segments based on purchasing behaviors?

**Hypothesis (H1):** Using **hierarchical clustering on country-level purchase characteristics** (average order value, product category distribution, seasonality strength), we can identify **3-4 distinct market segments** (e.g., "Premium Markets," "Emerging Markets," "Niche Specialty Markets").

**Null Hypothesis (H0):** Countries are **uniformly distributed** in characteristic space; no meaningful clustering exists.

**Methodology:**

* Calculate per-country metrics: mean transaction value, median frequency per customer, product category preferences (distribution), seasonality index, customer count.
* Normalize metrics and apply **hierarchical clustering** (Ward linkage) or **K-means**.
* Use **silhouette score** and **dendrogram inspection** to determine optimal number of clusters.
* Characterize each cluster by representative metrics and dominant product categories.
* Visualize using **heatmaps**, **dendrograms**, and **2D projections** (PCA, t-SNE).

---

## 6.  Innovation & Beyond Standard Hypotheses

* **RFM Segmentation with Behavioral Extensions:** Beyond standard RFM, incorporate product category affinity and seasonality patterns to create "behavioral personas" (e.g., "Premium Seasonal Buyer").
* **Network Analysis of Product Associations:** Use graph analysis to identify clusters of highly associated products and detect "product ecosystems."
* **Anomaly Detection:** Identify unusual purchasing patterns (e.g., sudden volume spikes, geographic outliers) that may indicate data quality issues or genuine market events.
* **Causal Inference:** Employ propensity score matching or difference-in-differences analysis (if promotional data is available) to infer causal effects of price changes or category promotions on purchasing behavior.
* **Interpretable AI for Frontline Teams:** Create simplified, rule-based explanations of model predictions using SHAP or LIME, enabling non-technical stakeholders to understand recommendations.
