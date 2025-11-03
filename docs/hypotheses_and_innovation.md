# Project Hypotheses & Innovation

This document outlines testable hypotheses for the **Retail Stock Insights** project.
Each hypothesis represents a clear and measurable assumption that will be validated through data analysis and machine learning methods.

---

## A. Product Performance & Revenue Dynamics — "What Drives Sales?"

### H1: Does Higher Price Mean Fewer Purchases?

**Hypothesis:** Higher-priced products sell in smaller quantities compared to lower-priced ones.**Testing Approach:**

- Compare unit price with quantity sold per product.
- Visualize the relationship between price and quantity.
- Analyze variation across product categories.
  **Expected Outcome:** Higher prices correlate with fewer units sold.

---

### H2: Can We Predict Big Orders?

**Hypothesis:** Product price, quantity, and category can predict whether an order exceeds £500.**Testing Approach:**

- Label orders above £500 as “big orders.”
- Train a machine learning model to classify order size.
- Measure accuracy and identify key predictive factors.
  **Expected Outcome:** Model predicts high-value orders with ≥75% accuracy.

---

### H3: Does 20% of Products Make 80% of Revenue? (Pareto Rule)

**Hypothesis:** A small fraction of products account for most revenue.**Testing Approach:**

- Rank products by total sales.
- Compute cumulative revenue contribution.
- Test Pareto distribution validity (top 20% ≈ 80% of revenue).
  **Expected Outcome:** Top 20% of products generate ~80% of total revenue.

---

## B. Temporal Patterns & Seasonality — "When Do People Buy?"

### H4: Does the Christmas Season Increase Sales?

**Hypothesis:** Sales rise during Q4 (Oct–Dec) compared to other quarters.**Testing Approach:**

- Aggregate monthly revenue.
- Visualize trends across the year.
- Identify seasonal peaks and low periods.
  **Expected Outcome:** Peak activity during the holiday season, with drops post-January.

---

### H5: Do Weekends Differ From Weekdays?

**Hypothesis:** Customer purchasing behavior varies between weekends and weekdays.**Testing Approach:**

- Classify orders as weekday or weekend.
- Compare average order values and frequency.
  **Expected Outcome:** Distinct spending or frequency patterns across days.

---

### H6: What Time of Day Do People Buy Most?

**Hypothesis:** Purchases cluster during daytime hours.**Testing Approach:**

- Extract transaction hours from timestamps.
- Aggregate order counts and values per hour.
- Identify peak shopping hours.
  **Expected Outcome:** Increased activity during working or lunch hours.

---

## C. Customer Segmentation & Behavior — "Who Are the Customers?"

### H7: Can We Group Customers Into Types?

**Hypothesis:** Customers can be grouped by behavioral traits (frequency, spending, recency).**Testing Approach:**

- Compute Recency, Frequency, and Monetary (RFM) scores.
- Perform clustering to identify customer segments.
- Characterize each cluster based on purchasing behavior.
  **Expected Outcome:** 3–4 clear customer groups with distinct spending habits.

---

### H8: Do Customers From Different Countries Shop Differently?

**Hypothesis:** Purchasing behavior differs significantly across countries.**Testing Approach:**

- Analyze top five countries by sales volume.
- Compare average order value, frequency, and product preferences.
  **Expected Outcome:** Regional variations in product preferences and order values.

---

## D. Association Rule Mining — "What Gets Bought Together?"

### H9: Which Products Are Frequently Bought Together?

**Hypothesis:** Certain product pairs co-occur in the same transaction frequently.**Testing Approach:**

- Generate association rules using market basket analysis.
- Calculate support, confidence, and lift for each pair.
  **Expected Outcome:** Identification of 10–20 high-support product combinations.

---

### H10: Can Product Recommendations Increase Sales?

**Hypothesis:** Recommending complementary products increases transaction value.**Testing Approach:**

- Simulate adding top product pairs as recommendations.
- Estimate revenue uplift based on cross-sell frequency.
  **Expected Outcome:** Recommendation strategies can raise average order value by 10–15%.

---

## E. Predictive Modeling & Interpretability — "How to Predict?"

### H11: Which Factors Most Influence Revenue?

**Hypothesis:** Price and quantity are the strongest predictors of order value.**Testing Approach:**

- Train regression and tree-based models for order value.
- Use feature importance to rank predictors.
  **Expected Outcome:** Price and quantity dominate, followed by category and timing.

---

### H12: Which Model Best Predicts Future Sales?

**Hypothesis:** Machine learning models outperform time-series baselines in predicting future sales.**Testing Approach:**

- Build ARIMA, basic regression, and ensemble ML models.
- Compare prediction errors on unseen data.
  **Expected Outcome:** ML models yield lower error rates.

---

### H13: Can We Predict Future Spending per Customer?

**Hypothesis:** Past six months’ activity predicts future spending behavior.**Testing Approach:**

- Train model on first-half spending data.
- Evaluate predictions against second-half results.
  **Expected Outcome:** Reliable identification of future high-value customers.

---

## F. Geographic & Market Expansion — "Where Is Growth?"

### H14: Which Countries Contribute Most Revenue?

**Hypothesis:** A few countries account for most of the total sales volume.**Testing Approach:**

- Calculate and rank country-level revenue.
- Analyze top contributors and monthly growth trends.
  **Expected Outcome:** Top five countries generate over 70% of revenue.

---

### H15: Can We Group Countries Into Market Types?

**Hypothesis:** Countries can be classified as “Premium,” “Emerging,” or “Niche” based on purchasing behavior.**Testing Approach:**

- Aggregate metrics per country: average order value, product preference, seasonality.
- Apply clustering to identify market categories.
  **Expected Outcome:** Discovery of 3–4 distinct market types with unique characteristics.

---

## Innovation & Extended Analysis

- **Customer Personas:** Build detailed behavioral profiles for key customer segments.
- **Product Families:** Identify clusters of products that frequently co-occur to design bundles.
- **Anomaly Detection:** Detect unusual spikes or declines in purchasing patterns.
- **Recommendation Engine:** Prototype a basic recommendation system using association rules.
- **Explainable AI:** Apply SHAP/LIME to interpret prediction results in simple business terms.

---
