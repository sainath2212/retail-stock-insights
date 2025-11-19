# Literature Review Summary

---

## A. Market Basket Analysis and Association Rule Mining (H9, H10)

Market Basket Analysis (MBA) is the foundational technique for identifying product relationships.  
The **Apriori algorithm** is the classic approach for discovering frequent itemsets and generating association rules, using metrics such as **Support**, **Confidence**, and **Lift**. Lift measures how much more likely item Y is purchased given item X, relative to their independent probabilities.

The **FP-Growth algorithm** offers a more memory- and time-efficient alternative to Apriori.

These techniques directly support **H9** (identifying frequently co-purchased products) and **H10** (assessing cross-selling opportunities).

---

## B. Customer Segmentation and RFM Modeling (H7)

The **Recency, Frequency, and Monetary (RFM)** model is the widely accepted standard for quantifying customer value and purchasing behavior.

- **Recency:** Time since the last purchase  
- **Frequency:** Total number of purchases  
- **Monetary:** Total spending  

**K-Means Clustering** is the dominant method used to segment customers based on normalized RFM scores.

To determine the optimal number of clusters, standard evaluation techniques such as the **Elbow Method** and **Silhouette Score** are applied.

This methodology supports **H7**, which focuses on identifying distinct customer groups with different purchasing behaviors.

---

## C. Predictive Modeling and Time-Series Analysis (H11, H12)

Traditional **basic time-series models** serve as strong baseline approaches for forecasting sales.

Ensemble Machine Learning models such as **XGBoost** and **Random Forest** commonly demonstrate superior performance in forecasting tasks.

For interpretability, **feature importance techniques** from tree-based models help explain predictions in business-understandable terms.

These approaches support:  
- **H11:** Identifying the strongest predictors of order value  
- **H12:** Comparing ensemble ML models with basic time-series models  

---

## D. Retail Benchmarks and Empirical Principles (H3, H4)

The **Pareto Principle (80/20 rule)** is frequently validated in retail datasets, suggesting that approximately **20% of products account for 80% of total revenue**, directly informing **H3**.

Seasonality effects, including the **holiday shopping effect**, are well-documented in retail behavior studies. These recurring temporal cycles support expectations for **H4**.

---

## References

[1] Agrawal, R., and Srikant, R., “Fast algorithms for mining association rules,” *Proc. 20th Int. Conf. Very Large Data Bases*, 1994, pp. 487–499.  

[2] Han, J., Pei, J., and Yin, Y., “Mining frequent patterns without candidate generation,” *ACM SIGMOD Rec.*, vol. 29, no. 2, pp. 1–12, 2000.  

[3] Dixon, C., and Wilkinson, N., *Marketing Analytics: Data Mining Techniques for Better Marketing Decisions*. Butterworth-Heinemann, 2011.
