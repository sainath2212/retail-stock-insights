# Retail Stock Insights

**Retail Stock Insights** is a comprehensive data-driven study of customer purchasing behavior, product performance, and market dynamics using the **UCI Online Retail Dataset (Dec 2010 – Dec 2011)**.

The project is structured around six analytical dimensions, each designed to address core research questions related to revenue drivers, seasonality, customer behavior, product associations, predictive modeling, and geographic expansion.

---

## Project Overview

This project examines transactional data from a UK-based online retail store to uncover:

- What drives product revenue
- When customers make purchases and how seasonality affects demand
- How customers can be segmented based on their purchasing behavior
- Which product combinations frequently co-occur
- How accurately future sales and customer spending can be predicted
- Where geographic opportunities exist for market expansion

Through statistical analysis, machine learning, clustering, and association rule mining, the project aims to generate actionable insights for improving retail operations and strategy.

---

## Research Objectives (Aligned to Six Dimensions)

---

## **A. Product Performance & Revenue Dynamics**

1. **Pricing Influence:** Analyze how product pricing affects the volume of units sold and overall demand.  
2. **High-Value Order Prediction:** Determine whether high-value orders can be predicted using basic product metrics such as price, category, and quantity.  
3. **Pareto Principle Validation:** Evaluate whether the 80/20 rule holds for revenue distribution—i.e., whether a small number of products contribute to the majority of sales.

---

## **B. Temporal Patterns & Seasonality**

4. **Seasonal Impact:** Quantify the effect of the Christmas season on sales volume and product demand.  
5. **Weekday vs. Weekend Behavior:** Examine differences in customer purchasing habits between weekdays and weekends.  
6. **Time-of-Day Clustering:** Identify whether transaction activity clusters around particular hours.

---

## **C. Customer Segmentation & Behavior**

7. **RFM Segmentation:** Group customers into distinct clusters using RFM (Recency, Frequency, Monetary) analysis.  
8. **Country-Based Differences:** Compare purchasing behavior, including average order value and product preferences, across different countries.

---

## **D. Association Rule Mining**

9. **Frequent Product Combinations:** Identify product sets that are often purchased together within the same transaction.  
10. **Revenue Uplift Estimation:** Estimate potential revenue improvements through cross-selling and targeted product recommendations.

---

## **E. Predictive Modeling & Interpretability**

11. **Key Predictive Features:** Determine which transactional features (price, quantity, category) are the strongest predictors of order value.  
12. **Model Performance Comparison:** Compare the performance of ensemble machine learning models with basic time-series forecasting models.  
13. **Customer Spending Prediction:** Evaluate whether past purchasing behavior can accurately forecast future spending.

---

## **F. Geographic & Market Expansion**

14. **Sales Concentration:** Identify whether total sales volume is concentrated in a few key countries.  
15. **Market Type Classification:** Cluster countries into market types (e.g., Premium, Emerging) based on purchasing metrics.

---

## Scope and Limitations

### **Scope**
- Covers a full 12-month retail period (Dec 2010 – Dec 2011)  
- Includes customers and orders from all available countries  
- Focuses on transactional patterns, customer behavior, and predictive modeling  
- Utilizes association rules, clustering, and forecasting techniques  

### **Limitations**
- Dataset represents only one retailer  
- No demographic attributes such as age or gender  
- Revenue = Quantity × UnitPrice (profit margins unavailable)  
- No promotional, marketing, or external economic data  
- One-year period limits long-term trend analysis  

---

## Tech Stack

- **Languages:** Python (Pandas, NumPy, Scikit-Learn)  
- **Visualization:** Matplotlib, Seaborn  
- **Modeling:** Random Forest, XGBoost
- **Clustering:** K-Means, DBSCAN, Hierarchical Clustering  
- **Association Rules:** Apriori, FP-Growth  
- **Dashboarding:** Streamlit  

---

## Project Summary Table

| Dimension | Core Question | Method | Expected Output |
|----------|---------------|--------|-----------------|
| Product Performance | How does pricing influence sales? | Correlation, Regression | Key revenue drivers |
| Temporal Analysis | When do people buy? | Seasonal Analysis | Insights for staffing & inventory |
| Customer Segmentation | Who buys and how? | RFM, Clustering | Defined customer groups |
| Basket Analysis | What products are bought together? | Apriori / FP-Growth | Cross-selling recommendations |
| Predictive Modeling | Can we predict value or sales? | ML + Time-Series | Demand forecasting |
| Geography | Where are the best markets? | Clustering, Trend Analysis | Market-type classification |

---

## Deliverables

- Cleaned and processed dataset  
- Analytical reports with visualizations  
- Customer and country clusters  
- Basket analysis and association rules  
- Predictive models and interpretability outputs  
- Interactive Streamlit dashboard  
- Strategic recommendations for retail growth  

---

## Acknowledgments

Dataset sourced from the **UCI Machine Learning Repository**.  
Thanks to open-source contributors for tools and frameworks used in this analysis.
