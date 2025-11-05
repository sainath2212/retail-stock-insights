# Dataset Description and Rationale

This document details the chosen dataset, its fields, source, and the specific rationale for its selection for the "Retail Stock Market Behavior" data mining project.

---

## 1. Dataset Overview

* **Name:** Online Retail Dataset
* **Source:** UCI Machine Learning Repository
* **File:** `online_retail.csv`
* **Time Period:** December 2010 to December 2011
* **Description:** This is a comprehensive transactional dataset containing all the purchases made by customers of a **UK-based online retail store** over a one-year period. It provides a rich, real-world source of sales data suitable for deep analytical tasks.

---

## 2. Data Fields

The dataset comprises 8 fields critical for analyzing customer purchasing behavior and retail trends:

| Field | Description | Data Type | Relevance to Project Objectives |
| :--- | :--- | :--- | :--- |
| **InvoiceNo** | A unique identifier for each transaction (invoice). Used for grouping items into a single basket. | Nominal | **Association Rule Mining** (Market Basket Analysis) |
| **StockCode** | A unique identifier for each product/item. | Nominal | **Product Association** and **Segmentation** |
| **Description** | The name of the product. | Nominal | Data cleaning and interpretation |
| **Quantity** | The number of units of a specific product sold per transaction. | Numeric | **Volume Analysis** and calculating total sales/basket size |
| **InvoiceDate** | The date and time when the transaction occurred. | Datetime | **Temporal Analysis** (daily, weekly, seasonal trends) |
| **UnitPrice** | The unit price of the product in Sterling (£). | Numeric | Calculating total expenditure and **Customer Segmentation (RFM)** |
| **CustomerID** | A unique identifier for each customer. Crucial for customer-level analysis. | Nominal | **Clustering** and **Behavioral Segmentation** |
| **Country** | The country where the customer resides. | Nominal | **Geographical Trend Analysis** and comparing sales performance |

---

## 3. Rationale for Dataset Selection

The **UCI Online Retail Dataset** was chosen because it directly addresses the core objectives outlined in the project plan and possesses ideal characteristics for the intended data mining techniques:

### A. Suitability for Association Rule Mining
The presence of **InvoiceNo**, **StockCode**, and **Quantity** in a transaction-level format makes the dataset perfectly structured for **Market Basket Analysis (Apriori/FPGrowth)**. This is essential for identifying frequently co-purchased product combinations.

### B. Richness for Customer Segmentation
The **CustomerID**, **InvoiceDate**, **Quantity**, and **UnitPrice** fields enable the construction of an **RFM (Recency, Frequency, Monetary)** model. This allows for robust **unsupervised clustering** to identify distinct customer segments based on purchasing value and behavior.

### C. Practical Relevance
The data is drawn from a real-world e-commerce setting, ensuring that the insights derived—such as optimizing inventory, promotional strategies, and personalized marketing—are **actionable and relevant** to the retail business context.