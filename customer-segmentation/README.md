# Mall Customer Segmentation Using Clustering
### A Machine Learning-Based Customer Segmentation System

This project is a Streamlit-based Customer Segmentation System that identifies different customer groups using the **K-Means Clustering Algorithm** based on **Annual Income** and **Spending Score**.

The application provides:

- Real-time customer segmentation
- Interactive Streamlit dashboard
- K-Means clustering visualization
- Customer segment prediction
- Business insights for each segment
- Customer profile analysis
- Modern responsive user interface

The system uses **Unsupervised Machine Learning**, **K-Means Clustering**, and **Interactive Visualizations** to help businesses understand customer behavior and improve marketing strategies.

---

## Features

- Customer Segmentation using K-Means
- Interactive Streamlit Dashboard
- Real-Time Customer Segment Prediction
- Annual Income & Spending Score Analysis
- Business Recommendations
- Cluster Visualization
- Customer Distribution Analysis
- Modern Responsive UI
- Customer Information Card
- Interactive Scatter Plot

---

## Project Structure

```
Mall-Customer-Segmentation/
│
├── app.py
├── shopping_Custmer_Data.csv
├── kmeans_model.pkl
├── scaler.pkl
├── K-means_clustering.ipynb
├── Hierarchial_clustering.ipynb
├── requirements.txt
└── README.md
```

---

## Tech Stack

- Python 3.x
- Streamlit
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Plotly
- Joblib

---

## Installation & Setup

### Clone the Repository

Repository Link: https://github.com/Ayaansohail11/Mall-Customer-Segmentation

```bash
git clone https://github.com/Ayaansohail11/Mall-Customer-Segmentation.git
cd Mall-Customer-Segmentation
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

---

## Dashboard Modules

| Module | Description |
|---|---|
| Customer Information | Enter customer details |
| Customer Segmentation | Predict customer segment |
| Cluster Visualization | Display customer clusters |
| Business Insights | Marketing recommendations |
| Customer Analysis | Income & Spending analysis |
| Developer Information | Project details |

---

## Dataset Features

The clustering model uses the following features:

- Annual Income (₹ in thousands)
- Spending Score (1–100)

**Additional User Inputs (UI only)**
- Customer Name
- Age
- Gender

> **Note:** Name, Age, and Gender are collected only for display purposes. The clustering model predicts segments using Annual Income and Spending Score.

---

## Machine Learning Algorithm

**Algorithm Used:** K-Means Clustering

**Data Preprocessing**
- Missing Value Handling
- Feature Scaling using StandardScaler

**Cluster Selection**
- Elbow Method
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Score

---

## Customer Segments

| Cluster | Customer Segment |
|---|---|
| Cluster 0 | Moderate Income, Low Spending Customers |
| Cluster 1 | Moderate Income, High Spending Customers |
| Cluster 2 | High Income, High Spending Customers |
| Cluster 3 | High Income, Low Spending Customers |

---

## Business Insights

**Moderate Income, Low Spending Customers**
- Low spending behavior
- Suitable for discounts and promotional campaigns

**Moderate Income, High Spending Customers**
- Regular loyal customers
- Suitable for loyalty programs and memberships

**High Income, High Spending Customers**
- Premium customers
- Best target for premium products and VIP services

**High Income, Low Spending Customers**
- Potential customers
- Personalized marketing can increase spending

---

## Model Evaluation

The clustering model was evaluated using:

- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Score

These metrics were used to determine the optimal number of clusters and evaluate cluster quality.

---

## How to Use

**Step 1** — Enter customer details
- Customer Name
- Age
- Gender
- Annual Income (₹ in thousands)
- Spending Score

**Step 2** — Click **Predict Customer Segment**

**Step 3** — The application will display
- Customer Segment
- Cluster Number
- Business Recommendation
- Customer Details

---

## Future Improvements

- Deploy on Streamlit Cloud
- Customer Database Integration
- Download Prediction Report
- Advanced Business Analytics
- Multiple Clustering Algorithms
- Customer Lifetime Value Prediction

---

## Disclaimer

This project is developed strictly for **educational and learning purposes**.

The predicted customer segments are generated using an unsupervised machine learning algorithm and should be considered analytical insights rather than definitive business decisions.

---

## Project Highlights

- Machine Learning Project
- Unsupervised Learning
- K-Means Clustering
- Customer Segmentation
- Streamlit Web Application
- Interactive Dashboard
- Business Analytics
- Portfolio & Resume Ready Project

---

## Contact

**Ayaan Sohail**

GitHub: https://github.com/Ayaansohail11
