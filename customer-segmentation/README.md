# Mall Customer Segmentation Using Clustering

*A Machine Learning-Based Customer Segmentation System*

This project is a **Streamlit-based Customer Segmentation System** that identifies different customer groups using the **K-Means Clustering Algorithm** based on **Annual Income** and **Spending Score**.

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

# Features

* Customer Segmentation using K-Means
* Interactive Streamlit Dashboard
* Real-Time Customer Segment Prediction
* Annual Income & Spending Score Analysis
* Business Recommendations
* Cluster Visualization
* Customer Distribution Analysis
* Modern Responsive UI
* Customer Information Card
* Interactive Scatter Plot

---

# Project Structure

```text
Customer-Segmentation-Using-KMeans/
│
├── app.py
├── shopping_Custmer_Data.csv
├── kmeans_model.pkl
├── scaler.pkl
├── KMeans_Customer_Segmentation.ipynb
├── requirements.txt
└── README.md
```

---

# Tech Stack

* Python 3.x
* Streamlit
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Plotly
* Joblib

---
## Installation & Setup

### Clone the Repository

**Repository Link:**

https://github.com/AmitSharma9754/Mall-Customer-Segmentation

**Clone using Git:**
git clone https://github.com/AmitSharma9754/Mall-Customer-Segmentation.git
cd Mall Customer Segmentation

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

---

# Dashboard Modules

| Module | Description |
|----------|-------------|
| Customer Information | Enter customer details |
| Customer Segmentation | Predict customer segment |
| Cluster Visualization | Display customer clusters |
| Business Insights | Marketing recommendations |
| Customer Analysis | Income & Spending analysis |
| Developer Information | Project details |

---

# Dataset Features

The clustering model uses the following features:

* Annual Income
* Spending Score

### Additional User Inputs (UI)

* Customer Name
* Age
* Gender

> **Note:** Name, Age, and Gender are collected only for display purposes. The clustering model predicts segments using **Annual Income** and **Spending Score**.

---

# Machine Learning Algorithm

## Algorithm Used

* K-Means Clustering

### Data Preprocessing

* Missing Value Handling
* Feature Scaling using StandardScaler

### Cluster Selection

* Elbow Method
* Silhouette Score
* Davies-Bouldin Index
* Calinski-Harabasz Score

---

# Customer Segments

| Cluster | Customer Segment |
|----------|------------------|
| Cluster 0 | Moderate Income, Low Spending Customers |
| Cluster 1 | Moderate Income, High Spending Customers |
| Cluster 2 | High Income, High Spending Customers |
| Cluster 3 | High Income, Low Spending Customers |

---

# Business Insights

### Moderate Income, Low Spending Customers

* Low spending behavior
* Suitable for discounts and promotional campaigns

### Moderate Income, High Spending Customers

* Regular loyal customers
* Suitable for loyalty programs and memberships

### High Income, High Spending Customers

* Premium customers
* High-value customers
* Best target for premium products and VIP services

### High Income, Low Spending Customers

* Potential customers
* Personalized marketing can increase spending

---

# Model Evaluation

The clustering model was evaluated using:

* Silhouette Score
* Davies-Bouldin Index
* Calinski-Harabasz Score

These metrics were used to determine the optimal number of clusters and evaluate cluster quality.

---


# How to Use

## Step 1

Enter customer details

* Customer Name
* Age
* Gender
* Annual Income
* Spending Score

---

## Step 2

Click

```text
Predict Customer Segment
```

---

## Step 3

The application will display

* Customer Segment
* Cluster Number
* Business Recommendation
* Customer Details

---

## Screenshots

### Home Page

![Home Page](https://github.com/user-attachments/assets/4e6e0ec1-d367-453e-9605-612c5add5969
)

---

### Customer Prediction 

![Prediction](https://github.com/user-attachments/assets/57ba5418-c639-4c2c-a0f4-03dee8b25596
)

---

### Cluster Visualization

![Dashboard](https://github.com/user-attachments/assets/86fb42db-5072-429e-b694-47a319ce61cb
)

---

### Model Performance Metrics

![Model Metrics](https://github.com/user-attachments/assets/c86a157e-1a3c-448a-8765-0d07d1b11acb
)

---

### Developer & Info

![Developer & Info](https://github.com/user-attachments/assets/c0085f69-8e94-4f06-863e-d923cb01dd5c
)


---

# Future Improvements

* Deploy on Streamlit Cloud
* Customer Database Integration
* Download Prediction Report
* Advanced Business Analytics
* Multiple Clustering Algorithms
* Customer Lifetime Value Prediction

---

# Disclaimer

This project is developed strictly for educational and learning purposes.

The predicted customer segments are generated using an unsupervised machine learning algorithm and should be considered analytical insights rather than definitive business decisions.

---

# Project Highlights

Machine Learning Project

Unsupervised Learning

K-Means Clustering

Customer Segmentation

Streamlit Web Application

Interactive Dashboard

Business Analytics

Portfolio & Resume Ready Project

---

# Contact

**Amit Sharma**

Email: Amitsharma97545@gmail.com

GitHub:

https://github.com/AmitSharma9754
