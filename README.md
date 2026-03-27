# 🩺 FitPulse Analytics — Fitness ML Pipeline

FitPulse Analytics is an end-to-end **Machine Learning pipeline for fitness data analysis**, built with Streamlit.
It processes Fitbit-style datasets to perform **data preprocessing, feature extraction, forecasting, clustering, and anomaly detection** within a unified interactive dashboard.

---

## 🚀 Key Features

### 🔧 Data Preprocessing

* Upload CSV datasets
* Detect & visualize missing values
* Automatic cleaning:

  * Numeric → interpolation
  * Categorical → "Unknown" fill
* Exploratory Data Analysis (EDA)

---

### 🤖 Pattern Extraction Pipeline

#### 📁 Multi-file Dataset Support

Automatically detects:

* Daily Activity
* Hourly Steps
* Hourly Intensities
* Minute Sleep
* Heart Rate

---

#### 🧪 TSFresh Feature Extraction

* Extracts statistical features from time-series data
* Generates user-level feature matrix
* Heatmap visualization

---

#### 📈 Time-Series Forecasting (Prophet)

* 30-day prediction for:

  * Heart Rate
  * Steps
  * Sleep
* Includes confidence intervals

---

#### 🔵 Clustering & Segmentation

* KMeans clustering
* DBSCAN (outlier detection)
* Feature scaling using StandardScaler

---

#### 📊 Dimensionality Reduction

* PCA (linear visualization)
* t-SNE (non-linear embedding)

---

#### 📉 Elbow Method

* Determines optimal number of clusters

---

#### 📊 Cluster Profiling

* Aggregated cluster insights
* Automatic labeling:

  * 🏃 Highly Active
  * 🚶 Moderately Active
  * 🛋️ Sedentary

---

## 🧱 Tech Stack

| Category            | Tools                       |
| ------------------- | --------------------------- |
| Frontend            | Streamlit                   |
| Data Processing     | Pandas, NumPy               |
| Visualization       | Matplotlib, Seaborn, Plotly |
| Machine Learning    | Scikit-learn                |
| Feature Engineering | TSFresh                     |
| Forecasting         | Prophet                     |

---

## 📂 Project Structure

```bash
fitpulse/
│
├── main_app.py         # Main Streamlit application
├── requirements.txt    # Dependencies
├── README.md           # Documentation
└── datasets/           # (Optional) sample data
```

---

## ⚙️ Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/your-username/fitpulse.git
cd fitpulse
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Application

```bash
streamlit run main_app.py
```

---

## 📊 Dataset

The project uses Fitbit-style datasets including:

* Daily activity logs
* Heart rate time-series
* Sleep tracking data

---

## 📸 Suggested Outputs

* TSFresh feature heatmap
* Prophet forecast plots
* PCA & t-SNE clustering
* Elbow curve
* Cluster profiles

---

## ⚠️ Limitations

* Basic preprocessing (no advanced outlier handling)
* No formal model evaluation metrics
* DBSCAN parameters require manual tuning
* Forecast accuracy depends on input data quality

---

## 🔮 Future Improvements

* Add model evaluation metrics
* Automate hyperparameter tuning
* Improve feature selection post-TSFresh
* Deploy as a web application
* Integrate real-time wearable data

---

## 📄 License

This project is licensed under the MIT License.
