# 🩺 FitPulse Analytics — Fitness ML Pipeline

FitPulse Analytics is an end-to-end **Machine Learning pipeline** built using **Streamlit** that processes Fitbit-style fitness data to perform:

* Data preprocessing & cleaning
* Feature extraction from time-series data
* Forecasting using time-series models
* User behavior clustering
* Interactive visualization

---

## 🚀 Features

### 🔧 1. Data Preprocessing

* Upload CSV fitness dataset
* Detect and visualize missing values
* Perform automatic cleaning:

  * Numeric → interpolation
  * Categorical → fill with "Unknown"
* Exploratory Data Analysis (EDA)

---

### 🤖 2. Pattern Extraction Pipeline

#### 📁 Multi-file Upload

Automatically detects required Fitbit datasets:

* Daily Activity
* Hourly Steps
* Hourly Intensities
* Minute Sleep
* Heart Rate

---

#### 🧪 TSFresh Feature Extraction

* Extracts statistical features from heart rate time-series
* Generates feature matrix for each user
* Displays heatmap visualization

---

#### 📈 Prophet Forecasting

* Forecasts future trends (30 days ahead):

  * Heart Rate
  * Steps
  * Sleep
* Includes confidence intervals

---

#### 🔵 Clustering

* **KMeans Clustering**
* **DBSCAN Clustering (outlier detection)**
* Feature scaling using StandardScaler

---

#### 📊 Dimensionality Reduction

* PCA (2D visualization)
* t-SNE (non-linear visualization)

---

#### 📉 Elbow Method

* Helps determine optimal number of clusters

---

#### 📊 Cluster Profiling

* Displays average values per cluster
* Automatically labels:

  * 🏃 Highly Active
  * 🚶 Moderately Active
  * 🛋️ Sedentary

---

## 🧱 Tech Stack

* **Frontend/UI**: Streamlit
* **Data Processing**: Pandas, NumPy
* **Visualization**: Matplotlib, Seaborn
* **Machine Learning**:

  * Scikit-learn (KMeans, DBSCAN, PCA, TSNE)
* **Time-Series**:

  * Prophet (by Meta)
* **Feature Engineering**:

  * TSFresh

---

## 📂 Project Structure

```
fitpulse/
│
├── app.py                # Main Streamlit app
├── requirements.txt      # Dependencies
├── README.md             # Project documentation
└── datasets/             # (Optional) Sample data
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

### 3. Run the App

```bash
streamlit run app.py
```

---

## 📸 Output Screens (Suggested)

Include screenshots of:

* TSFresh heatmap
* Prophet forecast
* Clustering (PCA & t-SNE)
* Elbow curve
* Cluster profiles

---

## 📊 Dataset

This project uses **Fitbit-style fitness datasets**, including:

* Daily activity logs
* Heart rate time-series
* Sleep records

---

## ⚠️ Limitations

* Basic data cleaning (no advanced outlier handling)
* No model evaluation metrics
* DBSCAN parameters not optimized dynamically
* Forecast accuracy depends on data quality

---

## 🌟 Future Improvements

* Add model evaluation metrics
* Auto-tune clustering parameters
* Improve feature selection after TSFresh
* Deploy as web application
* Add real-time data support

---

## 👨‍💻 Author

**SATIN**

---

## 📄 License

This project is for educational and academic use.
# FitPulse-Health-Anomaly-Detection-from-Fitness-Devices
