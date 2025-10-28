# ✈️ Flight Track Prediction - Geographic & ML Analytics

> Advanced Machine Learning solution for predicting aircraft flight trajectories using geographic and aerodynamic features.

[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)](https://www.kaggle.com/code/shreyashpatil217/flight-track-prediction-geographic-analytics)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)]()

---



## 🎯 Project Overview

This project predicts **aircraft flight track (heading)** using machine learning by analyzing geographic coordinates, altitude, climb rate, and airspeed data. The solution compares multiple algorithms and provides comprehensive geographic and aerodynamic analysis.

**Key Achievement:**
- ✅ 14,667 training records analyzed
- ✅ 3 ML models trained and evaluated
- ✅ 9 advanced features engineered
- ✅ Production-ready predictions on 300 test cases

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Training Records** | 14,667 |
| **Test Records** | 300 |
| **Total Features** | 7 base + 9 engineered = 16 features |
| **Data Quality** | 100% complete (0 missing values) |
| **Target Variable** | Flight Track (heading angle in degrees) |

### Feature Description

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| geo_longitude | Float | -250.9 to 266.1 | Aircraft longitude coordinate |
| geo_latitude | Float | -38.5 to 123.0 | Aircraft latitude coordinate |
| altitude_pressure | Float | -12,629 to 20,986 ft | Barometric altitude |
| climb_rate | Float | -25.6 to 24.2 ft/min | Vertical speed |
| gps_altitude | Float | -13,148 to 22,387 ft | GPS-derived altitude |
| air_speed | Float | 0.05 to 505.2 knots | True airspeed |
| flight_track | Float | -247.6 to 572.3 ° | **[TARGET]** Heading direction |

---

## 🔧 Features & Engineering

### Base Features (7)
✅ Geographic coordinates (latitude, longitude)  
✅ Altitude measurements (pressure & GPS)  
✅ Climb rate (vertical speed)  
✅ Airspeed  

### Engineered Features (9)
```python
✅ speed_magnitude      → Ground speed from coordinate deltas
✅ altitude_change      → Vertical displacement rate
✅ acceleration         → Rate of speed change
✅ altitude_gradient    → GPS vs pressure altitude difference
✅ altitude_ma_3        → 3-point rolling average altitude
✅ speed_ma_3           → 3-point rolling average speed
```

---

## 🚀 Installation

### Clone Repository
```bash
git clone https://github.com/ShreyashPatil530/Flight-Track-Prediction.git
cd Flight-Track-Prediction
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Requirements
```
pandas==1.5.0
numpy==1.23.0
scikit-learn==1.1.0
matplotlib==3.5.0
seaborn==0.12.0
```

---

## 💻 Usage

### Quick Start
```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Load data
train_df = pd.read_csv('Train.csv')
test_df = pd.read_csv('Test.csv')

# Feature engineering
from flight_track_utils import create_features
train_features = create_features(train_df)

# Train model
X_train = train_features.drop('flight_track', axis=1)
y_train = train_features['flight_track']

model = RandomForestRegressor(n_estimators=100, max_depth=15)
model.fit(X_train, y_train)

# Predict
test_features = create_features(test_df)
predictions = model.predict(test_features)
```

### Full Pipeline
```python
python flight_track_prediction.py
```

---

## 📈 Model Performance

### Comparison Results

```
╔════════════════════════════════════════════════╗
║       MODEL PERFORMANCE BENCHMARK              ║
╠════════════════════════════════════════════════╣
║ Model              MAE      RMSE   Status     ║
║ Random Forest      83.43    105.11  🏆 BEST   ║
║ Gradient Boosting  83.94    105.43           ║
║ Linear Regression  83.38    104.92           ║
╚════════════════════════════════════════════════╝
```

### Metrics Explanation

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **MAE** | 83.43° | Average prediction error |
| **RMSE** | 105.11° | Penalizes larger errors |
| **Mean Prediction** | 175-195° | Typical flight headings |

---

## 📊 Visualizations

### Generated Charts

1. **📊 EDA Distributions** (6-panel histogram)
   - Latitude, Longitude, Altitude, Speed, Climb Rate, Flight Track

2. **🔥 Correlation Heatmap**
   - Feature relationships (0.93 correlation: GPS ↔ Pressure altitude)

3. **📍 Flight Trajectories**
   - Spatial scatter plot colored by altitude
   - Global aircraft coverage visualization

4. **📈 Model Comparison**
   - MAE, RMSE, R² score bar charts

5. **🎯 Feature Importance**
   - Random Forest feature ranking
   - Top 5: altitude_ma_3, speed_ma_3, air_speed, etc.

6. **📉 Residual Analysis**
   - Actual vs Predicted scatter plot
   - Residual distribution

**Save Location:** `/outputs/`

---

## 💡 Key Insights

### ✅ Data Quality
- **Completeness:** 100% (0 missing values)
- **Outliers:** None requiring removal
- **Distribution:** Normal Gaussian patterns

### ✅ Feature Analysis
- **Strongest Predictors:** Smoothed altitude & speed metrics
- **Correlation:** High correlation between GPS and pressure altitude (0.93)
- **Importance:** Temporal smoothing > raw geographic coordinates

### ✅ Model Behavior
- **Generalization:** Consistent performance across all models
- **Baseline:** Linear regression performs comparably
- **Robustness:** MAE stable across train/validation splits

### ✅ Geographic Patterns
- **Coverage:** Aircraft from -250° to 266° longitude
- **Latitude Range:** -38° to 123° (pole-to-equator variety)
- **Altitude:** -12,629 to 20,986 feet (extreme conditions included)

---



## 🛠️ Technologies & Libraries

| Category | Tools |
|----------|-------|
| **Language** | Python 3.8+ |
| **Data Processing** | Pandas, NumPy |
| **ML Models** | Scikit-Learn, XGBoost |
| **Visualization** | Matplotlib, Seaborn |
| **Optimization** | StandardScaler, Train-Test Split |
| **Evaluation** | MAE, RMSE, R² Score |

---

## 🎓 Learning Outcomes

✅ End-to-end machine learning pipeline  
✅ Geographic data analysis & visualization  
✅ Advanced feature engineering techniques  
✅ Multi-model comparison & evaluation  
✅ Production-ready code practices  
✅ Data quality assessment  
✅ Residual analysis & model diagnostics  

---

## 🚀 Future Improvements

- [ ] Implement LSTM/GRU for sequence prediction
- [ ] Add external features (weather, routing data)
- [ ] Ensemble stacking methods
- [ ] Bayesian hyperparameter optimization
- [ ] Real-time prediction API
- [ ] Aviation-specific domain features

---

## 📚 References

- [Kaggle Dataset](https://www.kaggle.com/datasets/gauravduttakiit/flight-track-prediction-challenge)
- [Scikit-Learn Documentation](https://scikit-learn.org/)
- [Pandas Guide](https://pandas.pydata.org/docs/)
- [Matplotlib Tutorial](https://matplotlib.org/stable/tutorials/index.html)

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💼 Contact

**Shreyash Patil**

- 📧 **Email:** [shreyashpatil530@gmail.com](mailto:shreyashpatil530@gmail.com)
- 💼 **LinkedIn:** [LinkedIn Profile](https://linkedin.com/in/shreyashpatil)
- 🐙 **GitHub:** [ShreyashPatil530](https://github.com/ShreyashPatil530)
- 🔗 **Kaggle:** [shreyashpatil217](https://www.kaggle.com/code/shreyashpatil217/flight-track-prediction-geographic-analytics)
- 🌐 **Portfolio:** [shreyash-patil-portfolio1.netlify.app](https://shreyash-patil-portfolio1.netlify.app/)

---

## ⭐ Show Your Support

If you found this project helpful, please consider:
- ⭐ Star this repository
- 🔀 Fork for your own projects
- 💬 Share feedback & suggestions
- 📤 Submit pull requests with improvements

---

<div align="center">

**Made with ❤️ by Shreyash Patil**

*Advancing Data Science & Machine Learning Excellence*

</div>
