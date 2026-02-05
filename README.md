# 🏠 California Housing Price Prediction

This project builds a Machine Learning model to predict **median house values** based on housing and demographic features using the California Housing dataset.

The project covers complete ML workflow including:

- Data exploration
- Data preprocessing
- Model training & comparison
- Hyperparameter tuning
- Performance evaluation
- Price prediction for new inputs

---

## 📊 Dataset

Dataset Source:  
👉 https://www.kaggle.com/datasets/camnugent/california-housing-prices

The dataset contains housing data from California districts.

### Features

| Feature | Description |
|----------|------------|
| longitude | How far west the house is |
| latitude | How far north the house is |
| housing_median_age | Median age of houses |
| total_rooms | Total rooms in block |
| total_bedrooms | Total bedrooms in block |
| population | Block population |
| households | Number of households |
| median_income | Median household income |
| ocean_proximity | Location relative to ocean |
| median_house_value | Target variable |

---

## 🧠 Machine Learning Workflow

### 1️⃣ Data Analysis
- Missing value detection
- Distribution analysis
- Outlier detection
- Correlation analysis
- Feature relationship visualization

### 2️⃣ Data Preprocessing
Implemented using **Scikit-Learn Pipelines**

#### Numerical Features
- Median Imputation
- Standard Scaling

#### Categorical Features
- Most Frequent Imputation
- One Hot Encoding

---

### 3️⃣ Models Implemented

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- HistGradientBoosting Regressor

---

### 4️⃣ Model Evaluation

Used:

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score
- 5-Fold Cross Validation

---

### 5️⃣ Hyperparameter Tuning

Performed using:

```

GridSearchCV

```

Optimized HistGradientBoosting parameters including:

- Learning rate
- Max depth
- Max leaf nodes
- Regularization
- Minimum samples leaf

---

## 🏆 Final Model Performance

### Tuned HistGradientBoosting Model

| Metric | Score |
|----------|-----------|
| Test RMSE | 47044 |
| Test MAE | 31034 |
| Test R² | 0.831 |

---

## 🧰 Tech Stack

### Programming Language
- Python

### Libraries
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## 📦 Installation

Clone repository:

```

git clone <your-repo-link>
cd <repo-name>

```

Install dependencies:

```

pip install numpy pandas matplotlib seaborn scikit-learn

```

---

## ▶️ Running The Project

Open Jupyter Notebook:

```

jupyter notebook house-prediction-model.ipynb

````

Run all cells to:

- Train models
- Compare performance
- Generate predictions

---

## 🔮 Prediction Function

The project provides a function for predicting house price using trained model.

### Example Usage

```python
predict_house_price(
    model=hgb_best,
    longitude=-122.230,
    latitude=37.880,
    housing_median_age=41,
    total_rooms=880,
    total_bedrooms=129,
    population=322,
    households=126,
    median_income=8.3252,
    ocean_proximity="NEAR BAY"
)
````

---

## 📈 Visualization Included

* Feature distributions
* Correlation heatmap
* Residual plots
* Target distribution
* Outlier analysis

---

## 📁 Project Structure

```
├── house-prediction-model.ipynb
├── housing.csv
├── README.md
```

---

## 💡 Key Learnings

* Building reusable ML pipelines
* Handling missing data
* Model selection & comparison
* Hyperparameter optimization
* Model evaluation best practices

---

## 🚀 Future Improvements

* Deploy model as web application
* Add feature engineering
* Experiment with deep learning models
* Create interactive dashboard
* Save model using Pickle or Joblib

---


## ⭐ If You Like This Project

Give it a star on GitHub
