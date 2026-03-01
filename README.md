# 🏥 Medical Insurance Charges — End-to-End Regression Project

## 🚀 Live Demo
👉 [Try the Live App Here](https://medical-insurance-charge-predictor.streamlit.app)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://medical-insurance-charge-predictor.streamlit.app)

> Predicting medical insurance charges using machine learning.  
> Dataset: [Medical Insurance Charges 2021–2025 Enhanced](https://www.kaggle.com/) (Kaggle)

---

## 📌 Project Overview

This project builds an end-to-end regression pipeline to predict annual medical insurance charges based on patient demographics and health attributes. The project progresses from statistical analysis (OLS) through ensemble methods (Random Forest, Gradient Boosting), systematically validating each modeling decision with EDA evidence.

A Random Forest Classifier is also built as an extension to detect undisclosed smokers — a fraud detection use case for insurance companies.

---

## 🎯 Objectives

- Predict insurance charges accurately using regression models
- Understand which factors drive high insurance costs
- Compare linear vs tree-based models with full diagnostic analysis
- Build a smoker detection classifier for fraud detection

---

## 📁 Project Structure

```
├── Notebook_1_EDA_OLS.ipynb          # Exploratory Data Analysis + OLS Regression
├── Notebook_2_Random_Forest.ipynb    # Random Forest Regressor + Classifier
├── Notebook_3_Gradient_Boosting.ipynb# Gradient Boosting Regressor
├── Notebook_4_Final_Comparison.ipynb # Final model comparison + conclusions
├── medical_df_clean.csv              # Cleaned and feature engineered dataset
└── README.md
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | Kaggle — Medical Insurance Charges 2021–2025 |
| Rows | 1,337 |
| Features | 24 (raw) → 9 (after feature engineering) |
| Target | `charges` (annual insurance cost in USD) |

**Key Features Used:**

| Feature | Type | Description |
|---|---|---|
| age | Numeric | Age of the policyholder |
| bmi | Numeric | Body Mass Index |
| children | Numeric | Number of dependents |
| smoker_yes | Binary | Smoking status (1=Yes, 0=No) |
| sex_male | Binary | Gender (1=Male, 0=Female) |
| region_* | Binary | One-hot encoded region |
| age_smoker_interaction | Numeric | age × smoker (centered) |

---

## 🔍 Key EDA Findings

- **Smoking** is the single strongest predictor — smokers pay **3–6x more** than non-smokers regardless of age
- **Age and BMI** together with smoking explain **95% of charge variance**
- **Sex and region** have near-zero influence on charges
- Target variable is **heavily right-skewed** — confirmed via boxplot and Q-Q plot
- Three distinct patient subpopulations identified in residual plots — driving the move from OLS to tree-based models

---

## 🤖 Models Built

### Regression Models

| Model | Test R² | Test MAE | Test RMSE | R² Gap |
|---|---|---|---|---|
| OLS Full (9 features) | 0.751 | — | — | — |
| OLS Reduced (4 features) | 0.749 | — | — | — |
| OLS Cook's Cleaned | 0.747 | — | — | — |
| OLS VIF Fixed | 0.751 | — | — | — |
| Random Forest Default | 0.8409 | $2,733 | $4,643 | 0.1347 |
| Random Forest Tuned | 0.8647 | $2,464 | $4,282 | 0.0453 |
| GBM Default | 0.8684 | $2,358 | $4,224 | 0.0329 |
| GBM Tuned (300 rounds) | 0.8487 | $2,582 | $4,529 | 0.0842 |
| GBM Optimal (50 rounds) | 0.8649 | $2,608 | $4,279 | 0.0123 |

### Classification Model (Smoker Detection)

| Model | Test Accuracy | ROC AUC | Smoker Recall |
|---|---|---|---|
| RF Classifier (without charges) | 0.7724 | 0.5039 | 0.08 |
| RF Classifier (with charges) | 0.9552 | 0.9941 | 0.88 |

---

## 🏆 Best Models

**For Accuracy:**
> GBM Default — Test R² = 0.8684, MAE = $2,358

**For Production (Generalisation):**
> GBM Optimal (n_estimators=50) — R² Gap = 0.0123, Train/Test RMSE virtually identical

---

## 💡 Feature Importance

Both Random Forest and Gradient Boosting independently ranked features identically:

```
smoker_yes     →  61–64%  ██████████████████████████
bmi            →  20–21%  ████████
age            →  11–12%  █████
children       →   1–2%   ▌
everything else →  < 1%   
```

---

## 🔑 Business Insights

1. **Smoking drives costs** — A smoker aged 18–25 costs more ($28,490 avg) than a non-smoker aged 56–65 ($14,087 avg)
2. **Three data points are enough** — knowing smoker status, BMI and age predicts charges with 86%+ accuracy
3. **Sex and region are irrelevant** — should not be primary factors in premium pricing
4. **Fraud detection** — The classifier detects undisclosed smokers with 99.4% AUC using charges and demographics
5. **OLS is insufficient** — Linear regression capped at 75% R² due to non-linear subpopulation structure

---

## 🛠️ Tech Stack

```
Python 3.10
pandas
numpy
matplotlib
seaborn
scikit-learn
statsmodels
scipy
```

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/yourusername/medical-insurance-charges.git
cd medical-insurance-charges

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels scipy

# Run notebooks in order
Notebook_1 → Notebook_2 → Notebook_3 → Notebook_4
```

---

## 📈 Results Summary

```
OLS Baseline     →  R² = 0.751   (linear model ceiling)
Random Forest    →  R² = 0.8647  (+11.4% over OLS)
Gradient Boost   →  R² = 0.8684  (+11.7% over OLS)
```

Tree-based models solved the three-subpopulation problem that OLS could not handle — confirmed by the transformation from clustered to random residual plots.



---

## 👤 Author

**APARNA M P**  
[GitHub](https://github.com/aparna-2001) | [LinkedIn](https://linkedin.com/in/aparnamp) | [email](mailto:aparnamp966@gmail.com)

---

## 📄 License

This project is licensed under the MIT License.
