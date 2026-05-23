# Car Price Prediction Analysis
### NPTEL Python for Data Science — Certified Course Project
### Author: M. Charulatha | MSc Bioinformatics and Data Science

---

## Objective
Predict the selling price of a pre-owned car based on
features like age, engine power, kilometers driven,
vehicle type, gearbox, fuel type, brand, and damage status
using regression models.

---

## Dataset
- Name    : Pre-Owned Cars Dataset (German used car market)
- Records : 50,001 rows, 19 columns (original)
- After removing duplicates and applying working range: 42,772 records
- Features used: age, powerPS, kilometer, vehicleType,
  gearbox, model, fuelType, brand, notRepairedDamage
- Target  : price (log-transformed for modelling)

---

## Steps Performed
1. Loaded and explored dataset — checked structure, types, missing values
2. Dropped irrelevant columns: dateCrawled, name, dateCreated, postalCode, lastSeen
3. Removed duplicate records
4. Identified and removed extreme outliers using working range:
   - yearOfRegistration : 1950 to 2018
   - price              : 100 to 150,000
   - powerPS            : 10 to 500
   - Approximately 6,700 outlier records removed
5. Feature engineering: combined yearOfRegistration and monthOfRegistration
   into a single 'age' variable (car age in years)
6. Identified significant vs insignificant variables using
   visualisation and cross-tabulation analysis
7. Dropped insignificant variables: seller, offerType, abtest
8. Checked correlation between numerical variables and price
9. Applied log transformation on price to reduce right skewness
10. Built and evaluated models on two datasets:
    Set A — Rows with missing values dropped (omitted)
    Set B — Missing values imputed (median for numerical, mode for categorical)
11. Compared all models against a baseline RMSE benchmark

---

## Two Dataset Approaches for Missing Value Handling

| Approach | Method | Records Used |
|---|---|---|
| Set A | Drop all rows containing missing values | 32,884 |
| Set B | Impute — median for numerical columns, mode for categorical | 42,772 |

---

## Models Built

| Model | Description |
|---|---|
| Linear Regression | Baseline linear model assuming linear relationship between features and log-price |
| Random Forest Regressor | Ensemble of 100 decision trees — handles non-linear relationships |

Both models applied on Set A (omitted) and Set B (imputed).
Price log-transformed before modelling to correct right skewness.
Train-test split: 70% training, 30% testing (random_state=3)

---

## Complete Results — All Metrics

### SET A — Missing Values Omitted (32,884 records)
Train: 23,018 samples | Test: 9,866 samples | Features: 300

| Metric | Baseline | Linear Regression | Random Forest |
|---|---|---|---|
| RMSE | 1.1274 | 0.5455 | 0.4361 |
| R2 — Training | — | 0.7800 | 0.9202 |
| R2 — Testing | — | 0.7659 | 0.8540 |
| MSE | — | 0.2976 | 0.1902 |

### SET B — Missing Values Imputed (42,772 records)
Train: 29,940 samples | Test: 12,832 samples | Features: 303

| Metric | Baseline | Linear Regression | Random Forest |
|---|---|---|---|
| RMSE | 1.1884 | 0.6484 | 0.4943 |
| R2 — Training | — | 0.7072 | 0.9024 |
| R2 — Testing | — | 0.7023 | 0.8270 |
| MSE | — | 0.4204 | 0.2443 |

---

## Model Interpretation

### What is RMSE?
Root Mean Squared Error measures average prediction error.
Lower RMSE = better model predictions.
Both models far outperform the baseline — confirming genuine predictive power.

### What is R2 Score?
R-squared measures what percentage of price variance the model explains.
Random Forest R2 of 0.854 means it explains 85.4% of car price variance — strong result.

### Key Observations
- Random Forest outperforms Linear Regression in both Set A and Set B
- Set A (omitted) gives better results than Set B (imputed) for both models
- Overfitting present in Random Forest: Train R2 (0.920) vs Test R2 (0.854)
- Both models reduce RMSE by more than 50% compared to the baseline
- Log transformation successfully normalised the heavily right-skewed price distribution

---

## Visualisations — All 14 Plots with Findings

### Before Cleaning (Raw Data)
| Plot | File | Finding |
|---|---|---|
| Price Distribution — Raw | Price_distribution__density____raw_data.png | Extreme right skew extending to 12 million — confirms outliers must be removed before modelling |
| PowerPS Distribution — Raw | PowerPS_distribution___raw_data.png | Extreme right skew extending to 20,000 PS — most values concentrated near zero showing data quality issues |

### After Cleaning and Working Range Filter
| Plot | File | Finding |
|---|---|---|
| Price Distribution — Cleaned | Price_distribution_with_KDE___cleaned.png | After filtering to 100–150,000 range, distribution is still right-skewed — justifies log transformation for modelling |
| PowerPS Distribution — Cleaned | PowerPS_distribution_with_KDE___cleaned.png | After filtering to 10–500 PS, bimodal distribution — peaks at 75PS and 105PS representing economy and mid-range cars |
| Age Distribution | Age_distribution_with_KDE.png | Most cars are 10–20 years old — bimodal peak suggesting two market segments: newer and older used cars |

### Feature Relationships with Price
| Plot | File | Finding |
|---|---|---|
| PowerPS vs Price Scatter | PowerPS_vs_Price_scatter.png | Positive relationship — higher engine power corresponds to higher price. Fan-shaped pattern shows increasing variance at higher power values |
| Kilometer vs Price Boxplot | Kilometer_vs_Price_boxplot.png | Clear negative relationship — cars with more kilometers have lower median prices. Confirms kilometer is a significant predictor |
| Gearbox vs Price Boxplot | Gearbox_vs_Price_boxplot.png | Automatic cars have higher median price than manual cars — gearbox is a significant variable |
| Damage vs Price Boxplot | Damage_vs_Price_boxplot.png | Cars with no repaired damage (no) have significantly higher prices than damaged cars (yes) — strongest categorical predictor |

### Categorical Variable Distributions
| Plot | File | Finding |
|---|---|---|
| Vehicle Type Count | Vehicle_type_count_bar_chart.png | Limousine (12,000+) and small car (9,300+) dominate the market — station wagon third most common |
| Gearbox Count | Gearbox_count_bar_chart.png | Manual cars (32,000+) far outnumber automatic (9,400) — market is dominated by manual transmission |
| Fuel Type Count | Fuel_type_bar_chart.png | Petrol (26,500+) dominates over diesel (12,800) — other fuel types negligible in this market |

### Log Transformation Effect on Price
| Plot | File | Finding |
|---|---|---|
| Set B Imputed — Before vs After Log | imputed_data_before_vs_after.png | Before: heavily right-skewed (0–150,000). After log transform: near-normal bell shape centred at 8 — confirms log transformation works correctly for imputed dataset |
| Set A Omitted — Before vs After Log | OMIT_MISSING_VALUE_BEFORE_VS_AFTER.png | Before: heavily right-skewed (0–150,000). After log transform: near-normal distribution centred at 8 — confirms log transformation works correctly for omitted dataset |

---

## Significant vs Insignificant Variables

### Significant Variables (kept for modelling)
| Variable | Reason |
|---|---|
| age | Older cars have lower prices — strong negative relationship |
| powerPS | Higher power = higher price — positive relationship |
| kilometer | More kilometers = lower price — negative relationship |
| vehicleType | Different vehicle types have different price ranges |
| gearbox | Automatic cars priced higher than manual |
| model | Car model significantly affects pricing |
| fuelType | Fuel type impacts price — petrol vs diesel difference |
| brand | Brand name is a major price driver |
| notRepairedDamage | Damaged cars are significantly cheaper |

### Insignificant Variables (dropped)
| Variable | Reason for Dropping |
|---|---|
| seller | Almost all sellers are private — no variation |
| offerType | Almost all are standard offers — no variation |
| abtest | Equally distributed 50/50 — no predictive power |

---

## Tools and Libraries Used
| Tool | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data loading, manipulation, and cleaning |
| NumPy | Numerical operations and log transformation |
| Scikit-learn | ML models, train-test split, metrics |
| Seaborn | All statistical visualisations |
| Matplotlib | Plot rendering and saving |

---

## Files in This Repository

| File | Description |
|---|---|
| car_price_prediction.py | Complete Python code with all steps |
| cars_sampled.csv | Dataset used for analysis |
| Price_distribution__density____raw_data.png | Price dist before cleaning |
| PowerPS_distribution___raw_data.png | PowerPS dist before cleaning |
| Price_distribution_with_KDE___cleaned.png | Price dist after cleaning |
| PowerPS_distribution_with_KDE___cleaned.png | PowerPS dist after cleaning |
| Age_distribution_with_KDE.png | Car age distribution |
| PowerPS_vs_Price_scatter.png | PowerPS vs price relationship |
| Kilometer_vs_Price_boxplot.png | Kilometer vs price relationship |
| Gearbox_vs_Price_boxplot.png | Gearbox vs price comparison |
| Vehicle_type_count_bar_chart.png | Vehicle type distribution |
| Gearbox_count_bar_chart.png | Gearbox type distribution |
| Fuel_type_bar_chart.png | Fuel type distribution |
| Damage_vs_Price_boxplot.png | Damage status vs price |
| imputed_data_before_vs_after.png | Log transform effect — Set B |
| OMIT_MISSING_VALUE_BEFORE_VS_AFTER.png | Log transform effect — Set A |
