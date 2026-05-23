# ============================================================
# CAR PRICE PREDICTION ANALYSIS
# ============================================================
# Author   : M. Charulatha
# Degree   : MSc Bioinformatics and Data Science
# Course   : NPTEL Python for Data Science (Certified)
# Dataset  : Pre-Owned Cars Dataset (cars_sampled.csv)
#
# OBJECTIVE:
#   Predict the price of a pre-owned car based on features
#   like age, power, kilometer, vehicle type, gearbox,
#   fuel type, brand, and damage status.
#
# APPROACH:
#   Two datasets used for model building:
#   Set A — Missing values OMITTED  (32,884 records)
#   Set B — Missing values IMPUTED  (42,772 records)
#
# MODELS BUILT:
#   1. Linear Regression
#   2. Random Forest Regressor (100 trees)
#
# RESULTS SUMMARY:
#   SET A (Omitted):
#     Baseline RMSE        : 1.1274
#     Linear Regression R2 : Train=0.780, Test=0.766, RMSE=0.5455
#     Random Forest R2     : Train=0.920, Test=0.854, RMSE=0.4361
#
#   SET B (Imputed):
#     Baseline RMSE        : 1.1884
#     Linear Regression R2 : Train=0.707, Test=0.702, RMSE=0.6484
#     Random Forest R2     : Train=0.902, Test=0.827, RMSE=0.4943
# ============================================================


# ============================================================
# STEP 1 — IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection  import train_test_split
from sklearn.linear_model     import LinearRegression
from sklearn.ensemble         import RandomForestRegressor
from sklearn.metrics          import mean_squared_error

# Set default plot size
sns.set(rc={'figure.figsize': (11.7, 8.27)})


# ============================================================
# STEP 2 — LOAD DATASET
# Original dataset: 50,001 records, 19 columns
# ============================================================

car_data = pd.read_csv("cars_sampled.csv")
car      = car_data.copy()


# ============================================================
# STEP 3 — EXPLORATORY DATA ANALYSIS (EDA)
# Understanding structure, types, distributions
# ============================================================

car.info()
print("\nMissing values:\n", car.isnull().sum())

sum_num = car.describe()
print("\nNumerical Summary:\n", sum_num)

sum_cat = car.describe(include='O')
print("\nCategorical Summary:\n", sum_cat)

print("\nUnique car names:", np.unique(car['name']))


# ============================================================
# STEP 4 — DROP IRRELEVANT COLUMNS
# dateCrawled, name, dateCreated, postalCode, lastSeen
# are not useful for price prediction
# ============================================================

col = ['dateCrawled', 'name', 'dateCreated', 'postalCode', 'lastSeen']
car = car.drop(columns=col, axis=1)

# Remove duplicate records — keep first occurrence
car.drop_duplicates(keep='first', inplace=True)
print(f"\nShape after dropping duplicates: {car.shape}")


# ============================================================
# STEP 5 — SET WORKING RANGE FOR NUMERICAL VARIABLES
# Removes extreme outliers that would skew the model
#
# yearOfRegistration : 1950 to 2018 (valid car years)
# price              : 100 to 150,000 (realistic price range)
# powerPS            : 10 to 500 (realistic engine power)
# ============================================================

# --- yearOfRegistration ---
yearwisecount = car['yearOfRegistration'].value_counts().sort_index()
print("\nRecords above 2018:", sum(car['yearOfRegistration'] > 2018))
print("Records below 1950:", sum(car['yearOfRegistration'] < 1950))

sns.regplot(x='yearOfRegistration', y='price',
            scatter=True, fit_reg=False, data=car)
plt.title("Year of Registration vs Price")
plt.tight_layout()
plt.savefig("plot1_year_vs_price.png")
plt.show()

# --- price ---
pricecount = car['price'].value_counts().sort_index()
sns.distplot(car['price'])
plt.title("Price Distribution")
plt.savefig("plot2_price_distribution.png")
plt.show()

sns.boxplot(y=car['price'])
plt.title("Price Boxplot")
plt.savefig("plot3_price_boxplot.png")
plt.show()

print("\nRecords above 150000:", sum(car['price'] > 150000))
print("Records below 100:",    sum(car['price'] < 100))

# --- powerPS ---
powercount = car['powerPS'].value_counts().sort_index()
sns.distplot(car['powerPS'])
plt.title("Power (PS) Distribution")
plt.savefig("plot4_power_distribution.png")
plt.show()

sns.boxplot(y=car['powerPS'])
plt.title("Power (PS) Boxplot")
plt.savefig("plot5_power_boxplot.png")
plt.show()

sns.regplot(x='powerPS', y='price',
            scatter=True, fit_reg=False, data=car)
plt.title("Power (PS) vs Price")
plt.savefig("plot6_power_vs_price.png")
plt.show()

print("\nRecords above 500 powerPS:", sum(car['powerPS'] > 500))
print("Records below 10 powerPS: ", sum(car['powerPS'] < 10))

# Apply working range filter — approximately 6,700 records removed
car = car[
    (car.yearOfRegistration <= 2018) &
    (car.yearOfRegistration >= 1950) &
    (car.powerPS >= 10) &
    (car.powerPS <= 500) &
    (car.price >= 100) &
    (car.price <= 150000)
]
print(f"\nShape after working range filter: {car.shape}")


# ============================================================
# STEP 6 — FEATURE ENGINEERING
# Combine yearOfRegistration and monthOfRegistration
# into a single 'age' variable (car age in years)
# Drop original year and month columns after this
# ============================================================

car['monthOfRegistration'] = car['monthOfRegistration'] / 12
car['age'] = (2018 - car['yearOfRegistration']) + car['monthOfRegistration']
car['age'] = round(car['age'], 2)

print("\nCar Age Summary:\n", car['age'].describe())

car = car.drop(columns=['yearOfRegistration', 'monthOfRegistration'], axis=1)


# ============================================================
# STEP 7 — VISUALISE KEY VARIABLES AFTER CLEANING
# ============================================================

# Age distribution and boxplot
sns.distplot(car['age'])
plt.title("Car Age Distribution")
plt.savefig("plot7_age_distribution.png")
plt.show()

sns.boxplot(y=car['age'])
plt.title("Car Age Boxplot")
plt.savefig("plot8_age_boxplot.png")
plt.show()

# Price distribution and boxplot after range filter
sns.distplot(car['price'])
plt.title("Price Distribution (After Filter)")
plt.savefig("plot9_price_after_filter.png")
plt.show()

sns.boxplot(y=car['price'])
plt.title("Price Boxplot (After Filter)")
plt.savefig("plot10_price_boxplot_after.png")
plt.show()

# Age vs Price scatter
sns.regplot(x='age', y='price', scatter=True, fit_reg=False, data=car)
plt.title("Car Age vs Price")
plt.savefig("plot11_age_vs_price.png")
plt.show()

# PowerPS vs Price scatter
sns.regplot(x='powerPS', y='price', scatter=True, fit_reg=False, data=car)
plt.title("Power (PS) vs Price")
plt.savefig("plot12_power_vs_price_clean.png")
plt.show()


# ============================================================
# STEP 8 — IDENTIFY SIGNIFICANT VARIABLES
# Check each categorical variable for significance
# Variables with near-uniform distribution are insignificant
# ============================================================

# seller — INSIGNIFICANT (almost all are private)
print("\nSeller counts:\n",    car['seller'].value_counts())

# offerType — INSIGNIFICANT (almost all have offers)
print("\nOfferType counts:\n", car['offerType'].value_counts())

# abtest — INSIGNIFICANT (equally distributed 50/50)
print("\nAbtest distribution:")
print(pd.crosstab(car['abtest'], columns='count', normalize=True))

# vehicleType — SIGNIFICANT
sns.countplot(x='vehicleType', data=car)
plt.title("Vehicle Type Distribution")
plt.savefig("plot13_vehicletype.png")
plt.show()

# gearbox — SIGNIFICANT (affects price)
sns.boxplot(x='gearbox', y='price', data=car)
plt.title("Gearbox vs Price")
plt.savefig("plot14_gearbox_vs_price.png")
plt.show()

# kilometer — SIGNIFICANT (negative correlation with price)
sns.regplot(x='kilometer', y='price', scatter=True, fit_reg=False, data=car)
plt.title("Kilometer vs Price")
plt.savefig("plot15_kilometer_vs_price.png")
plt.show()

# fuelType — SIGNIFICANT
sns.boxplot(x='fuelType', y='price', data=car)
plt.title("Fuel Type vs Price")
plt.savefig("plot16_fueltype_vs_price.png")
plt.show()

# brand — SIGNIFICANT
sns.boxplot(x='brand', y='price', data=car)
plt.title("Brand vs Price")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig("plot17_brand_vs_price.png")
plt.show()

# notRepairedDamage — SIGNIFICANT (damaged cars cost less)
sns.boxplot(x='notRepairedDamage', y='price', data=car)
plt.title("Not Repaired Damage vs Price")
plt.savefig("plot18_damage_vs_price.png")
plt.show()

# Drop insignificant columns
col = ['seller', 'offerType', 'abtest']
car = car.drop(columns=col, axis=1)
print(f"\nShape after dropping insignificant columns: {car.shape}")


# ============================================================
# STEP 9 — CORRELATION ANALYSIS
# Check numerical variable correlations with price
# ============================================================

cars        = car.select_dtypes(exclude=[object])
correlation = cars.corr()
print("\nCorrelation Matrix:\n", round(correlation, 3))
print("\nCorrelation with Price (sorted):")
print(cars.corr().loc[:, 'price'].abs().sort_values(ascending=False)[1:])


# ============================================================
# STEP 10 — MODEL BUILDING
# Two approaches:
#   Set A — Omit rows with missing values
#   Set B — Impute missing values
#         (median for numerical, mode for categorical)
# Price is log-transformed to reduce skewness
# Split: 70% train, 30% test
# ============================================================

# ---- SET A: OMIT MISSING VALUES ----
car_omit = car.dropna(axis=0)
car_omit = pd.get_dummies(car_omit, drop_first=True)

x1 = car_omit.drop(['price'], axis='columns', inplace=False)
y1 = car_omit['price']

# Log transform price — reduces right skew for better model fit
y1 = np.log(y1)

x_train, x_test, y_train, y_test = train_test_split(
    x1, y1, test_size=0.3, random_state=3)
print(f"\nSet A — Train: {x_train.shape}, Test: {x_test.shape}")

# Baseline RMSE for Set A (predicting mean for all)
base_pred                  = np.repeat(np.mean(y_test), len(y_test))
base_root_mean_square_error = np.sqrt(mean_squared_error(y_test, base_pred))
print(f"Set A Baseline RMSE: {base_root_mean_square_error:.4f}")

# Linear Regression — Set A
lgr              = LinearRegression(fit_intercept=True)
model_lin_1      = lgr.fit(x_train, y_train)
car_pred_lin_1   = lgr.predict(x_test)
lin_rmse1        = np.sqrt(mean_squared_error(y_test, car_pred_lin_1))
r2_lin_test1     = model_lin_1.score(x_test,  y_test)
r2_lin_train1    = model_lin_1.score(x_train, y_train)

# Residual plot for Linear Regression Set A
residuals1 = y_test - car_pred_lin_1
sns.regplot(x=car_pred_lin_1, y=residuals1, scatter=True, fit_reg=False)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot — Linear Regression (Set A)")
plt.tight_layout()
plt.savefig("plot19_residuals_linear_seta.png")
plt.show()

# Random Forest Regressor — Set A
rf           = RandomForestRegressor(n_estimators=100, max_depth=100,
                                     min_samples_split=10, min_samples_leaf=4,
                                     random_state=1)
model_rf1    = rf.fit(x_train, y_train)
car_pred_rf1 = rf.predict(x_test)
rf_rmse1     = np.sqrt(mean_squared_error(y_test, car_pred_rf1))
r2_rf_test1  = model_rf1.score(x_test,  y_test)
r2_rf_train1 = model_rf1.score(x_train, y_train)


# ---- SET B: IMPUTE MISSING VALUES ----
# Numerical columns  → fill with median
# Categorical columns → fill with mode (most frequent value)
car_imputed = car.apply(
    lambda x: x.fillna(x.median())
    if x.dtype == 'float'
    else x.fillna(x.value_counts().index[0])
)
car_imputed = pd.get_dummies(car_imputed, drop_first=True)

x2 = car_imputed.drop(['price'], axis='columns', inplace=False)
y2 = car_imputed['price']
y2 = np.log(y2)

x_train1, x_test1, y_train1, y_test1 = train_test_split(
    x2, y2, test_size=0.3, random_state=3)
print(f"\nSet B — Train: {x_train1.shape}, Test: {x_test1.shape}")

# Baseline RMSE for Set B
base_pred_b                       = np.repeat(np.mean(y_test1), len(y_test1))
base_root_mean_square_error_imputed = np.sqrt(
    mean_squared_error(y_test1, base_pred_b))
print(f"Set B Baseline RMSE: {base_root_mean_square_error_imputed:.4f}")

# Linear Regression — Set B
lgr2           = LinearRegression(fit_intercept=True)
model_lin_2    = lgr2.fit(x_train1, y_train1)
car_pred_lin_2 = lgr2.predict(x_test1)
lin_rmse2      = np.sqrt(mean_squared_error(y_test1, car_pred_lin_2))
r2_lin_test2   = model_lin_2.score(x_test1,  y_test1)
r2_lin_train2  = model_lin_2.score(x_train1, y_train1)

# Random Forest Regressor — Set B
rf2          = RandomForestRegressor(n_estimators=100, max_depth=100,
                                     min_samples_split=10, min_samples_leaf=4,
                                     random_state=1)
model_rf2    = rf2.fit(x_train1, y_train1)
car_pred_rf2 = rf2.predict(x_test1)
rf_rmse2     = np.sqrt(mean_squared_error(y_test1, car_pred_rf2))
r2_rf_test2  = model_rf2.score(x_test1,  y_test1)
r2_rf_train2 = model_rf2.score(x_train1, y_train1)


# ============================================================
# STEP 11 — FINAL RESULTS
# ============================================================

print("\n" + "="*55)
print("SET A — MISSING VALUES OMITTED (32,884 records)")
print("="*55)
print(f"Baseline RMSE          : {base_root_mean_square_error:.4f}")
print(f"Linear Regression RMSE : {lin_rmse1:.4f}")
print(f"Linear Regression R2   : Train={r2_lin_train1:.4f}, Test={r2_lin_test1:.4f}")
print(f"Random Forest RMSE     : {rf_rmse1:.4f}")
print(f"Random Forest R2       : Train={r2_rf_train1:.4f}, Test={r2_rf_test1:.4f}")

print("\n" + "="*55)
print("SET B — MISSING VALUES IMPUTED (42,772 records)")
print("="*55)
print(f"Baseline RMSE          : {base_root_mean_square_error_imputed:.4f}")
print(f"Linear Regression RMSE : {lin_rmse2:.4f}")
print(f"Linear Regression R2   : Train={r2_lin_train2:.4f}, Test={r2_lin_test2:.4f}")
print(f"Random Forest RMSE     : {rf_rmse2:.4f}")
print(f"Random Forest R2       : Train={r2_rf_train2:.4f}, Test={r2_rf_test2:.4f}")

# ============================================================
# FINAL RESULTS FROM VARIABLE EXPLORER
# ============================================================
# SET A (Omitted — 32,884 records):
#   Baseline RMSE              : 1.1274
#   Linear Regression RMSE     : 0.5455
#   Linear Regression R2 Train : 0.7800
#   Linear Regression R2 Test  : 0.7659
#   Random Forest RMSE         : 0.4361
#   Random Forest R2 Train     : 0.9202
#   Random Forest R2 Test      : 0.8540
#
# SET B (Imputed — 42,772 records):
#   Baseline RMSE              : 1.1884
#   Linear Regression RMSE     : 0.6484
#   Linear Regression R2 Train : 0.7072
#   Linear Regression R2 Test  : 0.7023
#   Random Forest RMSE         : 0.4943
#   Random Forest R2 Train     : 0.9024
#   Random Forest R2 Test      : 0.8270
#
# CONCLUSION:
#   Random Forest outperforms Linear Regression in both sets
#   Set A gives better results than Set B for both models
#   Random Forest R2 of 0.854 means it explains 85.4% of
#   the variance in car prices — strong predictive performance
#   Key price drivers: age, powerPS, kilometer, brand,
#   notRepairedDamage, gearbox
# ============================================================
