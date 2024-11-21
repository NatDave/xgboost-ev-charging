import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np
import statsmodels.api as sm


df = pd.read_csv("./data20241116c.csv")

# One-hot encode categorical features
df = pd.get_dummies(df, columns=['hour_of_day', 'day_of_week','day_of_month','month_of_year'], drop_first=True)

# Prepare train and test data
X = df.drop(columns=['demand','timestamp','Unnamed: 0'])
y = df['demand']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)



# Add a constant to X for the intercept
X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)

# Fit the OLS regression model
model = sm.OLS(y_train, X_train_const).fit()

# Predictions
y_train_pred = model.predict(X_train_const)
y_test_pred = model.predict(X_test_const)



# Evaluate the model
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)

print(f"Training RMSE: {train_rmse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")

# Scatter plot of actual vs. predicted values for test set
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Line for perfect prediction
plt.xlabel("Actual Demand")
plt.ylabel("Predicted Demand")
plt.title("Linear Regression: Actual vs. Predicted Demand")
plt.show()

# Output the regression summary including t-statistics, R², adjusted R², and F-statistic
print("Model Summary:")
print(model.summary())

# Extracting specific evaluation metrics
r_squared = model.rsquared
adjusted_r_squared = model.rsquared_adj
f_statistic = model.fvalue
t_statistics = model.tvalues

print("\nEvaluation Metrics:")
print(f"R-squared: {r_squared}")
print(f"Adjusted R-squared: {adjusted_r_squared}")
print(f"F-statistic: {f_statistic}")
print("T-statistics for each feature:\n", t_statistics)
