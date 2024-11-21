import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import shap


df = pd.read_csv("./data20241116c.csv")

# Split the data
X = df.drop(columns=['demand','timestamp','Unnamed: 0'])
y = df['demand']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
categorical_features = ['hour_of_day', 'day_of_week','day_of_month','month_of_year']
dtrain = xgb.DMatrix(X_train, label=y_train,enable_categorical=True)
dtest = xgb.DMatrix(X_test, label=y_test,enable_categorical=True)

# Define parameters for XGBoost
params = {
    'objective': 'reg:squarederror',
    'max_depth': 8, 
    'min_child_weight': 1, 
    'subsample': 0.9972621730657976, 
    'colsample_bytree': 0.6429678916412152, 
    'learning_rate': 0.03567325422386057, 
    'n_estimators': 194
}

# Train the model and observe errors for each epoch

evals = [(dtrain, 'train'), (dtest, 'test')]
eval_result = {}  # To store evaluation results

xgb_model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=250,       # Number of epochs
    evals=evals,
    evals_result=eval_result,  
    verbose_eval=True          
)

# Plot train and test RMSE over epochs


epochs = len(eval_result['train']['rmse'])
x_axis = range(0, epochs)

plt.figure(figsize=(10, 6))
plt.plot(x_axis, eval_result['train']['rmse'], label='Train RMSE')
plt.plot(x_axis, eval_result['test']['rmse'], label='Test RMSE')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('XGBoost Training and Test RMSE over Epochs')
plt.legend()
plt.show()

y_test_pred = xgb_model.predict(dtest)



# Evaluate the model
#train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
# test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)

# #print(f"Training RMSE: {train_rmse:.2f}")
# print(f"Test RMSE: {test_rmse:.2f}")

# Scatter plot of actual vs. predicted values for test set
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Line for perfect prediction
plt.xlabel("Actual Demand")
plt.ylabel("Predicted Demand")
plt.title("XGBoost: Actual vs. Predicted Demand")



plt.show()


explainer = shap.Explainer(xgb_model, X_train)
shap_values = explainer(X_test)

# Plot SHAP feature importance
shap.summary_plot(shap_values, X_test, plot_type="bar")