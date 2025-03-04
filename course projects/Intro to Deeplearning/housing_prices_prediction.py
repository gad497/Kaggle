from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd

house_data = pd.read_csv("housing_prices_train.csv")
house_data.dropna(axis=0)
y = house_data.SalePrice
feature_cols = ["LotArea","YearBuilt","1stFlrSF","2ndFlrSF","FullBath","BedroomAbvGr","TotRmsAbvGrd"]
X = house_data[feature_cols]
print(f"Data description: \n{X.describe()}")
X_train,X_test,y_train,y_test = train_test_split(X,y)
model = DecisionTreeRegressor(random_state=1)
model.fit(X_train,y_train)
predictions = model.predict(X_test)
print(f"Predictions of firt 5 data using Decision Tree model: {predictions[:5]}")
print(f"Actual values: {list(y_test)[:5]}")
mse = mean_absolute_error(predictions,y_test)
print(f"mean absolute error for Decision Tree Model: {mse}")


model_2 = RandomForestRegressor(random_state=1)
model_2.fit(X_train,y_train)
predictions = model_2.predict(X_test)
print(f"Predictions of first 5 data using Random Forest model: {predictions[:5]}")
print(f"Actual values: {list(y_test)[:5]}")
mse = mean_absolute_error(predictions, y_test)
print(f"mean absolute error for Random Forest model: {mse}")