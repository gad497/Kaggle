import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

fuel_df = pd.read_csv("fuel.csv")
X = fuel_df.copy()
y = X.pop('FE')
preprocessor = make_column_transformer(
    (StandardScaler(), make_column_selector(dtype_include=np.number)),
    (OneHotEncoder(sparse_output=False),make_column_selector(dtype_include=object))
    )
X = preprocessor.fit_transform(X)
y = np.log(y)
input_shape = [X.shape[1]]
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])
model.compile(
    optimizer='adam',
    loss='mae'
)
history = model.fit(
    X,y,
    batch_size=128,
    epochs=200
)
history.df = pd.DataFrame(history.history)
history.df['loss'].plot()
plt.show()