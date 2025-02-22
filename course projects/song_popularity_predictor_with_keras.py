import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GroupShuffleSplit
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks

spotify_df = pd.read_csv('spotify.csv')
X = spotify_df.copy().dropna()
y = X.pop('track_popularity')
artists = X['track_artist']
features_num = ['danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness',
                'liveness', 'valence', 'tempo', 'duration_ms']
features_cat = ['playlist_genre']
preprocessor = make_column_transformer(
    (StandardScaler(), features_num),
    (OneHotEncoder(), features_cat)
)

def group_split(X,y,group,train_size=0.75):
    splitter = GroupShuffleSplit(train_size=train_size)
    train,test = next(splitter.split(X,y,groups=group))
    return (X.iloc[train], X.iloc[test], y.iloc[train],y.iloc[test])

X_train,X_valid,y_train,y_valid = group_split(X,y,artists)
X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)
y_train = y_train/100 # scaling popularity to 0-1 range
y_valid = y_valid/100
input_shape = [X_train.shape[1]]
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])
model.compile(
    optimizer='adam',
    loss='mae'
)
early_stopping = callbacks.EarlyStopping(
    min_delta=0.001,
    patience=5,
    restore_best_weights=True
)
history = model.fit(
    X_train,y_train,
    validation_data=(X_valid,y_valid),
    batch_size=512,
    epochs=50,
    callbacks=[early_stopping],
    verbose=0
    )
history_df = pd.DataFrame(history.history)
history_df.loc[0:,['loss','val_loss']].plot()
plt.show()
print("Minimum loss: {:0.4f}".format(history_df['loss'].min()))
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()))

# Model with Dropout layers
model2 = keras.Sequential([
    layers.Input(input_shape),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1)
])
model2.compile(
    optimizer='adam',
    loss='mae'
)
history2 = model2.fit(
    X_train,y_train,
    validation_data=(X_valid,y_valid),
    batch_size=512,
    epochs=50,
    callbacks=[early_stopping],
    verbose=0
    )
history_df2 = pd.DataFrame(history2.history)
history_df2.loc[0:,['loss','val_loss']].plot()
plt.show()
print("Minimum loss with dropout: {:0.4f}".format(history_df2['loss'].min()))
print("Minimum Validation Loss with dropout: {:0.4f}".format(history_df2['val_loss'].min()))