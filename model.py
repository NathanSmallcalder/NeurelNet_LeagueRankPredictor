import mysql.connector
import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
import numpy as np

# Database configuration
config = {
    'user': 'LeagueStats5',
    'password': 'password',
    'host': 'localhost',
    'port': 3306,
    'database': 'LeagueStats',
}

# Connect to the database
connection = mysql.connector.connect(**config)
cursor = connection.cursor()

# SQL query
query = """
    SELECT `SummonerMatchTbl`.ChampionFk, `MatchStatsTbl`.`MinionsKilled`,
           `MatchStatsTbl`.Lane, `MatchStatsTbl`.DmgDealt, `MatchStatsTbl`.DmgTaken,
           `MatchStatsTbl`.TurretDmgDealt, `MatchStatsTbl`.TotalGold, `MatchStatsTbl`.EnemyChampionFk,
           `MatchTbl`.GameDuration, `MatchStatsTbl`.DragonKills, `MatchStatsTbl`.BaronKills, `MatchStatsTbl`.Win, `MatchTbl`.RankFk
    FROM `SummonerMatchTbl`
    JOIN `MatchStatsTbl` ON `MatchStatsTbl`.SummonerMatchFk = `SummonerMatchTbl`.SummonerMatchId
    JOIN `MatchTbl` ON `MatchTbl`.MatchId = `SummonerMatchTbl`.MatchFk
    WHERE `MatchTbl`.`QueueType` = 'CLASSIC';
"""

# Execute the query
cursor.execute(query)
data = cursor.fetchall()

# Define column names
columns = ['ChampionFk', 'MinionsKilled', 'lane', 'DmgDealt', 'DmgTaken', 'TurretDmgDealt', 'TotalGold',
               'EnemyChampionFk', 'GameDuration', 'DragonKills', 'BaronKills', 'Win', 'RankFk']

# Create DataFrame
df_games = pd.DataFrame(data, columns=columns)

# Map 'lane' values to numerical
df_games['lane'] = df_games['lane'].map({'TOP': 0, 'JUNGLE': 1, 'MIDDLE': 2, 'BOTTOM': 3, 'NONE': 4})
df_games.to_csv("LeagueData", encoding='utf-8', index=False)
# Split data into features and target
X = df_games.drop(columns=['RankFk'])
y = df_games['RankFk']

# Ranks can be between the value of 1-10
# Adjust labels to be within the range of 0 to 9
def adjust_labels(label):
    if label < 1:
        return 0
    elif label > 10:
        return 9
    else:
        return label - 1  # Subtract 1 to map rank values to the range of 0 to 9

# Apply the adjustment function to each label
y_adjusted = y.apply(adjust_labels)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_adjusted, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Building
model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),  # Third hidden layer (new)
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Model Training
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=64, validation_split=0.2)

# Model Evaluation
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
print("Test Accuracy:", test_accuracy)

# Prediction
# You can use the trained model to predict the rank for new data
# For example, you can use model.predict(new_data) where new_data is a numpy array with the same shape as X_train_scaled
