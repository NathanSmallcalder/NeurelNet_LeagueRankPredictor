from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
import numpy as np
import pandas as pd
import mysql.connector
import pandas as pd
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

config = {
    'user': 'LeagueStats3223',
    'password': 'password',
    'host': 'localhost',  
    'port': 3306,
    'database': 'LeagueStats',
}

connection = mysql.connector.connect(**config)
cursor = connection.cursor()


query = """
    SELECT `SummonerMatchTbl`.ChampionFk, `MatchStatsTbl`.`MinionsKilled`,
           `MatchStatsTbl`.kills, `MatchStatsTbl`.assists, `MatchStatsTbl`.deaths, `MatchStatsTbl`.DmgDealt, `MatchStatsTbl`.DmgTaken,
           `MatchStatsTbl`.TurretDmgDealt, `MatchStatsTbl`.TotalGold, `MatchStatsTbl`.EnemyChampionFk,
           `MatchTbl`.GameDuration, `MatchStatsTbl`.DragonKills, `MatchStatsTbl`.BaronKills, `MatchStatsTbl`.Win, `MatchTbl`.RankFk
    FROM `SummonerMatchTbl`
    JOIN `MatchStatsTbl` ON `MatchStatsTbl`.SummonerMatchFk = `SummonerMatchTbl`.SummonerMatchId
    JOIN `MatchTbl` ON `MatchTbl`.MatchId = `SummonerMatchTbl`.MatchFk
    WHERE `MatchTbl`.`QueueType` = 'CLASSIC';
"""


cursor.execute(query)
data = cursor.fetchall()


columns = ['ChampionFk', 'MinionsKilled', 'kills','assists','deaths', 'DmgDealt', 'DmgTaken', 'TurretDmgDealt', 'TotalGold',
               'EnemyChampionFk', 'GameDuration', 'DragonKills', 'BaronKills', 'Win','RankFk']


df_games = pd.DataFrame(data, columns=columns)


df_games.to_csv("LeagueData", encoding='utf-8', index=False)
df_games = df_games.dropna()
def adjust_labels(label):
    if label < 1:
        return 0
    elif label > 10:
        return 9
    else:
        return label - 1  
    


# Separate features (X) and target variable (y)
X = df_games.drop(columns=['RankFk'])
y = df_games['RankFk']
y = y.apply(adjust_labels)
class_counts = y.value_counts()
print(df_games['RankFk'].value_counts()) 

smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
df_resampled['RankFk'] = y_resampled

print(df_resampled['RankFk'].value_counts())

df_resampled['GameDuration'] = df_resampled['GameDuration'] / 60
df_resampled['CreepScore'] = df_resampled['MinionsKilled'] / df_resampled['GameDuration']
df_resampled['DmgDealtPerMin'] = df_resampled['DmgDealt'] / df_resampled['GameDuration']
df_resampled['DmgTakenPerMin'] = df_resampled['DmgTaken'] / df_resampled['GameDuration']
df_resampled = df_resampled.dropna()
X = df_resampled.drop('RankFk', axis=1)
X = df_resampled.drop('GameDuration', axis=1)
y = df_resampled['RankFk']

def create_model():
    model = Sequential([
        Input(shape=(X.shape[1],)),
        Dense(254, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),  
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=64, verbose=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model.fit(X_train_scaled, y_train)

test_accuracy = model.score(X_test_scaled, y_test)
print("Test Accuracy:", test_accuracy)

from sklearn.model_selection import KFold
import numpy as np

## Cross Validation
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
accuracy_scores = []
loss_scores = []

for train_index, test_index in kfold.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Normalize features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and compile the model
    model = create_model()

    # Train the model
    model.fit(X_train_scaled, y_train, epochs=100, batch_size=64, verbose=0)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    loss_scores.append(loss)
    accuracy_scores.append(accuracy)

print(f"Mean Accuracy: {np.mean(accuracy_scores):.4f}")
print(f"Standard Deviation of Accuracy: {np.std(accuracy_scores):.4f}")
print(f"Mean Loss: {np.mean(loss_scores):.4f}")
print(f"Standard Deviation of Loss: {np.std(loss_scores):.4f}")
