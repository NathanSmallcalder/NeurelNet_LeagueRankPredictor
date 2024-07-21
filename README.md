# Predicting Player Ranks in League of Legends

## 1. Introduction

In this project, we aim to develop a predictive model that can accurately classify players' ranks in the popular multiplayer online battle arena game, League of Legends (LoL). We will utilize gameplay statistics and corresponding player ranks to train a machine learning model to predict player ranks based on in-game performance metrics.

### Problem Statement:
The objective is to predict the rank of LoL players based on their gameplay statistics, such as minion kills, damage dealt, objectives secured, etc.

### Objectives:
- Explore and preprocess the dataset
- Build a neural network model using TensorFlow/Keras
- Train and evaluate the model's performance
- Fine-tune hyperparameters to improve model performance
- Interpret the model's predictions and analyze results

### Database Setup

```
docker run --name=LeagueStats --env="MYSQL_ROOT_PASSWORD=root_password" -p 3306:3306 -d mysql:latest

docker exec -it LeagueStats mysql -h localhost -P 3306 --protocol=tcp -u root -proot_password

Insert the TableSetup.txt or TableSetupNoData.txt into the database.
```


## Data Collection

To Collect Data, the collect_summoners.py can be run by adding summoner name, region and tagline to lines 41-43, then run the script. After this the DataCollection.py script will cycle though all the summoners collected.

## GitHub

The project can be found on [Github](https://github.com/NathanSmallcalder/NeurelNet_LeagueRankPredictor)}

## 2. Data Collection

Data was collected from a [previous project](https://github.com/NathanSmallcalder/Dissertation), This data is from patch 13.6.1, so a few of the new avalbile champions/items are unavaliable. To collect new data, A [collection script ](https://github.com/NathanSmallcalder/DataAnalysisLeagueOfLegends/blob/main/DataCollection.py) and [database model](https://github.com/NathanSmallcalder/DataAnalysisLeagueOfLegends/blob/main/TableSetup.txt) can be implemented. To use the collection script enter a league of legends user and region in lines 31 and 32 and run the script. The data found in the LeagueData file containes training data. The dataset has already been prepared as seen below.

```
ChampionFk,MinionsKilled,lane,DmgDealt,DmgTaken,TurretDmgDealt,TotalGold,EnemyChampionFk,GameDuration,DragonKills,BaronKills,Win,RankFk
82,105,0.0,13538,26869,1,7581,115,1778,0,0,1,1
```

The example line shows the ChampionFk as an integer, which in this case would be Mordekaiser playing at lane 0 (Top Lane).

## 3. Data Preperation

To prepare the dataset for training a neural network, you need to map categorical features like "Rank" and "Lane" to integers so that they can be effectively used as input within the neural network. Ranks stored in the database had a value of 1-10, to be used in as an input the value of each rank was lowered by 1 value, to now be 0 to 9, with 0 being Unranked and 9 being Grandmaster.

```python
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
```

```python
df_games['lane'] = df_games['lane'].map({'TOP': 0, 'JUNGLE': 1, 'MIDDLE': 2, 'BOTTOM': 3, 'NONE': 4})
```

### Import Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

## 4. Data Loading and Exploration

```python
df.shape

(10038, 14)
```
```python
print(df['RankFk'].value_counts()) 

RankFk
7     2553
1     2513
8     2261
5     1302
9      460
4      385
6      214
10     186
3      110
2       35
0       19
Name: count, dtype: int64
```

In many real-world datasets, particularly those involving categorical outcomes such as player ranks, some classes may be underrepresented compared to others. This imbalance can result in biased models that perform poorly on the minority classes. For instance, if certain player ranks are significantly less frequent than others, a model might become biased towards predicting the more common ranks, leading to suboptimal performance for less frequent ones.

This imbalance often arises from data collection processes where an impartial variety of ranks might not be equally represented, resulting in a dataset where some classes have many more samples than others. To mitigate this issue, we use techniques such as oversampling to balance the class distribution in the dataset. Here, we'll perform class balancing by resampling the minority classes in our League of Legends dataset.

```python

X = df_games.drop(columns=['RankFk'])
y = df_games['RankFk']
y = y.apply(adjust_labels)

smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
df_resampled['RankFk'] = y_resampled
```


## 5. Feature Engineering

```python
# Feature engineering
df_games['GameDuration'] = df_games['GameDuration'] / 60
df_games['CreepScore'] = df_games['MinionsKilled'] / df_games['GameDuration']
df_games['DmgDealtPerMin'] = df_games['DmgDealt'] / df_games['GameDuration']
df_games['DmgTakenPerMin'] = df_games['DmgTaken'] / df_games['GameDuration']

# Define features and target variable
X = df_games.drop(columns=['RankFk', 'GameDuration'])
y = df_games['RankFk']
```

In predicting player ranks in League of Legends (LoL), certain gameplay statistics are critical in distinguishing the level of play. These attributes provide insights into a player's performance and can be highly indicative of their skill level and rank. In this section, we will explore three key attributes commonly used in rank prediction models:

 - Creep Score (CS)
 - Damage Dealt per Minute (DmgDealtPerMin)
 - Damage Taken per Minute (DmgTakenPerMin)

1. Creep Score (CS)

Creep Score represents the number of minions and neutral monsters a player has killed in a game. It is a fundamental metric in League of Legends as it directly correlates with a player's ability to farm effectively. Higher creep scores typically indicate better laning performance and resource management.
Effect:
 - Economic Advantage: A higher CS translates into more gold and better item progression.
 - Lane Control: Consistently high CS can signify strong lane control and positioning.

2. Damage Dealt per Minute (DmgDealtPerMin)

Damage Dealt per Minute measures the average amount of damage a player deals to enemies each minute. This metric provides insight into a player's offensive contributions and their effectiveness in team fights and skirmishes.
Effect:
 - Impact in Fights: Higher damage dealt indicates more significant contributions in team fights and skirmishes.
 - Role Performance: This metric helps in assessing the performance of damage dealers such as AD Carries and AP Mages.

3. Damage Taken per Minute (DmgTakenPerMin)

Damage Taken per Minute tracks the average amount of damage a player absorbs from opponents each minute. This attribute reflects the player's ability to manage damage and survive in engagements.
Effect:
 - Survivability: Lower damage taken often indicates better positioning and damage avoidance.
 - Tank Effectiveness: For tanky roles, a higher amount of damage taken might be expected, but the ability to mitigate this damage is crucial.

Using Key Attributes in Rank Prediction

In rank prediction models, these key attributes are utilized to gain a deeper understanding of player performance and to differentiate between different levels of play. By incorporating these metrics, the model can better capture the nuances of player skill and provide more accurate predictions.



## 6. Model

```python
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

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model.fit(X_train_scaled, y_train)

# Model Evaluation
test_accuracy = model.score(X_test_scaled, y_test)
```

## 7. Cross Validation

```python
## Cross Validation
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
accuracy_scores = []
loss_scores = []

for train_index, test_index in kfold.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

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
```
Implementation of Cross-Validation

In this section, we use k-fold cross-validation to assess the performance of our neural network model. We will split the dataset into 10 folds, train the model on 9 folds, and test it on the remaining fold. This process is repeated 10 times, with each fold serving as the test set exactly once.

Here is how we implement cross-validation for our neural network model:

Initialize Cross-Validation: We use KFold from scikit-learn to create the cross-validation splits. We set n_splits=10 for 10-fold cross-validation, and shuffle=True to ensure that the data is randomly shuffled before splitting.

Loop Through Folds: For each fold, we:
 - Split the data into training and testing sets based on the current fold indices.
 - Scale the features using StandardScaler to ensure they have similar ranges.
 - Train the model on the training set.
 - Evaluate the model on the testing set to obtain the loss and accuracy.

Record and Calculate Metrics: We record the loss and accuracy for each fold and then calculate the mean and standard deviation of these metrics across all folds to get a comprehensive view of model performance.

Overall, the cross-validation results indicate that the neural network model performs exceptionally well on this dataset and generalizes effectively across different subsets of the data. The low variability in accuracy and loss scores also suggests that the model is robust and stable.

## 8. Conclusion

The model demonstrates excellent performance with high accuracy and low loss. The results indicate that the neural network is effective at predicting player ranks based on gameplay statistics. Future work may include fine-tuning hyperparameters or exploring additional features.

```
Test Accuracy: 0.9881779222698389
Mean Accuracy: 0.9949
Standard Deviation of Accuracy: 0.0078
Mean Loss: 0.0146
Standard Deviation of Loss: 0.0064
```

### Notes:

1. **GitHub Link:** https://github.com/NathanSmallcalder/NeurelNet_LeagueRankPredictor
2. **Database Setup:** See Introduction.
3. **Data Collection Script:** https://github.com/NathanSmallcalder/NeurelNet_LeagueRankPredictor/blob/main/DataCollection.py

