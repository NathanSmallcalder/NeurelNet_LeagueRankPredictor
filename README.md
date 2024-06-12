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

## GitHub

The project can be found on GitHub {link}

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

