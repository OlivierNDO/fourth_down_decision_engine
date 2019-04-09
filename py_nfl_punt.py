# Punt Distance by Spot
##############################################################
# Package Imports & Configuation
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.linear_model import LogisticRegression, LinearRegression
import statsmodels.api as sm
from sklearn import linear_model
data_path = 'C:/Users/Nick/Desktop/nfl_sas/nfl_2009_2018.csv'

# Import & Explore Data
nfl_df = pd.read_csv(data_path)
play_types = set(nfl_df['play_type'])
nfl_df_10 = nfl_df.head(10)
nfl_cols = [c for c in nfl_df.columns]

# Isolate & Explore Punt Data
punt_df = nfl_df[nfl_df.play_type == 'punt'][['yardline_100', 'kick_distance', 'return_yards']]
punt_df['opponent_field_position'] = punt_df['yardline_100'] - punt_df['kick_distance'] + punt_df['return_yards']
punt_df['opponent_yds_to_endzone'] = 100 - punt_df['opponent_field_position']
punt_df['yardline_100_sq'] = [i**2 for i in punt_df['yardline_100']]
punt_df = punt_df.dropna()

plt.scatter(x = punt_df['yardline_100'],
            y = punt_df['opponent_yds_to_endzone'],
            alpha = 0.4,
            color = 'green',
            zorder = 2,
            label = '')
plt.xlabel('Punt Field Position')
plt.ylabel('Opponent Field Position After Punt')
plt.title('Punt Results')
plt.legend()
plt.show()

## Plot Regression Fit Against % Fourth Down Conversion Successes by Yards to Go
pred_distances = pd.DataFrame({'yardline_100': [i for i in range(100)],
                              'yardline_100_sq': [i**2 for i in range(100)]})

x = punt_df[['yardline_100', 'yardline_100_sq']].values
y = punt_df['opponent_yds_to_endzone'].values.reshape(punt_df.shape[0], 1)

regr = LinearRegression().fit(x,y)

plt.scatter(punt_df[['yardline_100']].values.reshape(punt_df.shape[0], 1), y,  color='black')
plt.plot(pred_distances[['yardline_100']].values, regr.predict(pred_distances.values), color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()

# Create Output File
pred_distances['opponent_yds_to_endzone'] = regr.predict(pred_distances.values)
pred_distances.drop('yardline_100_sq', axis = 1, inplace = True)
pred_distances.to_csv('C:/Users/Nick/Desktop/nfl_sas/punt_prob_df.csv', index = False)
