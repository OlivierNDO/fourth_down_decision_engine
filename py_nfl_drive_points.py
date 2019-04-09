# Expected Points by Drive Starting Position and Time Left in Game
##############################################################
# Package Imports & Configuation
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.linear_model import LogisticRegression, LinearRegression
import statsmodels.api as sm
from sklearn import linear_model
data_path = '.../nfl_2009_2018.csv'

nec_cols = ['game_id', 'drive', 'play_id', 'play_type', 'half_seconds_remaining',
            'posteam_score', 'posteam_score_post', 'yardline_100']

# Import & Explore Data
nfl_df = pd.read_csv(data_path, usecols = nec_cols).fillna(0)

# Create Primary Key for Each Drive
drive_prim_key = []
for i, r in nfl_df.iterrows():
    drive_prim_key.append(str(r['game_id']) + '_' + str(r['drive']))

nfl_df['drive_prim_key'] = drive_prim_key

# Get Min. Play ID for Each Drive
min_play_df = nfl_df[['drive_prim_key', 'play_id']].\
groupby(['drive_prim_key'], as_index = False).\
agg({'play_id':'min'})

first_play = pd.merge(nfl_df[['drive_prim_key', 'play_id', 'yardline_100', 'half_seconds_remaining']],
                      min_play_df,
                      'inner',
                      ['drive_prim_key', 'play_id'])

# Get Result of Drive
nfl_df['score_on_drive'] = nfl_df['posteam_score_post'] - nfl_df['posteam_score'] 

drive_score_df = nfl_df[['drive_prim_key', 'score_on_drive']].\
groupby(['drive_prim_key'], as_index = False).\
agg({'score_on_drive':'max'})

# Join Drive Result w/ Start of Drive Variables
drive_df = pd.merge(drive_score_df, first_play, 'inner', 'drive_prim_key')

# Add Interaction Variable
drive_df['yardline_seconds_intx'] = drive_df['yardline_100'] * drive_df['half_seconds_remaining']

drive_result = []
for s in drive_df['score_on_drive']:
    if s >= 6:
        drive_result.append('td')
    elif s >= 3:
        drive_result.append('fg')
    else:
        drive_result.append('no_score')
drive_df['drive_result'] = drive_result

# Undersample by Response Variable Level
min_class = np.min([drive_df[drive_df.drive_result == r].shape[0] for r in set(drive_df['drive_result'])])

us_df_list = []
for r in set(drive_df['drive_result']):
    sub_df = drive_df[drive_df.drive_result == r].sample(n = min_class)
    us_df_list.append(sub_df)
    
drive_df_us = pd.concat(us_df_list)

# Fit Regression
x = drive_df[['yardline_100', 'half_seconds_remaining', 'yardline_seconds_intx']].values
y = drive_df['score_on_drive'].values.reshape(drive_df.shape[0], 1)
regr = LinearRegression().fit(x,y)
fitted_vals = regr.predict(x)
fit_assess_df = pd.DataFrame({'pred': [i[0] for i in fitted_vals],
                              'actual': [y for y in drive_df['score_on_drive']]})

# Plot Pred v. Actual
plt.scatter(fit_assess_df['pred'], fit_assess_df['actual'],  color='black')
plt.xticks(())
plt.yticks(())
plt.show()

ax = sns.violinplot(x="actual", y="pred", hue="actual", data=fit_assess_df)
plt.show()   
