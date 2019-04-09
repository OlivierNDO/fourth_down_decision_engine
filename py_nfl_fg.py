# Field Goal Probability by Spot
##############################################################
# Package Imports & Configuation
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn import linear_model
data_path = 'C:/Users/Nick/Desktop/nfl_sas/nfl_2009_2018.csv'

# Import & Explore Data
nfl_df = pd.read_csv(data_path, usecols = ['play_type', 'down', 'yardline_100', 'field_goal_result'])
play_types = set(nfl_df['play_type'])
nfl_df_10 = nfl_df.head(10)
nfl_cols = [c for c in nfl_df.columns]

# Separate Dataframes by Fourth Down Playtype
field_goal_df = nfl_df[(nfl_df['play_type'] == 'field_goal') & (nfl_df['down'] == 4)]
del nfl_df; import gc; gc.collect()

# Field Goal Probability by Yardline
fg_prob_df = field_goal_df[(field_goal_df['play_type'] == 'field_goal') & (field_goal_df['down'] == 4)][['yardline_100', 'field_goal_result']]
fg_prob_df = fg_prob_df[fg_prob_df['yardline_100'] < 54]
fgr = []
for i in fg_prob_df['field_goal_result']:
    if i == 'made':
        fgr.append(1)
    else:
        fgr.append(0)

fg_prob_df['field_goal_result_binary'] = fgr
fg_prob_df['yardline_100_sq'] = [i**2 for i in fg_prob_df['yardline_100']]

#fg_prob_df['field_goal_distance'] = [i + 17 for i in fg_prob_df['yardline_100']]
#fg_prob_df['field_goal_distance_sq'] = [i**2 for i in fg_prob_df['field_goal_distance']]
fg_prob_df['n'] = [i for i in range(fg_prob_df.shape[0])]
fg_prob_df.dropna(axis = 0, inplace = True)



# Fit Regression
fg_clf = LogisticRegression(fit_intercept = True)
fg_clf.fit(fg_prob_df[['yardline_100', 'yardline_100_sq']], fg_prob_df[['field_goal_result_binary']])
coefficients = [c for c in np.exp(fg_clf.coef_)]
intercept = fg_clf.intercept_

## Plot Logistic Regression Fit Against % Made by Field Goal Distance
pred_distances = pd.DataFrame({'yardline_100': [i for i in range(61)]})
pred_distances['yardline_100_sq'] = [i**2 for i in pred_distances['yardline_100']]
pred_distances['expected_probability'] = [ep for ep in fg_clf.predict_proba(pred_distances)[:,1]]

fg_summ_df = fg_prob_df.\
groupby(['yardline_100'], as_index = False).\
agg({'field_goal_result_binary':'mean',
     'n':'nunique'})


plt.scatter(x = pred_distances['yardline_100'],
            y = pred_distances['expected_probability'],
            zorder = 1,
            lw = 1.5,
            label = 'Fitted Values')

plt.scatter(x = fg_summ_df['yardline_100'],
            y = fg_summ_df['field_goal_result_binary'],
            s = fg_summ_df['n'],
            alpha = 0.4,
            color = 'green',
            zorder = 2,
            label = '% FG Made - Sized by n')
plt.xlabel('Yardline')
plt.ylabel('Probability')
plt.title('FG Attempt Success is a Binomial Function of Distance')
plt.legend()
plt.show()

# Create Output File
##############################################################
output = pred_distances[['yardline_100', 'expected_probability']]
output.to_csv('C:/Users/Nick/Desktop/nfl_sas/fg_probability_df.csv', index = False)



