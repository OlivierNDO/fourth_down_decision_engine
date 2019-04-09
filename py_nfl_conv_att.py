# 1st Down Conversion Likelihood
##############################################################

# Package Imports & Configuation
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn import linear_model
data_path = '.../nfl_2009_2018.csv'

# Import & Explore Data
nfl_df = pd.read_csv(data_path)
play_types = set(nfl_df['play_type'])
nfl_df_10 = nfl_df.head(10)
nfl_cols = [c for c in nfl_df.columns]

# Separate Dataframes by Fourth Down Playtype
fourth_conv_att_df = nfl_df[(nfl_df['play_type'].isin(['run', 'pass'])) & (nfl_df['down'] == 4)][['ydstogo', 'yards_gained']]
conv_att_success = []
for i, x in enumerate(fourth_conv_att_df['ydstogo']):
     if x > list(fourth_conv_att_df['yards_gained'])[i]:
         conv_att_success.append(0)
     else:
         conv_att_success.append(1)

fourth_conv_att_df['conv_att_success'] = conv_att_success
fourth_conv_att_df['n'] = [i for i in range(fourth_conv_att_df.shape[0])]

# Fit Logistic Regression
clf = LogisticRegression()
clf.fit(fourth_conv_att_df[['ydstogo']], fourth_conv_att_df[['conv_att_success']])
coefficients = [c for c in np.exp(clf.coef_)]
intercept = clf.intercept_

## Plot Logistic Regression Fit Against % Fourth Down Conversion Successes by Yards to Go
pred_distances = pd.DataFrame({'ydstogo': [i for i in range(int(np.min(fourth_conv_att_df['ydstogo'])),          
                                                              int(np.max(fourth_conv_att_df['ydstogo']) + 1))]})
#pred_distances['field_goal_distance_sq'] = [i**2 for i in pred_distances['field_goal_distance']]
pred_distances['expected_probability'] = [ep for ep in clf.predict_proba(pred_distances)[:,1]]

conv_summ_df = fourth_conv_att_df.\
groupby(['ydstogo'], as_index = False).\
agg({'conv_att_success':'mean',
     'n':'nunique'})


plt.scatter(x = pred_distances['ydstogo'],
            y = pred_distances['expected_probability'],
            zorder = 1,
            lw = 1.5,
            label = 'Fitted Values')

plt.scatter(x = conv_summ_df['ydstogo'],
            y = conv_summ_df['conv_att_success'],
            s = conv_summ_df['n'],
            alpha = 0.4,
            color = 'green',
            zorder = 2,
            label = 'Conversion Success Rate')
plt.xlabel('Yards to Go')
plt.ylabel('Probability')
plt.title('4th Down Conversion Attempt ~ Yards to Go')
plt.legend()
plt.show()

# Create Output File
##############################################################
pred_distances.to_csv('.../conversion_probability_df.csv', index = False)
