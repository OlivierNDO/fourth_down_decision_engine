# Determine Optimal Fourth Down Decision
#####################################################################
# Package Imports
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
import gc

# Input Files
input_folder_path = '.../prob_output/'
punt_file = 'punt_prob_df.csv'
conversion_file = 'conversion_probability_df.csv'
field_goal_file = 'fg_probability_df.csv'
drive_score_file = 'drive_score_probability_df.csv'

# Define Class Object
class fourth_down_decision:
    def __init__(self, input_folder, conv_file, fg_file, ds_file, punt_file,
                 sec_left_in_half, yardline_100, yds_to_go):
        self.input_folder = input_folder
        self.conv_file = conv_file
        self.fg_file = fg_file
        self.ds_file = ds_file
        self.punt_file = punt_file
        self.sec_left_in_half = sec_left_in_half
        self.yardline_100 = yardline_100
        self.yds_to_go = yds_to_go
        
    def seconds_to_time(self):
        sec = self.sec_left_in_half
        if (sec // 3600) == 0:
            HH = '00'
        elif (sec // 3600) < 10:
            HH = '0' + str(int(sec // 3600))
        else:
            HH = str(int(sec // 3600))
        min_raw = (np.float64(sec) - (np.float64(sec // 3600) * 3600)) // 60
        if min_raw < 10:
            MM = '0' + str(int(min_raw))
        else:
            MM = str(int(min_raw))
        sec_raw = (sec - (np.float64(sec // 60) * 60))
        if sec_raw < 10:
            SS = '0' + str(int(sec_raw))
        else:
            SS = str(int(sec_raw))
        return MM + ':' + SS

    def conversion_probability(self):
        conv_df = pd.read_csv(self.input_folder + self.conv_file)
        if int(self.yds_to_go) not in [int(i) for i in conv_df['ydstogo']]:
            conv_prob = 0
        else:
            conv_prob = conv_df[conv_df.ydstogo == self.yds_to_go]['expected_probability']
        return float(conv_prob)
    
    def field_goal_probability(self):
        fg_df = pd.read_csv(self.input_folder + self.fg_file)
        if int(self.yardline_100) not in [int(i) for i in fg_df['yardline_100']]:
            fg_prob = 0
        else:
            fg_prob = fg_df[fg_df.yardline_100 == self.yardline_100]['expected_probability']
        return float(fg_prob)
    
    def opponent_field_position_after_punt(self):
        punt_df = pd.read_csv(self.input_folder + self.punt_file)
        if int(self.yardline_100) not in [int(i) for i in punt_df['yardline_100']]:
            print("Error: yardline_100 should be a value between 1 and 99 in both the input file and fourth_down_decision parameters")
        else:
            punt_outcome = punt_df[punt_df.yardline_100 == self.yardline_100]['opponent_yds_to_endzone']
        return int(punt_outcome)
    
    def expected_points_after_conversion_success(self):
        ds_df = pd.read_csv(self.input_folder + self.ds_file)
        ds_df = ds_df[ds_df.yardline_100.astype(int) == int(self.yardline_100)]
        ds_df = ds_df[ds_df.half_seconds_remaining.astype(int) == int(self.sec_left_in_half)]
        return float(ds_df['expected_points'])
    
    def opponent_expected_points_after_turnover(self):
        ds_df = pd.read_csv(self.input_folder + self.ds_file)
        ds_df = ds_df[ds_df.yardline_100.astype(int) == int((100 - self.yardline_100))]
        ds_df = ds_df[ds_df.half_seconds_remaining.astype(int) == int(self.sec_left_in_half)]
        return float(ds_df['expected_points'])
    
    def opponent_expected_points_after_punt(self):
        opp_field_position = self.opponent_field_position_after_punt()
        ds_df = pd.read_csv(self.input_folder + self.ds_file)
        ds_df = ds_df[ds_df.yardline_100.astype(int) == int(opp_field_position)]
        ds_df = ds_df[ds_df.half_seconds_remaining.astype(int) == int(self.sec_left_in_half)]
        return float(ds_df['expected_points'])
    
    def make_decision(self, explain_rationale = False):
        # conditional probabilities and expected outcomes
        fg_prob = self.field_goal_probability()
        conv_prob = self.conversion_probability()
        exp_pts_after_conv = self.expected_points_after_conversion_success()
        exp_opp_pts_after_turnover = self.opponent_expected_points_after_turnover() 
        exp_opp_pts_after_punt = self.opponent_expected_points_after_punt()
        turnover_point_delta = (float(exp_opp_pts_after_turnover) - float(exp_opp_pts_after_punt))
        # expected value of each decision given conditional probabilities
        fg_EV = (float(fg_prob) * 3) - ((1 - float(fg_prob)) * turnover_point_delta)
        conv_EV = (float(conv_prob) * float(exp_pts_after_conv)) - float(turnover_point_delta * (1 - float(exp_pts_after_conv)))
        if (fg_EV < 0 and conv_EV < 0):
            decision = 'punt the ball'
        elif (fg_EV > 0 and fg_EV > conv_EV):
            decision = 'kick a field goal'
        else:
            decision = 'attempt a 1st down conversion'
        if explain_rationale:
            situation_expl = "4th down and {x}, {y} yards from the end zone with {t} left on the clock\n\n\n".format(x = str(self.yds_to_go),
                                                                                                                     y = str(self.yardline_100),
                                                                                                                     t = self.seconds_to_time())                                                                                                 
            fg_expl = "> field goal likelihood: {fgl} %\n\n".format(fgl = str(np.round(fg_prob * 100,1)))
            conv_expl = "> likelihood of getting a first down: {lfd} %\n\n".format(lfd = str(np.round(conv_prob * 100, 1)))
            conv_pts_expl = "> if you convert the first down, expected points scored on drive: {p}\n\n".format(p = str(np.round(exp_pts_after_conv, 1)))
            to_delta_expl = "> if you turn the ball over, your opponent is expected to score {p} more points in the next drive\n\n".format(p = str(np.round(turnover_point_delta,3)))
            decision_expl = "> you should {d}".format(d = decision)
            agg_expl = situation_expl + fg_expl + conv_expl + conv_pts_expl + to_delta_expl + decision_expl
            return decision
            print(agg_expl)
        else:
            return decision
        
# Compare Actual Fourth Down Decisions to Algorithm Recommendations            
nfl_df = pd.read_csv('C:/Users/Nick/Desktop/nfl_sas/nfl_2009_2018.csv')
keep_cols = ['ydstogo', 'down', 'play_type', 'yardline_100', 'half_seconds_remaining']
fourth_down = nfl_df[nfl_df.down == 4][keep_cols]
fourth_down = fourth_down[fourth_down.play_type.isin(['field_goal', 'punt', 'run', 'pass'])]
del nfl_df; gc.collect()

decision_list = []
for i in fourth_down['play_type']:
    if i in ['run', 'pass']:
        decision_list.append('attempt a 1st down conversion')
    elif i == 'punt':
        decision_list.append('punt the ball')
    else:
        decision_list.append('kick a field goal')

fourth_down['decision'] = decision_list
        
recommendation = []

for i, r in fourth_down.iterrows():
    my_fourth_down = fourth_down_decision(input_folder = input_folder_path,
                                          conv_file = conversion_file,
                                          fg_file = field_goal_file,
                                          ds_file = drive_score_file,
                                          punt_file = punt_file,
                                          sec_left_in_half = int(r['half_seconds_remaining']),
                                          yardline_100 = int(r['yardline_100']),
                                          yds_to_go = int(r['ydstogo']))
    recommendation.append(my_fourth_down.make_decision())
    if (i // 1000) == (i / 1000):
        print("{it} of {length} completed".format(it = str(i), length = str(fourth_down.shape[0])))
        
output = pd.DataFrame({'recommendation': recommendation,
                       'actual_decision': decision_list})
    
output.to_csv(input_folder_path + 'rec_vs_actual.csv')
