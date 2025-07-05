def get_target_and_features(tourney_data):
    tourney_data['target'] = tourney_data['T1_Score'] - tourney_data['T2_Score']
    tourney_data['target_binary'] = (tourney_data['target'] > 0).astype('int')

    non_features = ['Season', 'DayNum', 'T1_TeamID', 'T1_Score', 'T2_TeamID', 'T2_Score',
                    'T1_Playin_seed', 'T2_Playin_seed', 'target', 'target_binary', 'PointDiff',
                    'T1_Poss', 'T2_Poss', 'T1_OE', 'T2_OE', 'T1_DE', 'T2_DE', 'Pace']
    features = [col for col in list(tourney_data.columns) if col not in non_features]

    return tourney_data, features


def get_xgb_params():
    return {
        "n_estimators": 1000,
        "objective": 'reg:squarederror',
        "eval_metric": "mae",
        "learning_rate": 0.02,
        "subsample": 0.35,
        "colsample_bytree": 0.7,
        "num_parallel_tree": 1,
        "min_child_weight": 40,
        "max_depth": 3,
        "verbosity": 0,
        "early_stopping_rounds": 25,
    }


MENS_FEATURES = [
    'ordinal_rank_mean_Diff', 'OE_Rating_Diff', 'ordinal_rank_max_Diff',
    'NumGames_Diff', 'reg_opponent_PF_Diff', 'T1_seed', 'T2_seed',
    'T1_OE_Rating', 'T2_OE_Rating', 'T1_ordinal_rank_mean', 'T2_ordinal_rank_mean',
    'WinRatio_Diff', 'T1_reg_Score', 'T2_reg_Score', 'T1_reg_FGA3',
    'T2_reg_FGA3', 'T1_reg_opponent_DR', 'T2_reg_opponent_DR'
]

WOMENS_FEATURES = [
    'OE_Rating_Diff', 'WinRatio_Diff', 'seed_Diff',
    'reg_Blk_Diff', 'reg_opponent_Blk_Diff'
]