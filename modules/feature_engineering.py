import pandas as pd
import numpy as np


def add_possesions_feats(data):
    data['T1_Poss'] = data['T1_FGA'] - data['T1_OR'] + data['T1_TO'] + 0.44 * data['T1_FTA']
    data['T2_Poss'] = data['T2_FGA'] - data['T2_OR'] + data['T2_TO'] + 0.44 * data['T2_FTA']

    data['T1_OE'] = data['T1_Score'] / data['T1_Poss']
    data['T2_OE'] = data['T2_Score'] / data['T2_Poss']

    data['T1_DE'] = data['T2_Score'] / data['T2_Poss']
    data['T2_DE'] = data['T1_Score'] / data['T1_Poss']

    data['Pace'] = (data['T1_Poss'] + data['T2_Poss']) / 2

    return data


def add_boxcore_feats(regular_data, tourney_data, data_type='detailed', sub_data=False):
    boxscore_cols = ['T1_Score', 'T2_Score', 'PointDiff',
                     'T1_FGM', 'T1_FGA', 'T1_FGM3', 'T1_FGA3', 'T1_FTM', 'T1_FTA', 'T1_OR', 'T1_DR', 'T1_Ast', 'T1_TO',
                     'T1_Stl', 'T1_Blk', 'T1_PF',
                     'T2_FGM', 'T2_FGA', 'T2_FGM3', 'T2_FGA3', 'T2_FTM', 'T2_FTA', 'T2_OR', 'T2_DR', 'T2_Ast', 'T2_TO',
                     'T2_Stl', 'T2_Blk', 'T2_PF']
    if data_type == 'compact':
        boxscore_cols = ['T1_Score', 'T2_Score', 'PointDiff']

    season_statistics = regular_data.groupby(["Season", 'T1_TeamID'])[boxscore_cols].agg('mean').reset_index()
    season_statistics.columns = [''.join(col).strip() for col in season_statistics.columns.values]

    season_statistics_T1 = season_statistics.copy()
    season_statistics_T2 = season_statistics.copy()

    season_statistics_T1.columns = ["T1_" + "reg_" + x.replace("T1_", "").replace("T2_", "opponent_") for x in
                                    list(season_statistics_T1.columns)]
    season_statistics_T2.columns = ["T2_" + "reg_" + x.replace("T1_", "").replace("T2_", "opponent_") for x in
                                    list(season_statistics_T2.columns)]
    season_statistics_T1.columns.values[0] = "Season"
    season_statistics_T2.columns.values[0] = "Season"
    season_statistics_T1.columns.values[1] = "T1_TeamID"
    season_statistics_T2.columns.values[1] = "T2_TeamID"

    cols_for_drop = ['T1_FGM', 'T1_FGA', 'T1_FGM3', 'T1_FGA3', 'T1_FTM', 'T1_FTA', 'T1_OR', 'T1_DR', 'T1_Ast', 'T1_TO',
                     'T1_Stl', 'T1_Blk', 'T1_PF', 'T2_FGM', 'T2_FGA', 'T2_FGM3', 'T2_FGA3', 'T2_FTM', 'T2_FTA', 'T2_OR',
                     'T2_DR', 'T2_Ast', 'T2_TO', 'T2_Stl', 'T2_Blk', 'T2_PF']
    if data_type == 'detailed' and not sub_data:
        tourney_data = tourney_data.drop(columns=cols_for_drop)

    tourney_data = pd.merge(tourney_data, season_statistics_T1, on=['Season', 'T1_TeamID'], how='left')
    tourney_data = pd.merge(tourney_data, season_statistics_T2, on=['Season', 'T2_TeamID'], how='left')

    return tourney_data


def add_last_14_days_feats(regular_data, tourney_data):
    last14days_stats_T1 = regular_data.loc[regular_data.DayNum > 118].reset_index(drop=True)
    last14days_stats_T1['win'] = np.where(last14days_stats_T1['PointDiff'] > 0, 1, 0)
    last14days_stats_T1 = last14days_stats_T1.groupby(['Season', 'T1_TeamID'])['win'].mean().reset_index(
        name='T1_win_ratio_14d')

    last14days_stats_T2 = regular_data.loc[regular_data.DayNum > 118].reset_index(drop=True)
    last14days_stats_T2['win'] = np.where(last14days_stats_T2['PointDiff'] < 0, 1, 0)
    last14days_stats_T2 = last14days_stats_T2.groupby(['Season', 'T2_TeamID'])['win'].mean().reset_index(
        name='T2_win_ratio_14d')

    tourney_data = pd.merge(tourney_data, last14days_stats_T1, on=['Season', 'T1_TeamID'], how='left')
    tourney_data = pd.merge(tourney_data, last14days_stats_T2, on=['Season', 'T2_TeamID'], how='left')

    return tourney_data


def add_win_loss_feats(regular_data, tourney_data):
    num_games = regular_data.groupby(['Season', 'T1_TeamID']).count().reset_index()
    num_games = num_games[['Season', 'T1_TeamID', 'DayNum']].rename(
        columns={"DayNum": "NumGames", "T1_TeamID": "TeamID"})

    num_win = regular_data[regular_data['PointDiff'] > 0].groupby(['Season', 'T1_TeamID']).count().reset_index()
    num_win = num_win[['Season', 'T1_TeamID', 'DayNum']].rename(columns={"DayNum": "NumWins", "T1_TeamID": "TeamID"})

    num_loss = regular_data[regular_data['PointDiff'] < 0].groupby(['Season', 'T1_TeamID']).count().reset_index()
    num_loss = num_loss[['Season', 'T1_TeamID', 'DayNum']].rename(
        columns={"DayNum": "NumLosses", "T1_TeamID": "TeamID"})

    win_loss_stat = pd.merge(num_games, num_win, on=['Season', 'TeamID'], how='left')
    win_loss_stat = pd.merge(win_loss_stat, num_loss, on=['Season', 'TeamID'], how='left')
    win_loss_stat = win_loss_stat.fillna(0)

    win_loss_stat['WinRatio'] = win_loss_stat['NumWins'] / win_loss_stat['NumGames']

    win_loss_stat_T1 = win_loss_stat.copy()
    win_loss_stat_T1.columns = ['Season', 'T1_TeamID', 'T1_NumGames', 'T1_NumWins', 'T1_NumLosses', 'T1_WinRatio']
    win_loss_stat_T2 = win_loss_stat.copy()
    win_loss_stat_T2.columns = ['Season', 'T2_TeamID', 'T2_NumGames', 'T2_NumWins', 'T2_NumLosses', 'T2_WinRatio']

    tourney_data = pd.merge(tourney_data, win_loss_stat_T1, on=['Season', 'T1_TeamID'], how='left')
    tourney_data = pd.merge(tourney_data, win_loss_stat_T2, on=['Season', 'T2_TeamID'], how='left')

    return tourney_data


def add_seeds_feats(tourney_data, seeds):
    seeds['Playin_seed'] = np.where(seeds['Seed'].str.len() > 3, 1, 0)
    seeds['seed'] = seeds['Seed'].apply(lambda x: int(x[1:3]))

    seeds_T1 = seeds[['Season', 'TeamID', 'seed', 'Playin_seed']].copy()
    seeds_T2 = seeds[['Season', 'TeamID', 'seed', 'Playin_seed']].copy()
    seeds_T1.columns = ['Season', 'T1_TeamID', 'T1_seed', 'T1_Playin_seed']
    seeds_T2.columns = ['Season', 'T2_TeamID', 'T2_seed', 'T2_Playin_seed']

    tourney_data = pd.merge(tourney_data, seeds_T1, on=['Season', 'T1_TeamID'], how='left')
    tourney_data = pd.merge(tourney_data, seeds_T2, on=['Season', 'T2_TeamID'], how='left')

    return tourney_data


def add_massey_ordinals_feats(MMasseyOrdinals, tourney_data):
    massey_ordinals = MMasseyOrdinals[MMasseyOrdinals['RankingDayNum'] == 128].reset_index(drop=True)
    massey_ordinals = massey_ordinals.groupby(['Season', 'TeamID'])['OrdinalRank'].agg(
        ['mean', 'max', 'min']).reset_index()

    massey_ordinals_T1 = massey_ordinals.copy()
    massey_ordinals_T2 = massey_ordinals.copy()
    massey_ordinals_T1.columns = ['Season', 'T1_TeamID', 'T1_ordinal_rank_mean', 'T1_ordinal_rank_max',
                                  'T1_ordinal_rank_min']
    massey_ordinals_T2.columns = ['Season', 'T2_TeamID', 'T2_ordinal_rank_mean', 'T2_ordinal_rank_max',
                                  'T2_ordinal_rank_min']

    tourney_data = pd.merge(tourney_data, massey_ordinals_T1, on=['Season', 'T1_TeamID'], how='left')
    tourney_data = pd.merge(tourney_data, massey_ordinals_T2, on=['Season', 'T2_TeamID'], how='left')

    return tourney_data


def add_diff_feats(tourney_data, data_type='detailed'):
    box_score_names = ['Score', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA',
                       'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']
    if data_type == 'compact':
        box_score_names = []

    box_score_other_names = ['PointDiff']
    box_score_opponent_names = ['opponent_' + col for col in box_score_names]
    box_score_reg_names = ['reg_' + col for col in box_score_names + box_score_opponent_names + box_score_other_names]
    other_feats_names = ['win_ratio_14d', 'seed', 'ordinal_rank_mean', 'ordinal_rank_max', 'ordinal_rank_min',
                         'NumGames', 'NumWins', 'NumLosses', 'WinRatio',
                         'PointDiff_Rating', 'OE_Rating', 'DE_Rating', 'Pace_Rating']
    if data_type == 'compact':
        other_feats_names = ['win_ratio_14d', 'seed', 'NumGames', 'NumWins', 'NumLosses', 'WinRatio']

    cols_to_diff = box_score_reg_names + other_feats_names

    for col in cols_to_diff:
        tourney_data[col + '_Diff'] = tourney_data['T1_' + col] - tourney_data['T2_' + col]

    return tourney_data


def merge_team_ratings(tourney_df, point_diff_df, oe_df, de_df, pace_df):
    point_diff_df = point_diff_df[['TeamID', 'Season', 'PointDiff_Rating']]
    oe_df = oe_df[['TeamID', 'Season', 'OE_Rating']]
    de_df = de_df[['TeamID', 'Season', 'DE_Rating']]
    pace_df = pace_df[['TeamID', 'Season', 'Pace_Rating']]

    df = tourney_df.merge(point_diff_df,
                          left_on=['Season', 'T1_TeamID'],
                          right_on=['Season', 'TeamID'],
                          how='left')
    df.rename(columns={'PointDiff_Rating': 'T1_PointDiff_Rating'}, inplace=True)
    df.drop(columns='TeamID', inplace=True)

    df = df.merge(oe_df, left_on=['Season', 'T1_TeamID'],
                  right_on=['Season', 'TeamID'], how='left')
    df.rename(columns={'OE_Rating': 'T1_OE_Rating'}, inplace=True)
    df.drop(columns='TeamID', inplace=True)

    df = df.merge(de_df, left_on=['Season', 'T1_TeamID'],
                  right_on=['Season', 'TeamID'], how='left')
    df.rename(columns={'DE_Rating': 'T1_DE_Rating'}, inplace=True)
    df.drop(columns='TeamID', inplace=True)

    df = df.merge(pace_df, left_on=['Season', 'T1_TeamID'],
                  right_on=['Season', 'TeamID'], how='left')
    df.rename(columns={'Pace_Rating': 'T1_Pace_Rating'}, inplace=True)
    df.drop(columns='TeamID', inplace=True)

    df = df.merge(point_diff_df,
                  left_on=['Season', 'T2_TeamID'],
                  right_on=['Season', 'TeamID'],
                  how='left')
    df.rename(columns={'PointDiff_Rating': 'T2_PointDiff_Rating'}, inplace=True)
    df.drop(columns='TeamID', inplace=True)

    df = df.merge(oe_df, left_on=['Season', 'T2_TeamID'],
                  right_on=['Season', 'TeamID'], how='left')
    df.rename(columns={'OE_Rating': 'T2_OE_Rating'}, inplace=True)
    df.drop(columns='TeamID', inplace=True)

    df = df.merge(de_df, left_on=['Season', 'T2_TeamID'],
                  right_on=['Season', 'TeamID'], how='left')
    df.rename(columns={'DE_Rating': 'T2_DE_Rating'}, inplace=True)
    df.drop(columns='TeamID', inplace=True)

    df = df.merge(pace_df, left_on=['Season', 'T2_TeamID'],
                  right_on=['Season', 'TeamID'], how='left')
    df.rename(columns={'Pace_Rating': 'T2_Pace_Rating'}, inplace=True)
    df.drop(columns='TeamID', inplace=True)

    return df