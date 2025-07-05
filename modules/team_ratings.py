import pandas as pd
from sklearn.linear_model import Ridge


def get_team_pointdiff_rating(reg_data):
    all_seasons = sorted(reg_data['Season'].unique())
    list_ratings = []

    for season in all_seasons:
        print(season, end=', ')
        sub_data = reg_data[reg_data['Season'] == season].copy()

        team1_dummies = pd.get_dummies(sub_data['T1_TeamID'], prefix='T1')
        team2_dummies = pd.get_dummies(sub_data['T2_TeamID'], prefix='T2')

        X = pd.concat([team1_dummies, team2_dummies, sub_data['location']], axis=1)
        y = sub_data['PointDiff']

        ridge_model = Ridge(alpha=10.0)
        ridge_model.fit(X, y)

        coef_series = pd.Series(ridge_model.coef_, index=X.columns)
        intercept = ridge_model.intercept_
        home_coef = coef_series['location']
        t1_coefs = coef_series[coef_series.index.str.startswith('T1_')]
        t2_coefs = coef_series[coef_series.index.str.startswith('T2_')]

        df_t1 = pd.DataFrame({
            'TeamID': t1_coefs.index.str.replace('T1_', '').astype(int),
            'T1_coef': t1_coefs.values
        })
        df_t2 = pd.DataFrame({
            'TeamID': t2_coefs.index.str.replace('T2_', '').astype(int),
            'T2_coef': t2_coefs.values
        })

        merged = pd.merge(df_t1, df_t2, on='TeamID', how='outer')
        merged['Season'] = season
        merged['HomeAdvantage'] = home_coef
        merged['Intercept'] = intercept
        merged['PointDiff_Rating'] = (merged['T1_coef'] - merged['T2_coef']) / 2.0
        list_ratings.append(merged)

    ratings_df = pd.concat(list_ratings, ignore_index=True)

    return ratings_df


def get_team_oe_rating(reg_data):
    all_seasons = sorted(reg_data['Season'].unique())
    list_ratings = []

    for season in all_seasons:
        print(season, end=', ')
        sub_data = reg_data[reg_data['Season'] == season].copy()

        team1_dummies = pd.get_dummies(sub_data['T1_TeamID'], prefix='T1')
        team2_dummies = pd.get_dummies(sub_data['T2_TeamID'], prefix='T2')

        X = pd.concat([team1_dummies, team2_dummies, sub_data['location']], axis=1)
        y = sub_data['T1_OE']

        ridge_model = Ridge(alpha=10.0)
        ridge_model.fit(X, y)

        coef_series = pd.Series(ridge_model.coef_, index=X.columns)
        intercept = ridge_model.intercept_
        loc_coef = coef_series['location']

        t1_coefs = coef_series[coef_series.index.str.startswith('T1_')]
        t2_coefs = coef_series[coef_series.index.str.startswith('T2_')]

        df_t1 = pd.DataFrame({
            'TeamID': t1_coefs.index.str.replace('T1_', '').astype(int),
            'T1_coef': t1_coefs.values
        })
        df_t2 = pd.DataFrame({
            'TeamID': t2_coefs.index.str.replace('T2_', '').astype(int),
            'T2_coef': t2_coefs.values
        })

        merged = pd.merge(df_t1, df_t2, on='TeamID', how='outer')
        merged['Season'] = season
        merged['LocationCoef'] = loc_coef
        merged['Intercept'] = intercept
        merged['OE_Rating'] = (merged['T1_coef'] - merged['T2_coef']) / 2.0 + merged['Intercept']
        list_ratings.append(merged)

    ratings_df = pd.concat(list_ratings, ignore_index=True)
    return ratings_df


def get_team_de_rating(reg_data):
    all_seasons = sorted(reg_data['Season'].unique())
    list_ratings = []

    for season in all_seasons:
        print(season, end=', ')
        sub_data = reg_data[reg_data['Season'] == season].copy()

        team1_dummies = pd.get_dummies(sub_data['T1_TeamID'], prefix='T1')
        team2_dummies = pd.get_dummies(sub_data['T2_TeamID'], prefix='T2')

        X = pd.concat([team1_dummies, team2_dummies, sub_data['location']], axis=1)
        y = sub_data['T1_DE']

        ridge_model = Ridge(alpha=10.0)
        ridge_model.fit(X, y)

        coef_series = pd.Series(ridge_model.coef_, index=X.columns)
        intercept = ridge_model.intercept_
        loc_coef = coef_series['location']

        t1_coefs = coef_series[coef_series.index.str.startswith('T1_')]
        t2_coefs = coef_series[coef_series.index.str.startswith('T2_')]

        df_t1 = pd.DataFrame({
            'TeamID': t1_coefs.index.str.replace('T1_', '').astype(int),
            'T1_coef': t1_coefs.values
        })
        df_t2 = pd.DataFrame({
            'TeamID': t2_coefs.index.str.replace('T2_', '').astype(int),
            'T2_coef': t2_coefs.values
        })

        merged = pd.merge(df_t1, df_t2, on='TeamID', how='outer')
        merged['Season'] = season
        merged['LocationCoef'] = loc_coef
        merged['Intercept'] = intercept
        merged['DE_Rating'] = (merged['T1_coef'] - merged['T2_coef']) / 2.0 + merged['Intercept']
        list_ratings.append(merged)

    ratings_df = pd.concat(list_ratings, ignore_index=True)
    return ratings_df


def get_team_pace_rating(reg_data):
    all_seasons = sorted(reg_data['Season'].unique())
    list_ratings = []

    for season in all_seasons:
        print(season, end=', ')
        sub_data = reg_data[reg_data['Season'] == season].copy()

        team1_dummies = pd.get_dummies(sub_data['T1_TeamID'], prefix='T1')
        team2_dummies = pd.get_dummies(sub_data['T2_TeamID'], prefix='T2')

        X = pd.concat([team1_dummies, team2_dummies, sub_data['location']], axis=1)
        y = sub_data['Pace']

        ridge_model = Ridge(alpha=10.0)
        ridge_model.fit(X, y)

        coef_series = pd.Series(ridge_model.coef_, index=X.columns)
        intercept = ridge_model.intercept_
        loc_coef = coef_series['location']

        t1_coefs = coef_series[coef_series.index.str.startswith('T1_')]
        t2_coefs = coef_series[coef_series.index.str.startswith('T2_')]

        df_t1 = pd.DataFrame({
            'TeamID': t1_coefs.index.str.replace('T1_', '').astype(int),
            'T1_coef': t1_coefs.values
        })
        df_t2 = pd.DataFrame({
            'TeamID': t2_coefs.index.str.replace('T2_', '').astype(int),
            'T2_coef': t2_coefs.values
        })

        merged = pd.merge(df_t1, df_t2, on='TeamID', how='outer')
        merged['Season'] = season
        merged['LocationCoef'] = loc_coef
        merged['Intercept'] = intercept
        merged['Pace_Rating'] = (merged['T1_coef'] + merged['T2_coef']) / 2.0 + merged['Intercept']
        list_ratings.append(merged)

    ratings_df = pd.concat(list_ratings, ignore_index=True)
    return ratings_df