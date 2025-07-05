import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error


def train_and_predict_oe(tourney_df_merged, team, for_sub_data=False, sub_data=None, logs=True):
    all_seasons = sorted(tourney_df_merged['Season'].unique())
    if for_sub_data:
        all_seasons.append(2025)
    result_list, mses = [], []

    for i, s in enumerate(all_seasons):
        if for_sub_data and s != 2025:
            continue
        if not for_sub_data and i == 0:
            continue

        train_seasons = all_seasons[:i]
        train_df = tourney_df_merged[tourney_df_merged['Season'].isin(train_seasons)]
        test_df = tourney_df_merged[tourney_df_merged['Season'] == s]
        if sub_data is not None:
            test_df = sub_data

        features = [
            'T1_PointDiff_Rating', 'T1_OE_Rating', 'T1_DE_Rating', 'T1_Pace_Rating',
            'T2_PointDiff_Rating', 'T2_OE_Rating', 'T2_DE_Rating', 'T2_Pace_Rating',
        ]
        X_train, X_test = train_df[features], test_df[features]
        y_train = train_df[f'{team}_OE']
        if sub_data is None:
            y_test = test_df[f'{team}_OE']

        model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        temp = test_df[['Season', 'T1_TeamID', 'T2_TeamID']].copy()
        temp[f'{team}_OE_Pred'] = y_pred
        result_list.append(temp)

        if sub_data is None:
            mse_val = mean_squared_error(y_test, y_pred)
            mses.append(mse_val)
            if logs:
                print(f"Season {s}: MSE = {mse_val:.4f}")

    if logs and mses:
        print(f'Mean seasons MSE = {np.mean(mses):.4f}')

    final_result = pd.concat(result_list, ignore_index=True)
    return final_result


def train_and_predict_pace(tourney_df_merged, for_sub_data=False, sub_data=None, logs=True):
    all_seasons = sorted(tourney_df_merged['Season'].unique())
    if for_sub_data:
        all_seasons.append(2025)
    result_list, mses = [], []

    for i, s in enumerate(all_seasons):
        if for_sub_data and s != 2025:
            continue
        if not for_sub_data and i == 0:
            continue

        train_seasons = all_seasons[:i]
        train_df = tourney_df_merged[tourney_df_merged['Season'].isin(train_seasons)]
        test_df = tourney_df_merged[tourney_df_merged['Season'] == s]
        if sub_data is not None:
            test_df = sub_data

        features = [
            'T1_PointDiff_Rating', 'T1_OE_Rating', 'T1_DE_Rating', 'T1_Pace_Rating',
            'T2_PointDiff_Rating', 'T2_OE_Rating', 'T2_DE_Rating', 'T2_Pace_Rating',
        ]
        X_train, X_test = train_df[features], test_df[features]
        y_train = train_df['Pace']
        if sub_data is None:
            y_test = test_df['Pace']

        model = xgb.XGBRegressor(n_estimators=50, max_depth=2, learning_rate=0.1)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        temp = test_df[['Season', 'T1_TeamID', 'T2_TeamID']].copy()
        temp['Pace_Pred'] = y_pred
        result_list.append(temp)

        if sub_data is None:
            mse_val = mean_squared_error(y_test, y_pred)
            mses.append(mse_val)
            if logs:
                print(f"Season {s}: MSE = {mse_val:.4f}")

    if logs and mses:
        print(f'Mean seasons MSE = {np.mean(mses):.4f}')

    final_result = pd.concat(result_list, ignore_index=True)
    return final_result