"""Main pipeline for NCAA March Madness 2025 prediction."""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

from modules.data_preparation import prepare_data
from modules.team_ratings import (get_team_pointdiff_rating, get_team_oe_rating,
                                  get_team_de_rating, get_team_pace_rating)
from modules.feature_engineering import (add_possesions_feats, add_boxcore_feats,
                                         add_last_14_days_feats, add_win_loss_feats,
                                         add_seeds_feats, add_massey_ordinals_feats,
                                         add_diff_feats, merge_team_ratings)
from modules.predictions import train_and_predict_oe, train_and_predict_pace
from modules.validation import season_cv, log_season, get_xgb_cv_best_iterations
from modules.models import get_target_and_features, get_xgb_params, MENS_FEATURES, WOMENS_FEATURES


def main():
    print("NCAA March Madness 2025 Prediction Pipeline")
    print("=" * 50)

    # 1. Load data
    print("\n1. Loading data...")
    path = 'data/'

    # Tournament results
    WNCAATourneyCompactResults = pd.read_csv(path + 'WNCAATourneyCompactResults.csv')
    MNCAATourneyCompactResults = pd.read_csv(path + 'MNCAATourneyCompactResults.csv')
    WNCAATourneyDetailedResults = pd.read_csv(path + 'WNCAATourneyDetailedResults.csv')
    MNCAATourneyDetailedResults = pd.read_csv(path + 'MNCAATourneyDetailedResults.csv')

    # Seeds
    WNCAATourneySeeds = pd.read_csv(path + 'WNCAATourneySeeds.csv')
    MNCAATourneySeeds = pd.read_csv(path + 'MNCAATourneySeeds.csv')

    # Regular season results
    WRegularSeasonDetailedResults = pd.read_csv(path + 'WRegularSeasonDetailedResults.csv')
    MRegularSeasonDetailedResults = pd.read_csv(path + 'MRegularSeasonDetailedResults.csv')
    WRegularSeasonCompactResults = pd.read_csv(path + 'WRegularSeasonCompactResults.csv')
    MRegularSeasonCompactResults = pd.read_csv(path + 'MRegularSeasonCompactResults.csv')

    # Massey Ordinals
    MMasseyOrdinals = pd.read_csv(path + 'MMasseyOrdinals.csv')

    # Combine men's and women's data
    tourney_compact_results = pd.concat([MNCAATourneyCompactResults, WNCAATourneyCompactResults], ignore_index=True)
    tourney_detailed_results = pd.concat([MNCAATourneyDetailedResults, WNCAATourneyDetailedResults], ignore_index=True)
    seeds = pd.concat([MNCAATourneySeeds, WNCAATourneySeeds], ignore_index=True)
    regular_compact_results = pd.concat([MRegularSeasonCompactResults, WRegularSeasonCompactResults], ignore_index=True)
    regular_detailed_results = pd.concat([MRegularSeasonDetailedResults, WRegularSeasonDetailedResults],
                                         ignore_index=True)

    print("Data loaded successfully!")

    # 2. Prepare data
    print("\n2. Preparing data...")
    regular_detailed_data = prepare_data(regular_detailed_results, data_type='Detailed')
    tourney_detailed_data = prepare_data(tourney_detailed_results, data_type='Detailed')
    regular_compact_data = prepare_data(regular_compact_results, data_type='Compact')
    tourney_compact_data = prepare_data(tourney_compact_results, data_type='Compact')

    # 3. Feature engineering - Possessions
    print("\n3. Adding possession features...")
    regular_detailed_data = add_possesions_feats(regular_detailed_data)
    tourney_detailed_data = add_possesions_feats(tourney_detailed_data)

    # 4. Calculate team ratings
    print("\n4. Calculating team ratings...")
    print("Point Diff Rating: ", end='')
    point_diff_rating = get_team_pointdiff_rating(regular_compact_data)
    print("\nOE Rating: ", end='')
    oe_rating = get_team_oe_rating(regular_detailed_data)
    print("\nDE Rating: ", end='')
    de_rating = get_team_de_rating(regular_detailed_data)
    print("\nPace Rating: ", end='')
    pace_rating = get_team_pace_rating(regular_detailed_data)

    # 5. Merge ratings with tournament data
    print("\n\n5. Merging team ratings...")
    tourney_detailed_data = merge_team_ratings(tourney_detailed_data, point_diff_rating,
                                               oe_rating, de_rating, pace_rating)

    # 6. Add regular season statistics
    print("\n6. Adding regular season statistics...")
    tourney_detailed_data = add_boxcore_feats(regular_detailed_data, tourney_detailed_data, data_type='detailed')
    tourney_compact_data = add_boxcore_feats(regular_compact_data, tourney_compact_data, data_type='compact')

    # 7. Add other features
    print("\n7. Adding other features...")
    tourney_detailed_data = add_last_14_days_feats(regular_detailed_data, tourney_detailed_data)
    tourney_detailed_data = add_win_loss_feats(regular_detailed_data, tourney_detailed_data)
    tourney_detailed_data = add_seeds_feats(tourney_detailed_data, seeds)
    tourney_detailed_data = add_massey_ordinals_feats(MMasseyOrdinals, tourney_detailed_data)
    tourney_detailed_data = add_diff_feats(tourney_detailed_data, data_type='detailed')

    # 8. Add predictive features
    print("\n8. Creating predictive features...")
    T1_OE_preds = train_and_predict_oe(tourney_detailed_data, 'T1', logs=False)
    T2_OE_preds = train_and_predict_oe(tourney_detailed_data, 'T2', logs=False)
    Pace_preds = train_and_predict_pace(tourney_detailed_data, logs=False)

    tourney_detailed_data = pd.merge(tourney_detailed_data, T1_OE_preds, how='left',
                                     on=['Season', 'T1_TeamID', 'T2_TeamID'])
    tourney_detailed_data = pd.merge(tourney_detailed_data, T2_OE_preds, how='left',
                                     on=['Season', 'T1_TeamID', 'T2_TeamID'])
    tourney_detailed_data = pd.merge(tourney_detailed_data, Pace_preds, how='left',
                                     on=['Season', 'T1_TeamID', 'T2_TeamID'])
    tourney_detailed_data['OE_Pred_Diff'] = tourney_detailed_data['T1_OE_Pred'] - tourney_detailed_data['T2_OE_Pred']

    # 9. Prepare submission data
    print("\n9. Preparing submission data...")
    sub = pd.read_csv(path + 'SampleSubmissionStage2.csv')
    sub["Season"] = 2025
    sub["T1_TeamID"] = sub["ID"].apply(lambda x: x[5:9]).astype(int)
    sub["T2_TeamID"] = sub["ID"].apply(lambda x: x[10:14]).astype(int)

    # Apply all transformations to submission data
    sub = merge_team_ratings(sub, point_diff_rating, oe_rating, de_rating, pace_rating)
    sub = add_boxcore_feats(regular_detailed_data, sub, data_type='detailed', sub_data=True)
    sub = add_win_loss_feats(regular_detailed_data, sub)
    sub = add_massey_ordinals_feats(MMasseyOrdinals, sub)
    sub = add_seeds_feats(sub, seeds)
    sub = add_last_14_days_feats(regular_detailed_data, sub)
    sub = add_diff_feats(sub, data_type='detailed')

    # Add predictive features for submission
    T1_OE_sub_preds = train_and_predict_oe(tourney_detailed_data, 'T1', for_sub_data=True, sub_data=sub, logs=False)
    T2_OE_sub_preds = train_and_predict_oe(tourney_detailed_data, 'T2', for_sub_data=True, sub_data=sub, logs=False)
    Pace_sub_preds = train_and_predict_pace(tourney_detailed_data, for_sub_data=True, sub_data=sub, logs=False)

    sub = pd.merge(sub, T1_OE_sub_preds, how='left', on=['Season', 'T1_TeamID', 'T2_TeamID'])
    sub = pd.merge(sub, T2_OE_sub_preds, how='left', on=['Season', 'T1_TeamID', 'T2_TeamID'])
    sub = pd.merge(sub, Pace_sub_preds, how='left', on=['Season', 'T1_TeamID', 'T2_TeamID'])
    sub['OE_Pred_Diff'] = sub['T1_OE_Pred'] - sub['T2_OE_Pred']

    # 10. Train models
    print("\n10. Training models...")

    # Prepare target and features
    tourney_detailed_data, features_detailed = get_target_and_features(tourney_detailed_data)
    train_data = tourney_detailed_data.query('Season > 2003').reset_index(drop=True)

    # Men's model
    print("\n10.1 Training Men's Model (XGBoost)...")
    train_data_m = train_data.query('T1_TeamID < 3000').reset_index(drop=True)

    # Initialize XGBoost model
    xgb_params = get_xgb_params()
    xgb_model = xgb.XGBRegressor(**xgb_params)

    # Run cross-validation
    params = {
        "model": xgb_model,
        "train": train_data_m,
        "target": train_data_m['target'],
        "test": sub,
        "features": MENS_FEATURES,
        "start_val_year": 2015,
    }

    oof_pred, oof_true = season_cv(**params)

    # Train final model
    X_train_m = train_data_m[MENS_FEATURES].copy()
    y_train_m = train_data_m['target']
    sub_m = sub.query('T1_TeamID < 3000')
    X_test_m = sub_m[MENS_FEATURES].copy()

    xgb_model.set_params(n_estimators=1000, early_stopping_rounds=25)
    cv_best_iterations = get_xgb_cv_best_iterations(X_train_m, y_train_m, xgb_model.get_params(), repeat_cv=10)
    n_estimators = int(np.mean(cv_best_iterations) * 1.05)
    xgb_model.set_params(n_estimators=n_estimators, early_stopping_rounds=None)

    xgb_model.fit(X_train_m, y_train_m)
    y_test_m_pred = xgb_model.predict(X_test_m)

    # Calibrate predictions
    X_log = np.array(oof_pred).reshape(-1, 1)
    y_log = np.array(oof_true)
    log_model = LogisticRegression(C=10)
    log_model.fit(X_log, y_log)
    pred_m = log_model.predict_proba(y_test_m_pred.reshape(-1, 1))[:, 1]
    sub_m.loc[:, 'Pred'] = pred_m

    # Women's model
    print("\n10.2 Training Women's Model (Logistic Regression)...")
    train_data_w = train_data.query('T1_TeamID >= 3000').reset_index(drop=True)

    log_reg_model = LogisticRegression(C=10, max_iter=10000)
    log_season(log_reg_model, train_data_w, train_data_w['target_binary'], WOMENS_FEATURES,
               start_val_year=2015, end_val_year=2024)

    # Train final model
    X_train_w = train_data_w[WOMENS_FEATURES].copy()
    y_train_w = train_data_w['target_binary']
    sub_w = sub.query('T1_TeamID >= 3000')
    X_test_w = sub_w[WOMENS_FEATURES].fillna(0).copy()

    scaler = MinMaxScaler()
    X_train_w = scaler.fit_transform(X_train_w)
    X_test_w = scaler.transform(X_test_w)

    log_reg_model = LogisticRegression(C=10, max_iter=10000)
    log_reg_model.fit(X_train_w, y_train_w)
    pred_w = log_reg_model.predict_proba(X_test_w)[:, 1]
    sub_w.loc[:, 'Pred'] = pred_w

    # 11. Create submissions
    print("\n11. Creating submission files...")

    # Combine predictions
    submission = pd.concat([sub_m, sub_w])

    # Create three submission files
    submission_2 = submission.copy()
    submission_3 = submission.copy()

    # Apply 0/1 flip for Baylor vs Mississippi St
    submission_2.loc[(submission_2.T1_TeamID == 1124) & (submission_2.T2_TeamID == 1280), 'Pred'] = 0
    submission_3.loc[(submission_3.T1_TeamID == 1124) & (submission_3.T2_TeamID == 1280), 'Pred'] = 1

    # Save submissions
    submission[['ID', 'Pred']].to_csv("submissions/ncaa-2025-submission-1.csv", index=None)
    submission_2[['ID', 'Pred']].to_csv("submissions/ncaa-2025-submission-2.csv", index=None)
    submission_3[['ID', 'Pred']].to_csv("submissions/ncaa-2025-submission-3.csv", index=None)

    print("\nPipeline completed successfully!")
    print("Submission files saved to submissions/ directory")


if __name__ == "__main__":
    main()