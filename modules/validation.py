import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import log_loss
from modules.metrics import brier_score


def get_xgb_cv_best_iterations(X, y, params, repeat_cv=3, n_splits=5, eval_metric='mae'):
    dtrain = xgb.DMatrix(X, y)
    xgb_cv = []
    for i in range(repeat_cv):
        xgb_cv.append(
            xgb.cv(params=params, dtrain=dtrain, num_boost_round=params['n_estimators'],
                   folds=KFold(n_splits=n_splits, shuffle=True, random_state=i),
                   early_stopping_rounds=25, verbose_eval=0))

    best_iterations = [np.argmin(x[f'test-{eval_metric}-mean'].values) for x in xgb_cv]

    return best_iterations


def season_cv(model, train, target, test, features, start_val_year):
    seasons = train['Season'].unique()
    val_seasons = [i for i in seasons if i >= start_val_year]
    oof_pred, oof_true = [], []
    season_brier_scores, season_log_losses = [], []

    for season in seasons[3:]:
        train_fold = train[train['Season'] < season].copy()
        val_fold = train[train['Season'] == season].copy()
        X_train, X_val = train_fold[features].copy(), val_fold[features].copy()
        y_train, y_val = target[train_fold.index].copy(), target[val_fold.index].copy()
        y_val_binary = np.where(y_val > 0, 1, 0)

        cv_best_iterations = get_xgb_cv_best_iterations(X_train, y_train, model.get_params())
        model.set_params(n_estimators=int(np.mean(cv_best_iterations) * 1.05), early_stopping_rounds=None)
        model.fit(X_train, y_train, verbose=0)
        model.set_params(n_estimators=1000, early_stopping_rounds=25)

        y_val_pred = model.predict(X_val)

        if season not in val_seasons:
            oof_pred.extend(y_val_pred)
            oof_true.extend(y_val_binary)

        if season in val_seasons:
            X_log = np.array(oof_pred).reshape(-1, 1)
            y_log = np.array(oof_true)
            log_model = LogisticRegression(C=10)
            log_model.fit(X_log, y_log)
            glm_y_val_pred = log_model.predict_proba(y_val_pred.reshape(-1, 1))[:, 1]

            season_brier_score = brier_score(y_val_binary, glm_y_val_pred)
            season_brier_scores.append(season_brier_score)
            season_log_loss = log_loss(y_val_binary, glm_y_val_pred)
            season_log_losses.append(season_log_loss)
            print(f'Season {season} brier score = {season_brier_score:.5f}. log loss = {season_log_loss:.5f}')
            oof_pred.extend(y_val_pred)
            oof_true.extend(y_val_binary)

    print(f'Seasons mean brier score = {np.mean(season_brier_scores):.5f}. ', end='')
    print(f'log loss = {np.mean(season_log_losses):.5f}')

    return oof_pred, oof_true


def log_season(model, train, target, features, start_val_year=2015, end_val_year=2024):
    seasons = train['Season'].unique()
    val_seasons = [year for year in seasons if start_val_year <= year <= end_val_year]
    cvs_brier_w, cvs_log_w = [], []

    for season in val_seasons:
        past_seasons = sorted(seasons[seasons < season])
        train_idx, val_idx = train['Season'].isin(past_seasons), train['Season'] == season
        X_train = train[features][train_idx].reset_index(drop=True).copy()
        X_val = train[features][val_idx].reset_index(drop=True).copy()
        y_train, y_val = target[train_idx], np.array(target[val_idx])

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        model.fit(X_train, y_train)
        y_val_pred = model.predict_proba(X_val)[:, 1]

        season_log_loss_w = log_loss(y_val, y_val_pred)
        season_brier_score_w = brier_score(y_val, y_val_pred)
        cvs_brier_w.append(season_brier_score_w)
        cvs_log_w.append(season_log_loss_w)

        print(f'Season {season}:   ', end='')
        print(f'log loss w = {season_log_loss_w:.5f}   ', end='')
        print(f'brier score w = {season_brier_score_w:.5f}')

    print(f'\nMean seasons brier score w: {np.mean(cvs_brier_w):.5f}')
    print(f'Mean seasons log loss w: {np.mean(cvs_log_w):.5f}\n')