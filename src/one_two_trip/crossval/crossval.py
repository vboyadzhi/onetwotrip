import pandas as pd
import numpy as np
import lightgbm as lgb
import logging
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from hyperopt import  STATUS_OK, STATUS_RUNNING
import datetime

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedStratifiedKFold


class CV_score:
    def __init__(self, params,
                 cols_all,
                 col_target,
                 col_client,
                 cols_cat='auto',
                 num_boost_round=99999,
                 early_stopping_rounds=50,
                 valid=True,
                 n_splits=10,
                 n_repeats=1,
                 cv_byclient=False
                 ):
        self.params = params
        self.cols_all = cols_all
        self.col_target = col_target
        self.cols_cat = cols_cat
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.valid = valid
        self.models = {}
        self.scores = None
        self.score_max = None
        self.num_boost_optimal = None
        self.std = None
        self.n_repeats = n_repeats
        self.n_splits = n_splits
        self.col_client=col_client
        self.cv_byclient=cv_byclient

    def fit(self, df):
        log = logging.getLogger(__name__)
        scores = {}
        scores_avg = []
        log.info(self.params)

        x = df[self.cols_all]
        y = df[self.col_target]


        if self.cv_byclient:
            self._unique_clients = df[self.col_client].unique()
            self.model_validation = KFold(n_splits = self.n_splits, shuffle = True, random_state=0)
            self.model_validation.get_n_splits(self._unique_clients)
            self._split = self._unique_clients
        else:
            self.model_validation = RepeatedStratifiedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=0)
            self._split = np.zeros(x.shape), y

        # for fold in fold_list:
        for fold, (train_idx, val_idx) in enumerate(self.model_validation.split(self._split)):
            if self.cv_byclient:
                val_idx = df[self.col_client].isin(self._unique_clients[val_idx])
                train_idx = df[self.col_client].isin(self._unique_clients[train_idx])
            else:
                val_idx = df.index.isin(df.iloc[val_idx].index)
                train_idx = df.index.isin(df.iloc[train_idx].index)

            X_train, X_val = x.loc[train_idx], x.loc[val_idx]
            y_train, y_val = y.loc[train_idx], y.loc[val_idx]

            dtrain = lgb.Dataset(data=X_train.astype(float), label=y_train,
                                 categorical_feature=self.cols_cat)
            dvalid = lgb.Dataset(data=X_val.astype(float), label=y_val,
                                 categorical_feature=self.cols_cat)

            log.info(f'CROSSVALIDATION FOLD {fold} START')

            # Обучение
            evals_result = {}
            if self.valid:
                model = lgb.train(params=self.params,
                                  train_set=dtrain,
                                  valid_sets=[dtrain, dvalid],
                                  valid_names=['train', 'eval'],
                                  num_boost_round=self.num_boost_round,
                                  evals_result=evals_result,
                                  categorical_feature=self.cols_cat,
                                  early_stopping_rounds=self.early_stopping_rounds,
                                  verbose_eval=False)
            else:
                model = lgb.train(params=self.params,
                                  train_set=dtrain,
                                  num_boost_round=self.num_boost_round,
                                  categorical_feature=self.cols_cat,
                                  verbose_eval=False)

            self.models[fold] = model
            if self.valid:
                # Построение прогнозов при разном виде взаимодействия
                scores[fold] = evals_result['eval']['auc']
                scores_avg.append(np.max(evals_result['eval']['auc']))

        if self.valid:
            self.scores = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in scores.items()]))
            mask = self.scores.isnull().sum(axis=1) == 0
            self.num_boost_optimal = np.argmax(self.scores[mask].mean(axis=1))
            self.score_max = self.scores[mask].mean(axis=1)[self.num_boost_optimal]
            # self.score_max = np.mean(scores_avg)
            self.std = self.scores[mask].std(axis=1)[self.num_boost_optimal]
            # self.std = np.std(scores_avg)

            result = {'loss': -self.score_max,
                      'status': STATUS_OK,
                      'std': self.std,
                      'score_max': self.score_max,
                      'scores_all': scores_avg,
                      'num_boost': int(self.num_boost_optimal),
                      }
            log.info(result)
            return result
        return self

    def transform_train(self, df):

        x = df[self.cols_all]
        y = df[self.col_target]

        for fold, (train_idx, val_idx) in enumerate(self.model_validation.split(self._split)):
            if self.cv_byclient:
                val_idx = df[self.col_client].isin(self._unique_clients[val_idx])
                train_idx = df[self.col_client].isin(self._unique_clients[train_idx])
            else:
                val_idx = df.index.isin(df.iloc[val_idx].index)
                train_idx = df.index.isin(df.iloc[train_idx].index)

            X_train, X_val = x.loc[train_idx], x.loc[val_idx]
            y_train, y_val = y.loc[train_idx], y.loc[val_idx]

            # Подготовка данных в нужном формате
            model = self.models[fold]
            df.loc[X_val.index, 'PREDICT'] = \
                model.predict(X_val[model.feature_name()].astype(float),
                              num_iteration=self.num_boost_optimal) / self.n_repeats

        return df['PREDICT']

    def transform_test(self, test):
        models_len = len(self.models.keys())

        test['PREDICT'] = 0
        for fold in self.models.keys():
            model = self.models[fold]
            test['PREDICT'] += model.predict(test[model.feature_name()].astype(float),
                                             num_iteration=self.num_boost_optimal) / models_len

        return test['PREDICT']

    def shap(self, df: pd.DataFrame):
        fig = plt.figure(figsize=(10, 10))
        log = logging.getLogger(__name__)
        shap_df_fin = pd.DataFrame(columns=['feature'])


        x = df[self.cols_all]
        y = df[self.col_target]

        for fold, (train_idx, val_idx) in enumerate(self.model_validation.split(self._split)):
            if self.cv_byclient:
                val_idx = df[self.col_client].isin(self._unique_clients[val_idx])
                train_idx = df[self.col_client].isin(self._unique_clients[train_idx])
            else:
                val_idx = df.index.isin(df.iloc[val_idx].index)
                train_idx = df.index.isin(df.iloc[train_idx].index)

            X_train, X_val = x.loc[train_idx], x.loc[val_idx]
            y_train, y_val = y.loc[train_idx], y.loc[val_idx]

            model = self.models[fold]
            explainer = shap.TreeExplainer(model)
            df_sample = X_val[model.feature_name()].sample(n=500, random_state=0, replace=True).astype(float)
            shap_values = explainer.shap_values(df_sample)[1]
            shap_df = pd.DataFrame(zip(model.feature_name(), np.mean(np.abs(shap_values), axis=0)),
                                   columns=['feature', 'shap_' + str(fold)])
            shap_df_fin = pd.merge(shap_df_fin, shap_df, how='outer', on='feature')

        shap_feature_stats = shap_df_fin.set_index('feature').agg(['mean', 'std'], axis=1).sort_values('mean',
                                                                                                       ascending=False)
        cols_best = shap_feature_stats[:30].index

        best_features = shap_df_fin.loc[shap_df_fin['feature'].isin(cols_best)]
        best_features_melt = pd.melt(best_features, id_vars=['feature'],
                                     value_vars=[feature for feature in best_features.columns.values.tolist() if
                                                 feature not in ['feature']])

        sns.barplot(x='value', y='feature', data=best_features_melt, estimator=np.mean, order=cols_best)
        return fig, shap_feature_stats

    def shap_summary_plot(self, test: pd.DataFrame):
        fig = plt.figure()
        log = logging.getLogger(__name__)
        shap_df_fin = pd.DataFrame(columns=['feature'])


        # Подготовка данных в нужном формате
        model = self.models[0]
        explainer = shap.TreeExplainer(model)
        df_sample = test[model.feature_name()].sample(n=500, random_state=0, replace=True).astype(float)
        shap_values = explainer.shap_values(df_sample)[1]
        shap_df = pd.DataFrame(zip(model.feature_name(), np.mean(np.abs(shap_values), axis=0)),
                               columns=['feature', 'shap_'])
        shap_df_fin = pd.merge(shap_df_fin, shap_df, how='outer', on='feature')

        shap.summary_plot(shap_values, df_sample, show=False, )
        return fig


def adversarial(params,
                    df,
                    cols_all,
                    col_target='target_adversarial',
                    cols_cat='auto',
                    test_size=0.2,
                    shap_flg=True):
    fig = plt.figure(figsize=(10, 10))
    log = logging.getLogger(__name__)
    log.info('ADVERSARIAL VALIDATION: START')
    x_train, x_test, y_train, y_test = train_test_split(df[cols_all].astype(float), df[col_target], test_size=test_size,
                                                        random_state=0)
    x_train = pd.DataFrame(x_train, columns=cols_all)
    x_test = pd.DataFrame(x_test, columns=cols_all)
    dtrain = lgb.Dataset(data=x_train.astype(float), label=y_train,
                         categorical_feature=cols_cat)
    dvalid = lgb.Dataset(data=x_test.astype(float), label=y_test,
                         categorical_feature=cols_cat)

    evals_result = {}
    model = lgb.train(params=params,
                    train_set=dtrain,
                    valid_sets=[dtrain, dvalid],
                    valid_names=['train', 'eval'],
                    evals_result=evals_result,
                    num_boost_round=100,
                    early_stopping_rounds=50,
                    categorical_feature=cols_cat,
                    verbose_eval=False)
    results = pd.Series(evals_result['eval']['auc'])
    score_max = results.max()
    num_boost_round_max = np.argmax(results)


    log.info(f'ADVERSARIAL VALIDATION SCORE: {score_max}')
    log.info(f'ADVERSARIAL VALIDATION NUMBOOST: {num_boost_round_max}')

    if shap_flg:
        explainer = shap.TreeExplainer(model)
        df_sample = x_test[model.feature_name()].sample(n=min(1000, x_test.shape[0]),
                                                        random_state=0).astype(
            float)
        shap_values = explainer.shap_values(df_sample)[1]
        shap.summary_plot(shap_values, df_sample, show=False, )
        return [{'score_max':score_max}, fig]
    log.info('ADVERSARIAL VALIDATION: END')
    return {'score_max': score_max}


def train_final(params,
                df: pd.DataFrame,
                output,
                cols_all,
                numboost,
                col_target,
                cols_cat='auto' ) -> None:
    log = logging.getLogger(__name__)

    dtrain = lgb.Dataset(data=df[cols_all].astype(float), label=df[col_target],
                         categorical_feature=cols_cat)

    model = lgb.train(params=params,
                      train_set=dtrain,
                      categorical_feature=cols_cat,
                      num_boost_round=numboost,
                      verbose_eval=False)

    # dt_now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    model.save_model(output)

    return True