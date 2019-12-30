import pandas as pd
import logging
import numpy as np
from one_two_trip.crossval.crossval import CV_score

from kedro.config import ConfigLoader

conf_paths = ['conf/base','conf/local']
conf_loader = ConfigLoader(conf_paths)
conf_credentials = conf_loader.get('credentials*','credentials*/**')
conf_parameters = conf_loader.get('parameters*','parameters*/**')
conf_catalog = conf_loader.get('catalog*', 'catalog*/**')

cols_target = conf_parameters['model']['cols_target']
col_id = conf_parameters['model']['col_id']
col_client = conf_parameters['model']['col_client']
cols_cat = conf_parameters['model']['cols_cat']
cv_byclient = conf_parameters['model']['cv_byclient']
n_splits = conf_parameters['model']['n_splits']
n_repeats = conf_parameters['model']['n_repeats']

params = conf_parameters['lightgbm']['params']

def union_data_node(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    df_union = pd.concat([train, test], sort=False, ignore_index=True)
    return df_union

def mean_byuser_node(df_union: pd.DataFrame) -> pd.DataFrame:
    cols = list(set(df_union.columns) - set(cols_target) - set([col_id]) - set([col_client]))
    result = df_union.groupby(col_client)[cols].agg('mean')
    result.columns = ['mean_byuser_' + str(i) for i in result.columns]
    return result

def get_train_node(train: pd.DataFrame, df_mean_byuser: pd.DataFrame) -> pd.DataFrame:
    result = pd.merge(train, df_mean_byuser, left_on=col_client, right_index=True)
    return result

def get_test_node(test: pd.DataFrame, df_mean_byuser: pd.DataFrame) -> pd.DataFrame:
    result = pd.merge(test, df_mean_byuser, left_on=col_client, right_index=True)
    return result

def crossval_node(df_train: pd.DataFrame):
    log = logging.getLogger(__name__)

    cols_all = df_train.columns.values
    cols_all = np.delete(cols_all, np.argwhere(cols_all == col_id))
    cols_all = np.delete(cols_all, np.argwhere(cols_all == col_client))
    cols_all = cols_all[~np.isin(cols_all, cols_target)]

    cv_models={}
    cv_results={}
    for col_target in cols_target:
        cv_models[col_target] = CV_score(params=params,
                            cols_all=cols_all,
                            col_target=col_target,
                            cols_cat=cols_cat,
                            col_client=col_client,
                            num_boost_round=999999,
                            early_stopping_rounds=50,
                            valid=True,
                            n_splits=n_splits,
                            n_repeats=n_repeats,
                            cv_byclient=cv_byclient
                            )

        cv_results[col_target] = cv_models[col_target].fit(df=df_train)

    return [cv_results, cv_models]

def test_predict_node(df_test: pd.DataFrame, cv_models):

    for col_target in cols_target:
        df_test[col_target] = cv_models[col_target].transform_test(df_test)

    return [df_test[[col_id,'goal1']], df_test[[col_id,'goal21','goal22','goal23','goal24','goal25']]]
