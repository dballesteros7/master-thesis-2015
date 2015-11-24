import os

N_FOLDS = 10
N_ITEMS = 10
DATA_PATH = '/local/workspace/master-thesis-2015/data/'
RANKING_PATH = os.path.join(DATA_PATH, 'ranking_test')
DATASET_NAME = 'path_set'
ITEMS_DATA_PATH_TPL = os.path.join(DATA_PATH, '{dataset}_items.csv')
TRAIN_DATA_PATH_TPL = os.path.join(DATA_PATH, '{dataset}_train_fold_{fold}.csv')
TEST_DATA_PATH_TPL = os.path.join(DATA_PATH, '{dataset}_test_fold_{fold}.csv')
RANKING_MODEL_PATH_TPL = os.path.join(
    RANKING_PATH, '{dataset}_{model}_fold_{fold}.pkl')
