import os

N_FOLDS = 10
N_ITEMS = 10
DATA_PATH = os.path.join(os.path.dirname(__file__), os.path.pardir, 'data')
IMAGE_PATH = os.path.join(os.path.dirname(__file__), os.path.pardir, 'images')
RANKING_PATH = os.path.join(DATA_PATH, 'ranking_test')
MODEL_PATH = os.path.join(DATA_PATH, 'models')
DATASET_NAME = 'path_set'
ITEMS_DATA_PATH_TPL = os.path.join(DATA_PATH, '{dataset}_items.csv')
ITEMS_FEATURE_PATH_TPL = os.path.join(DATA_PATH, '{dataset}_features_{i}.csv')
DATA_PATH_TPL = os.path.join(DATA_PATH, '{dataset}_{type}_fold_{fold}.csv')
TRAIN_DATA_PATH_TPL = os.path.join(DATA_PATH, '{dataset}_train_fold_{fold}.csv')
TEST_DATA_PATH_TPL = os.path.join(DATA_PATH, '{dataset}_test_fold_{fold}.csv')
MODEL_PATH_TPL = os.path.join(MODEL_PATH, '{dataset}_{model}_fold_{fold}.csv')
GROUND_TRUTH_DATA_PATH_TPL = os.path.join(
    RANKING_PATH, '{dataset}_gt_fold_{fold}.csv')
PARTIAL_DATA_PATH_TPL = os.path.join(
    RANKING_PATH, '{dataset}_partial_fold_{fold}.csv')
RANKING_MODEL_PATH_TPL = os.path.join(
    RANKING_PATH, '{dataset}_{model}_fold_{fold}.csv')
NCE_DATA_PATH_TPL = os.path.join(
        DATA_PATH, '{dataset}_nce_data_features_{index}_fold_{fold}.csv')
NCE_NOISE_PATH_TPL = os.path.join(
        DATA_PATH, '{dataset}_nce_noise_features_{index}_fold_{fold}.csv')
NCE_FEATURES_PATH_TPL = os.path.join(DATA_PATH, '{dataset}_nce_features_{index}.csv')
NCE_OUT_PATH_TPL = os.path.join(MODEL_PATH, '{dataset}_nce_out_features_{index}_dim_{dim}_fold_{fold}.csv')
SEED = 20150820
