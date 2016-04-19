import os
import time

N_FOLDS = 8
N_ITEMS = 10
N_PHOTOS = 168607
DATA_PATH = os.path.join(os.path.dirname(__file__), os.path.pardir, 'data')
IMAGE_PATH = os.path.join(os.path.dirname(__file__), os.path.pardir, 'images')
RANKING_PATH = os.path.join(DATA_PATH, 'ranking_test')
MODEL_PATH = os.path.join(DATA_PATH, 'models')
DATASET_NAME = 'path_set'
DATASET_NAME_TPL = 'path_set_{}'
GMM_DATASET_NAME = 'gmm_photos'
LOCAL_PHOTO_CACHE = os.path.join(DATA_PATH, 'photos_{city}.csv')
CLUSTER_FILE = os.path.join(DATA_PATH, 'clusters_{id}.csv')
CLUSTER_ASSIGNATION_FILE = os.path.join(DATA_PATH, 'cluster_assignment_{id}.csv')
ITEMS_DATA_PATH_TPL = os.path.join(DATA_PATH, '{dataset}_items.csv')
CLUSTER_CENTERS_DATA_PATH_TPL = os.path.join(DATA_PATH, '{dataset}_clusters_k_{k}.csv')
ITEMS_FEATURE_PATH_TPL = os.path.join(DATA_PATH, '{dataset}_features_{i}.csv')
ALL_DATA_PATH_TPL = os.path.join(DATA_PATH, '{dataset}.csv')
DATA_PATH_TPL = os.path.join(DATA_PATH, '{dataset}_{type}_fold_{fold}.csv')
TRAIN_DATA_PATH_TPL = os.path.join(DATA_PATH, '{dataset}_train_fold_{fold}.csv')
TEST_DATA_PATH_TPL = os.path.join(DATA_PATH, '{dataset}_test_fold_{fold}.csv')
MODEL_PATH_TPL = os.path.join(MODEL_PATH, '{dataset}_{model}_fold_{fold}.csv')
MODULAR_MODEL_ERROR_PATH_TPL = os.path.join(MODEL_PATH, '{dataset}_{model}_error_utilities.csv')
GROUND_TRUTH_DATA_PATH_TPL = os.path.join(
    RANKING_PATH, '{dataset}_gt_fold_{fold}.csv')
PARTIAL_DATA_PATH_TPL = os.path.join(
    RANKING_PATH, '{dataset}_partial_fold_{fold}.csv')
GROUND_TRUTH_MARKOV_DATA_PATH_TPL = os.path.join(
    RANKING_PATH, '{dataset}_gt_markov_fold_{fold}.csv')
PARTIAL_DATA_MARKOV_PATH_TPL = os.path.join(
    RANKING_PATH, '{dataset}_partial_markov_fold_{fold}.csv')
RANKING_MODEL_PATH_TPL = os.path.join(
    RANKING_PATH, '{dataset}_{model}_fold_{fold}.csv')
NCE_NOISE_FACTOR = 5
NCE_DATA_PATH_TPL = os.path.join(
        DATA_PATH, '{dataset}_nce_data_features_{index}_fold_{fold}_noise_{noise_factor}.csv')
NCE_NOISE_PATH_TPL = os.path.join(
        DATA_PATH, '{dataset}_nce_noise_features_{index}_fold_{fold}.csv')
NCE_FEATURES_PATH_TPL = os.path.join(DATA_PATH, '{dataset}_nce_features_{index}.csv')
NCE_OUT_PATH_TPL = os.path.join(MODEL_PATH, '{dataset}_nce_out_features_{index}_dim_{dim}_fold_{fold}.csv')
NCE_OUT_GENERAL_PATH_OLD_TPL = os.path.join(MODEL_PATH, '{dataset}_nce_out_features_{index}_l_dim_{l_dim}_k_dim_{k_dim}_fold_{fold}.csv')
NCE_OUT_GENERAL_PATH_TPL = os.path.join(MODEL_PATH, '{dataset}_nce_out_features_{index}_l_dim_{l_dim}_k_dim_{k_dim}_fold_{fold}_iter_{iter}_eta_{eta_0}_adagrad_{adagrad}_noise_{noise}.csv')
NCE_OUT_OBJECTIVE_PATH_TPL = os.path.join(MODEL_PATH, '{dataset}_nce_objective_features_{index}_l_dim_{l_dim}_k_dim_{k_dim}_fold_{fold}_iter_{iter}_eta_{eta_0}_adagrad_{adagrad}_noise_{noise}.csv')
SEED = int(time.time())
