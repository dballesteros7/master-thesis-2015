import scipy.io

def load_amazon_names(filename):
    with codecs.open(filename, 'rb') as f:
        return [name.decode('unicode_escape') for name in f]


def load_amazon_data(filename):
    with open(filename, 'r') as f:
        return [[int(x) - 1 for x in line.strip().split(',')]
                for line in f if line.strip()]

def load_amazon_ranking_data(filename):
    """
    Only difference to amazon_data is that item indices must not be corrected.
    """
    with open(filename, 'r') as f:
        return [[int(x) for x in line.strip().split(',')]
                for line in f if line.strip()]


def load_dpp_result(dataset, fold, RESULT_PATH):
    model_f = '{0}/{1}_fold_{2}.mat'.format(
            RESULT_PATH, dataset, fold + 1)
    print("Loading matlab model from %s." % (model_f))
    mat = scipy.io.loadmat(model_f)
    ll_em = mat['ll_test_em'][0][0]
    rt_em = mat['rt_em'][0][0]
    ll_picard = mat['ll_test_picard'][0][0]
    rt_picard = mat['rt_picard'][0][0]


    return {'em': (-ll_em, rt_em), 'picard': (-ll_picard, rt_picard)}

