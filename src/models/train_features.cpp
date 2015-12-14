
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <array>

#include "train_features.h"

using namespace std;

double expit(double x) {
    if (x > 0) {
        return 1. / (1. + exp(-x));
    } else {
        return 1. - 1. / (1. + exp(x));
    }
}

double log1exp(double x) {
    if (x > 0) {
        return x + log1p(exp(-x));
    } else {
        return log1p(exp(x));
    }
}

void train_with_features(string data_file_path,
                         string features_file_path,
                         string noise_file_path,
                         int n_steps, double eta_0, double iter_power,
                         size_t l_dimensions,
                         string output_file_path) {
    cout << data_file_path << endl;
    cout << features_file_path << endl;
    cout << noise_file_path << endl;

    // Data loading
    fstream data_file_input;
    data_file_input.open(data_file_path, ios::in);
    string line;
    vector<string> tokens;
    getline(data_file_input, line);
    boost::split(tokens, line, boost::is_any_of(","));
    size_t data_size = stoul(tokens[0]);
    size_t n_samples = stoul(tokens[1]);

    vector<size_t> start_indexes(n_samples);
    vector<size_t> permutation(n_samples);
    vector<int> data(data_size + n_samples);
    long n_data = 0;
    long n_noise = 0;
    for (size_t i = 0, data_index = 0; i < n_samples; ++i) {
        getline(data_file_input, line);
        boost::split(tokens, line, boost::is_any_of(","));
        start_indexes[i] = data_index;
        permutation[i] = i;
        for (size_t j = 0; j < tokens.size(); ++j) {
            data[data_index + j] = stoi(tokens[j]);
        }
        if (data[data_index] == 0) {
            ++n_noise;
        } else {
            ++n_data;
        }
        data[data_index + tokens.size()] = -1;
        data_index += tokens.size() + 1;
    }

    fstream features_file_input;
    features_file_input.open(features_file_path, ios::in);
    getline(features_file_input, line);
    boost::split(tokens, line, boost::is_any_of(","));
    size_t n_items = stoul(tokens[0]);
    size_t m_features = stoul(tokens[1]);

    auto index_features = [m_features](size_t i, size_t j) -> size_t {
        return (i * m_features) + j;
    };
    vector<double> features(n_items * m_features);
    for (size_t i = 0; i < n_items; ++i) {
        getline(features_file_input, line);
        boost::split(tokens, line, boost::is_any_of(","));
        for (size_t j = 0; j < m_features; ++j) {
            features[index_features(i, j)] = stod(tokens[j]);
        }
    }

    fstream noise_file_input;
    noise_file_input.open(noise_file_path, ios::in);
    getline(noise_file_input, line);
    boost::split(tokens, line, boost::is_any_of(","));
    vector<double> noise_weights(m_features);
    for (size_t i = 0; i < m_features; ++i) {
        noise_weights[i] = stod(tokens[i]);
    }
    vector<double> noise_utilities(n_items);
    for (size_t i = 0; i < n_items; ++i) {
        noise_utilities[i] = 0;
        for (size_t j = 0; j < m_features; ++j) {
            noise_utilities[i] += features[index_features(i, j)] * noise_weights[j];
        }
    }


    // Calculate constant quantities.
    double log_nu = log(n_noise / n_data);
    double logz_noise = 0;
    for (size_t i = 0; i < n_items; ++i) {
        logz_noise += log1exp(noise_utilities[i]);
    }
    cout << "Log nu: " << log_nu << endl;
    cout << "Log noise: " << logz_noise << endl;

    // Initialize parameters.
    auto random_engine = default_random_engine{};
    random_engine.seed(10000000);
    uniform_real_distribution<double_t> udouble_dist(0, 1e-3);
    vector<double> b_weights(m_features * l_dimensions);
    vector<double> a_weights(m_features);
    double n_logz = 0;

    auto index_b_weights = [l_dimensions](size_t i, size_t j) -> size_t {
        return (i * l_dimensions) + j;
    };

    for (size_t i = 0; i < m_features; ++i) {
        a_weights[i] = noise_weights[i];
    }
    for (size_t i = 0; i < m_features; ++i) {
        for (size_t j = 0; j < l_dimensions; ++j) {
            b_weights[index_b_weights(i, j)] = udouble_dist(random_engine);
        }
    }
    for (size_t i = 0; i < n_items; ++i) {
        n_logz -= log1exp(noise_utilities[i]);
    }

    vector<double> a_gradient(m_features);
    for (size_t iter = 0; iter < n_steps; ++iter) {
        shuffle(begin(permutation), end(permutation), random_engine);
        for (size_t sub_iter = 0; sub_iter < n_samples; ++sub_iter) {
            size_t start_idx = start_indexes[permutation[sub_iter]];
            size_t end_idx;
            if (permutation[sub_iter] == n_samples - 1) {
                end_idx = data_size + n_samples - 1;
            } else {
                end_idx = start_indexes[permutation[sub_iter] + 1] - 1;
            }
            int label = data[start_idx];
            double p_model = n_logz;
            double p_noise = -logz_noise;
            for (size_t j = 0; j < m_features; ++j) {
                a_gradient[j] = 0;
                for (size_t i = start_idx + 1; i < end_idx; ++i) {
                    a_gradient[j] += features[index_features(data[i], j)];
                    p_model += features[index_features(data[i], j)] *
                            a_weights[j];
                    p_noise += features[index_features(data[i], j)] *
                            noise_weights[j];
                }
            }

            vector<double> max_weights(l_dimensions);
            vector<int> max_weight_indexes(l_dimensions);
            for (size_t j = 0; j < l_dimensions; ++j) {
                max_weights[j] = -1;
                max_weight_indexes[j] = -1;
                for (size_t i = start_idx + 1; i < end_idx; ++i) {
                    double weight = 0;
                    for (size_t k = 0; k < m_features; ++k) {
                        weight += features[index_features(data[i], k)] *
                                b_weights[index_b_weights(k, j)];
                    }
                    if (weight > max_weights[j]) {
                        max_weights[j] = weight;
                        max_weight_indexes[j] = data[i];
                    }
                    p_model -= weight;
                }
                p_model += max_weights[j];
            }

            double learning_rate = eta_0 *
                    pow((iter * n_samples) + sub_iter + 1, -iter_power);
            double factor = learning_rate *
                    (label - expit(p_model - p_noise - log_nu));

            for (size_t i = 0; i < m_features; ++i) {
                a_weights[i] += factor * a_gradient[i];
                for (size_t j = 0; j < l_dimensions; ++j) {
                    b_weights[index_b_weights(i, j)] += factor *
                        (features[index_features(max_weight_indexes[j], i)] -
                                a_gradient[i]);
                    if (b_weights[index_b_weights(i, j)] <= 0) {
                        b_weights[index_b_weights(i, j)] =
                                udouble_dist(random_engine);
                    }
                }
            }
            n_logz += factor;
        }
    }

    fstream output_file(output_file_path, ios::out);

    cout << n_logz << endl;
    output_file << n_logz << endl;

    for (size_t i = 0; i < m_features - 1; ++i) {
        cout << a_weights[i] << ",";
        output_file << a_weights[i] << ",";
    }
    cout << a_weights[m_features - 1] << endl;
    output_file << a_weights[m_features - 1] << endl;

    for (size_t i = 0; i < m_features; ++i) {
        for (size_t j = 0; j < l_dimensions - 1; ++j) {
            cout << b_weights[index_b_weights(i, j)] << ",";
            output_file << b_weights[index_b_weights(i, j)] << ",";
        }
        output_file << b_weights[index_b_weights(i, l_dimensions - 1)] << endl;
        cout << endl;
    }

}

int main(int argc, char* argv[]) {
    int fold_number = stoi(argv[1]);
    int dim_number = stoi(argv[2]);
    int feature_set = stoi(argv[3]);
    for (int d = 1; d <= dim_number; ++d) {
        for (int i = 1; i <= fold_number; ++i) {
            train_with_features(
                    (boost::format("/home/diegob/workspace/master-thesis-2015/data/path_set_nce_data_features_%1%_fold_%2%.csv") % feature_set % i).str(),
                    (boost::format("/home/diegob/workspace/master-thesis-2015/data/path_set_nce_features_%1%.csv") % feature_set).str(),
                    (boost::format("/home/diegob/workspace/master-thesis-2015/data/path_set_nce_noise_features_%1%_fold_%2%.csv") % feature_set % i).str(),
                    10, 0.01, 0.1, d,
                    (boost::format("/home/diegob/workspace/master-thesis-2015/data/models/path_set_nce_out_features_%1%_dim_%2%_fold_%3%.csv") % feature_set % d % i).str());
        }
    }

}
