
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

// Parameters
// ==========
// data : the data samples separated by -1, first entry is the label 1 / 0
// data_size : the size of the above vector
// n_steps : how many SGD steps to perform
// eta_0, power, start_step : the step size used is
//
//      eta_0 / (start_step + iteration) ** power;
//
// features: The features for all items. Should be of size n_items x m_feat.
// b_weights : the diversity weights will be stored here. Will not be initialized and the
//             provided data will be used as the first iterate. Assumed to be
//             stored in *column-first* order (FORTRAN). Should be of size m_feat x l_dim.
// a_weights : the utility weights will be stored here. Should be of size m_feat.
// unaries : the unaries will be stored here. Should be of size n
//void train(const long *data, size_t data_size, long n_steps,
//           double eta_0, double power, int start_step,
//           const double *features,
//           double *b_weights, double *a_weights, double *n_logz,
//           size_t n_items, size_t l_dim, size_t m_feat) {
//#define IDX_F(i, j) ((i) * m_feat + (j))
//#define IDX_W(i, j) ((i) * l_dim + (j))
//#define IDX_B(i, j) ((i) * l_dim + (j))
//
//    std::vector<size_t> start_indices;
//    start_indices.reserve(data_size);
//    start_indices.push_back(0);
//    long n_noise = (data[0] == 0.);
//    long n_model = (data[0] == 1.);
//    for (size_t i = 1; i < data_size; i++) {
//        if (data[i] == -1.) {
//            assert(i + 1 < data_size);
//            if (data[i+1] == 1.) {
//                n_model++;
//            } else {
//                assert(data[i+1] == 0.);
//                n_noise++;
//            }
//            start_indices.push_back(i + 1);
//        }
//    }
//
//    std::vector<double> unaries_noise(n_items);
//    for (size_t i_item = 0; i_item < n_items; ++i_item) {
//         unaries_noise[i_item] = 0.;
//         for (size_t j_feat = 0; j_feat < m_feat; ++j_feat) {
//             unaries_noise[i_item] += a_weights[j_feat] * features[IDX_F(i_item, j_feat)];
//         }
//    }
//
//
//    double logz_noise = 0.;
//    for (size_t i = 0; i < n_items; i++) {
//        logz_noise += log1exp(unaries_noise[i]);
//    }
//
//    double log_nu = std::log(
//            static_cast<double>(n_noise) / static_cast<double>(n_model));
//
//    clock_t t;
//    t = clock();
//
//    std::vector<size_t> perm(start_indices.size(), 0);
//    for (size_t i = 0; i < start_indices.size(); i++) {
//        perm[i] = i;
//    }
//
//    // Initialize the utilities for all items with initial values of a.
//    std::vector<double> utilities(n_items);
//    std::vector<double> weights(n_items * l_dim);
//
//    std::srand(start_step);
//    std::vector<long> positions(l_dim, 0);
//    std::vector<double> feat_sums(m_feat);
//    for (int i_step = 0; i_step < n_steps; i_step++) {
//        // Update the utilities for all items.
//        for (size_t i_item = 0; i_item < n_items; ++i_item) {
//            utilities[i_item] = 0.;
//            for (size_t j_feat = 0; j_feat < m_feat; ++j_feat) {
//                utilities[i_item] += a_weights[j_feat] * features[IDX_F(i_item, j_feat)];
//            }
//        }
//
//        // Update the diversity weights.
//        for (size_t i_item = 0; i_item < n_items; ++i_item) {
//            for (size_t j_dim = 0; j_dim < l_dim; ++j_dim) {
//                weights[IDX_W(i_item ,j_dim)] = 0.;
//                for (size_t k_feat = 0; k_feat < m_feat; ++k_feat) {
//                    weights[IDX_W(i_item, j_dim)] += features[IDX_F(i_item, k_feat)] * b_weights[IDX_B(k_feat, j_dim)];
//                }
//            }
//        }
//
//
//        if (i_step % start_indices.size() == 0) {
//            // permute
//            std::random_shuffle(perm.begin(), perm.end());
//        }
//
//        size_t idx = perm[i_step % start_indices.size()];
//
//        double step = eta_0 * pow(start_step + i_step + 1, -power);
//        assert (idx < start_indices.size());
//
//        size_t start_idx = start_indices[idx];
//        size_t end_idx;
//        if (idx + 1 == start_indices.size()) {
//            end_idx = data_size;
//        } else {
//            end_idx = start_indices[idx + 1] - 1;
//        }
//
//        // we cannot handle empty sets yet
//        assert (start_idx < end_idx);
//
//        double f_model = *n_logz;
//        double f_noise = -logz_noise;
//
//        for (size_t i_item = start_idx + 1; i_item < end_idx; ++i_item) {
//            f_model += utilities[data[i_item]];
//            f_noise += unaries_noise[data[i_item]];
//        }
//
//
//        for (size_t j = 0; j < l_dim; j++) {
//            double max = -1;
//            long max_idx = n_items;
//            for (size_t k = start_idx + 1; k < end_idx; k++) {
//                assert(data[k] >= 0);
//                assert(IDX_W(data[k], j) >= 0);
//                assert(IDX_W(data[k], j) < n_items * l_dim);
//                f_model -= weights[IDX_W(data[k], j)];
//
//                if (weights[IDX_W(data[k], j)] > max) {
//                    max = weights[IDX_W(data[k], j)];
//                    max_idx = IDX_W(data[k], j);
//                }
//            }
//            assert(max_idx != n_items * l_dim);
//
//            f_model += max;
//
//            positions[j] = max_idx;
//        }
//
//        // We can now take the gradient step.
//        double label = data[start_idx];
//        double factor = step * (label - expit(f_model - f_noise - log_nu));
//
//        // Calculate the sum of features over dimensions for items in set.
//        // Update the a weights.
//        for (size_t i_feat = 0; i_feat < m_feat; ++i_feat) {
//            feat_sums[i_feat] = 0.;
//            for (size_t j_item = start_idx + 1; j_item < end_idx; ++j_item) {
//                assert(data[j_item] >= 0);
//                assert(data[j_item] < n_items);
//                feat_sums[i_feat] += features[IDX_F(data[j_item], i_feat)];
//            }
//            a_weights[i_feat] += factor * feat_sums[i_feat];
//        }
//
//        for (size_t j_dim = 0; j_dim < l_dim; ++j_dim) {
//            b_weights[positions[j_dim]] += factor;
//            for (size_t k = start_idx + 1; k < end_idx; ++k) {
//                b_weights[IDX_B(k, j_dim)] -= factor;
//                if (b_weights[IDX_B(k, j_dim)] <= 0) {
//                    b_weights[IDX_B(k, j_dim)] = 1e-3 * (
//                            static_cast<double>(std::rand()) / RAND_MAX);
//                }
//            }
//        }
//
//        *n_logz += factor;
//    }
//
//    t = clock() - t;
//    std::cout << "It took me "  << (((float)t)/CLOCKS_PER_SEC) << "seconds" << std::endl;
//}

int main() {
    size_t dim = 2;
    for (int i = 1; i <= 10; ++i) {
        train_with_features(
                (boost::format("/home/diegob/workspace/master-thesis-2015/data/path_set_nce_data_fold_%1%.csv") % i).str(),
                (boost::format("/home/diegob/workspace/master-thesis-2015/data/path_set_nce_features_fold_%1%.csv") % i).str(),
                (boost::format("/home/diegob/workspace/master-thesis-2015/data/path_set_nce_noise_fold_%1%.csv") % i).str(),
                10, 0.01, 0.1, dim,
                (boost::format("/home/diegob/workspace/master-thesis-2015/data/models/path_set_nce_out_dim_%2%_fold_%1%.csv") % i % dim).str());
    }

}
