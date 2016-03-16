#include "utils.h"

#include <fstream>
#include <iostream>
#include <ctime>


#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>

void train_with_features(std::string data_file_path,
                         std::string features_file_path,
                         std::string noise_file_path,
                         int n_steps, double eta_0, double iter_power,
                         size_t l_dimensions, size_t k_dimensions,
                         std::string output_file_path,
                         std::string objective_output_file_path) {
    // Data loading
    std::fstream data_file_input;
    data_file_input.open(data_file_path, std::ios::in);
    std::string line;
    std::vector<std::string> tokens;
    getline(data_file_input, line);
    boost::split(tokens, line, boost::is_any_of(","));
    size_t data_size = stoul(tokens[0]);
    size_t n_samples = stoul(tokens[1]);

    std::vector<size_t> start_indexes(n_samples);
    std::vector<size_t> permutation(n_samples);
    std::vector<int> data(data_size + n_samples);
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

    size_t n_items;
    size_t m_features;
    std::vector<double> features;
    auto index_features = [&m_features](size_t i, size_t j) -> size_t {
        return (i * m_features) + j;
    };
    masterthesis::readFeatureFile(
            features_file_path, &n_items, &m_features, &features);

    std::fstream noise_file_input;
    noise_file_input.open(noise_file_path, std::ios::in);
    getline(noise_file_input, line);
    boost::split(tokens, line, boost::is_any_of(","));
    std::vector<double> noise_weights(m_features);
    for (size_t i = 0; i < m_features; ++i) {
        noise_weights[i] = stod(tokens[i]);
    }
    std::vector<double> noise_utilities(n_items);
    for (size_t i = 0; i < n_items; ++i) {
        noise_utilities[i] = 0;
        for (size_t j = 0; j < m_features; ++j) {
            noise_utilities[i] +=
                    features[index_features(i, j)] * noise_weights[j];
        }
    }


    // Calculate constant quantities.
    double nu = n_noise / n_data;
    double log_nu = log(nu);
    double logz_noise = 0;
    for (size_t i = 0; i < n_items; ++i) {
        logz_noise += masterthesis::log1exp(noise_utilities[i]);
    }

    // Initialize parameters.
    auto random_engine = std::default_random_engine{};
    random_engine.seed(std::time(NULL));
    std::uniform_real_distribution<double_t> udouble_dist(0, 1);
    std::vector<double> b_weights(m_features * l_dimensions);
    std::vector<double> c_weights(m_features * k_dimensions);
    std::vector<double> a_weights(m_features);
    double n_logz = 0;

    auto index_b_weights = [&l_dimensions](size_t i, size_t j) -> size_t {
        return (i * l_dimensions) + j;
    };

    auto index_c_weights = [&k_dimensions](size_t i, size_t j) -> size_t {
        return (i * k_dimensions) + j;
    };

    for (size_t i = 0; i < m_features; ++i) {
        a_weights[i] = noise_weights[i];
    }
    for (size_t i = 0; i < m_features; ++i) {
        for (size_t j = 0; j < l_dimensions; ++j) {
            b_weights[index_b_weights(i, j)] = udouble_dist(random_engine);
        }
        for (size_t j = 0; j < k_dimensions; ++j) {
            c_weights[index_c_weights(i, j)] = udouble_dist(random_engine);
        }
    }
    for (size_t i = 0; i < n_items; ++i) {
        n_logz -= masterthesis::log1exp(noise_utilities[i]);
    }

    std::vector<double> a_gradient(m_features);
    std::vector<double> objectives(n_steps);
    for (size_t iter = 0; iter < n_steps; ++iter) {
        double objective = 0;
        for (size_t sub_iter = 0; sub_iter < n_samples; ++sub_iter) {
            size_t start_idx = start_indexes[sub_iter];
            size_t end_idx;
            if (sub_iter == n_samples - 1) {
                end_idx = data_size + n_samples - 1;
            } else {
                end_idx = start_indexes[sub_iter + 1] - 1;
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
            for (size_t j = 0; j < l_dimensions; ++j) {
                double max_b_weight = -1;
                for (size_t i = start_idx + 1; i < end_idx; ++i) {
                    double weight = 0;
                    for (size_t k = 0; k < m_features; ++k) {
                        weight += features[index_features(data[i], k)] *
                                  b_weights[index_b_weights(k, j)];
                    }
                    if (weight > max_b_weight) {
                        max_b_weight = weight;
                    }
                    p_model -= weight;
                }
                if (max_b_weight >= 0) {
                    p_model += max_b_weight;
                }
            }
            for (size_t j = 0; j < k_dimensions; ++j) {
                double max_c_weight = -1;
                for (size_t i = start_idx + 1; i < end_idx; ++i) {
                    double weight = 0;
                    for (size_t k = 0; k < m_features; ++k) {
                        weight += features[index_features(data[i], k)] *
                                  c_weights[index_c_weights(k, j)];
                    }
                    if (weight > max_c_weight) {
                        max_c_weight = weight;
                    }
                    p_model += weight;
                }
                if (max_c_weight >= 0) {
                    p_model -= max_c_weight;
                }
            }
            if (label == 1) {
                objective -= masterthesis::log1exp(log_nu + p_noise - p_model);
            } else {
                objective -= masterthesis::log1exp(p_model - log_nu - p_noise);
            }
        }
        objectives[iter] = objective;
        shuffle(begin(permutation), end(permutation), random_engine);
        for (size_t sub_iter = 0; sub_iter < n_samples; ++sub_iter) {
            size_t start_idx = start_indexes[permutation[sub_iter]];
            size_t end_idx;
            if (permutation[sub_iter] == n_samples - 1) {
                end_idx = data_size + n_samples - 1;
            } else {
                end_idx = start_indexes[permutation[sub_iter] + 1] - 1;
            }
            bool isEmpty = (start_idx == end_idx - 1);
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

            std::vector<double> max_b_weights(l_dimensions);
            std::vector<int> max_weight_b_indexes(l_dimensions);
            std::vector<double> max_c_weights(k_dimensions);
            std::vector<int> max_weight_c_indexes(k_dimensions);
            if (!isEmpty) {
                for (size_t j = 0; j < l_dimensions; ++j) {
                    max_b_weights[j] = -1;
                    max_weight_b_indexes[j] = -1;
                    for (size_t i = start_idx + 1; i < end_idx; ++i) {
                        double weight = 0;
                        for (size_t k = 0; k < m_features; ++k) {
                            weight += features[index_features(data[i], k)] *
                                      b_weights[index_b_weights(k, j)];
                        }
                        if (weight > max_b_weights[j]) {
                            max_b_weights[j] = weight;
                            max_weight_b_indexes[j] = data[i];
                        }
                        p_model -= weight;
                    }
                    p_model += max_b_weights[j];
                }

                for (size_t j = 0; j < k_dimensions; ++j) {
                    max_c_weights[j] = -1;
                    max_weight_c_indexes[j] = -1;
                    for (size_t i = start_idx + 1; i < end_idx; ++i) {
                        double weight = 0;
                        for (size_t k = 0; k < m_features; ++k) {
                            weight += features[index_features(data[i], k)] *
                                      c_weights[index_c_weights(k, j)];
                        }
                        if (weight > max_c_weights[j]) {
                            max_c_weights[j] = weight;
                            max_weight_c_indexes[j] = data[i];
                        }
                        p_model += weight;
                    }
                        p_model -= max_c_weights[j];
                }
            }

            double learning_rate = eta_0 *
                                   pow((iter * n_samples) + sub_iter + 1, -iter_power);
            double factor = learning_rate *
                            (label - masterthesis::expit(p_model - p_noise - log_nu));

            if (!isEmpty) {
                for (size_t i = 0; i < m_features; ++i) {
                    a_weights[i] += factor * a_gradient[i];
                    for (size_t j = 0; j < l_dimensions; ++j) {
                        b_weights[index_b_weights(i, j)] += factor *
                                                            (features[index_features(
                                                                    max_weight_b_indexes[j],
                                                                    i)] -
                                                             a_gradient[i]);
                        if (b_weights[index_b_weights(i, j)] <= 0) {
                            b_weights[index_b_weights(i, j)] = udouble_dist(
                                    random_engine);
                        }
                    }
                    for (size_t j = 0; j < k_dimensions; ++j) {
                        c_weights[index_c_weights(i, j)] += factor *
                                                            (a_gradient[i] -
                                                             features[index_features(
                                                                     max_weight_c_indexes[j],
                                                                     i)]);
                        if (c_weights[index_c_weights(i, j)] <= 0) {
                            c_weights[index_c_weights(i, j)] = udouble_dist(
                                    random_engine);
                        }
                    }
                }
            }
            n_logz += factor;
        }
    }

    std::fstream output_file(output_file_path, std::ios::out);

    output_file << n_logz << std::endl;
    for (size_t i = 0; i < m_features - 1; ++i) {
        output_file << a_weights[i] << ",";
    }
    output_file << a_weights[m_features - 1] << std::endl;

    for (size_t i = 0; i < m_features; ++i) {
        if (l_dimensions > 0) {
            for (size_t j = 0; j < l_dimensions - 1; ++j) {
                output_file << b_weights[index_b_weights(i, j)] << ",";
            }
            output_file << b_weights[index_b_weights(i, l_dimensions - 1)] << std::endl;
        }
    }

    for (size_t i = 0; i < m_features; ++i) {
        if (k_dimensions > 0) {
            for (size_t j = 0; j < k_dimensions - 1; ++j) {
                output_file << c_weights[index_c_weights(i, j)] << ",";
            }
            output_file << c_weights[index_c_weights(i, k_dimensions - 1)] << std::endl;
        }
    }

    output_file.close();

    std::fstream objective_output_file(
            objective_output_file_path, std::ios::out);

    for (size_t i = 0; i < n_steps; ++i) {
        objective_output_file << objectives[i] << std::endl;
    }

    objective_output_file.close();
}

int main(int argc, char* argv[]) {
    int fold_number = std::stoi(argv[1]);
    int l_dimensions = std::stoi(argv[2]);
    int k_dimensions = std::stoi(argv[3]);
    char* feature_set = argv[4];
    char* dataset_name = argv[5];
    int iterations = std::stoi(argv[6]);
    for (int i = 1; i <= fold_number; ++i) {
        train_with_features(
                (boost::format(
                        "/home/diegob/workspace/master-thesis-2015/data/path_set_%1%_nce_data_features_%2%_fold_%3%.csv") %
                 dataset_name % feature_set % i).str(),
                (boost::format(
                        "/home/diegob/workspace/master-thesis-2015/data/path_set_%1%_nce_features_%2%.csv") %
                 dataset_name % feature_set).str(),
                (boost::format(
                        "/home/diegob/workspace/master-thesis-2015/data/path_set_%1%_nce_noise_features_%2%_fold_%3%.csv") %
                 dataset_name % feature_set % i).str(),
                iterations, 0.01, 0.1,
                static_cast<size_t>(l_dimensions),
                static_cast<size_t>(k_dimensions),
                (boost::format(
                        "/home/diegob/workspace/master-thesis-2015/data/models/path_set_%1%_nce_out_features_%2%_l_dim_%3%_k_dim_%4%_fold_%5%.csv") %
                 dataset_name % feature_set % l_dimensions % k_dimensions % i).str(),
                (boost::format(
                        "/home/diegob/workspace/master-thesis-2015/data/models/path_set_%1%_nce_objective_features_%2%_l_dim_%3%_k_dim_%4%_fold_%5%.csv") %
                 dataset_name % feature_set % l_dimensions % k_dimensions % i).str()
        );
        std::cout << "Fold finished." << std::endl;
    }
}
