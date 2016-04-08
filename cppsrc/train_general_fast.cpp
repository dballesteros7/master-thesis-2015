#include "utils.h"

#include <fstream>
#include <iostream>
#include <ctime>
#include <thread>

#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>

#include <Eigen/Dense>

using namespace Eigen;

void train_with_features(std::string data_file_path,
                         std::string features_file_path,
                         std::string noise_file_path,
                         int n_steps, double eta_0, double iter_power,
                         size_t l_dimensions, size_t k_dimensions,
                         std::string output_file_path,
                         std::string objective_output_file_path,
                         double *elapsed_time) {
    auto random_engine = std::mt19937(std::time(0));
    time_t start = std::time(0);
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
    Matrix<double, Dynamic, Dynamic, RowMajor> features;
    masterthesis::readFeatureFileBoost(
            features_file_path, n_items, m_features, features);

    std::fstream noise_file_input;
    noise_file_input.open(noise_file_path, std::ios::in);
    getline(noise_file_input, line);
    boost::split(tokens, line, boost::is_any_of(","));
    VectorXd a_weights(m_features);
    for (size_t i = 0; i < m_features; ++i) {
        a_weights(i) = stod(tokens[i]);
    }
    getline(noise_file_input, line);
    boost::split(tokens, line, boost::is_any_of(","));
    RowVectorXd noise_utilities(n_items);
    for (size_t i = 0; i < n_items; ++i) {
        noise_utilities(i) = stod(tokens[i]);
    }


    // Calculate constant quantities.
    double nu = ((float)n_noise) / ((float)n_data);
    double log_nu = log(nu);
    double logz_noise = 0;
    for (size_t i = 0; i < n_items; ++i) {
        logz_noise += masterthesis::log1exp(noise_utilities[i]);
    }

    // Initialize parameters.
    std::uniform_real_distribution<double_t> udouble_dist(0, 1e-3);
    MatrixXd b_weights(m_features, l_dimensions);
    MatrixXd c_weights(m_features, k_dimensions);
    double n_logz = 0;

    for (size_t i = 0; i < m_features; ++i) {
        for (size_t j = 0; j < l_dimensions; ++j) {
            b_weights(i, j) = udouble_dist(random_engine);
        }
        for (size_t j = 0; j < k_dimensions; ++j) {
            c_weights(i, j) = udouble_dist(random_engine);
        }
    }
    VectorXd item_utilities = features*a_weights;
    for (size_t i = 0; i < n_items; ++i) {
        n_logz -= masterthesis::log1exp(item_utilities[i]);
    }

    std::vector<double> objectives(n_steps);
    VectorXd a_gradient(m_features);
    MatrixXd b_weights_gradient(m_features, l_dimensions);
    MatrixXd c_weights_gradient(m_features, k_dimensions);

#ifdef ADAGRAD
    // for adagrad
    VectorXd g_a_weights = VectorXd::Constant(m_features, 1e-2);
    MatrixXd g_b_weights = MatrixXd::Constant(m_features, l_dimensions, 1e-2);
    MatrixXd g_c_weights = MatrixXd::Constant(m_features, k_dimensions, 1e-2);
    double g_n_logz = 1e-2;
#endif
    // initialize all entries of g_a_weights, g_b_weights, g_c_weights to 1e-2
    for (size_t iter = 0; iter < n_steps; ++iter) {
#ifdef COMPUTE_OBJ
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

            size_t set_size = end_idx - start_idx - 1;
            if (set_size > 0) {
                Matrix<double, Dynamic, Dynamic, RowMajor> sub_feature_matrix(set_size, m_features);
                for (size_t i = start_idx + 1; i < end_idx; ++i) {
                    RowVectorXd feature_row = features.row(data[i]);
                    sub_feature_matrix.row(i - start_idx - 1) = feature_row;
                    p_model += feature_row.dot(a_weights);
                    p_noise += noise_utilities[data[i]];
                }

                MatrixXd div_weights_slice(set_size, l_dimensions);
                MatrixXd coh_weights_slice(set_size, k_dimensions);
                div_weights_slice.noalias() = sub_feature_matrix * b_weights;
                coh_weights_slice.noalias() = sub_feature_matrix * c_weights;
                p_model -= div_weights_slice.sum();
                p_model += div_weights_slice.colwise().maxCoeff().sum();
                p_model += coh_weights_slice.sum();
                p_model -= coh_weights_slice.colwise().maxCoeff().sum();
            }
            if (label == 1) {
                objective -= masterthesis::log1exp(log_nu + p_noise - p_model);
            } else {
                objective -= masterthesis::log1exp(p_model - log_nu - p_noise);
            }
        }
#ifdef PRINT_DEBUG
        std::cout << objective << std::endl;
#endif
        objectives[iter] = objective;
#endif
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

            size_t set_size = end_idx - start_idx - 1;
            a_gradient.setZero();
            if (set_size > 0) {
                Matrix<double, Dynamic, Dynamic, RowMajor> sub_feature_matrix(set_size, m_features);
                MatrixXd div_weights_slice(set_size, l_dimensions);
                MatrixXd coh_weights_slice(set_size, k_dimensions);
                for (size_t i = start_idx + 1; i < end_idx; ++i) {
                    RowVectorXd feature_row = features.row(data[i]);
                    sub_feature_matrix.row(i - start_idx - 1) = feature_row;
                    a_gradient += feature_row.transpose();
                    p_model += feature_row.dot(a_weights);
                    p_noise += noise_utilities[data[i]];
                }
                div_weights_slice.noalias() = sub_feature_matrix * b_weights;
                coh_weights_slice.noalias() = sub_feature_matrix * c_weights;
                p_model -= div_weights_slice.sum();
                p_model += coh_weights_slice.sum();

                for (size_t i = 0; i < l_dimensions; ++i) {
                    int index;
                    p_model += div_weights_slice.col(i).maxCoeff(&index);
                    b_weights_gradient.col(i) = features.row(data[start_idx + 1 + index]).transpose();
                }
                for (size_t i = 0; i < k_dimensions; ++i) {
                    int index;
                    p_model -= coh_weights_slice.col(i).maxCoeff(&index);
                    c_weights_gradient.col(i) = features.row(data[start_idx + 1 + index]).transpose();
                }
            }
#ifdef ADAGRAD
            double learning_rate = eta_0;
            double tfactor = (label - masterthesis::expit(p_model - p_noise - log_nu));
            double tfactor_sq = tfactor * tfactor;
            double step = learning_rate * tfactor;
#else
            double learning_rate = eta_0 * pow((iter * n_samples) + sub_iter + 1, -iter_power);
            double step = learning_rate * (label - masterthesis::expit(p_model - p_noise - log_nu));
#endif
            if (set_size > 0) {
                b_weights_gradient.colwise() -= a_gradient;
                c_weights_gradient.colwise() -= a_gradient;
                c_weights_gradient *= -1;
#ifdef ADAGRAD
                g_a_weights += tfactor_sq*(a_gradient.cwiseProduct(a_gradient));
                g_b_weights += tfactor_sq*(b_weights_gradient.cwiseProduct(b_weights_gradient));
                g_c_weights += tfactor_sq*(c_weights_gradient.cwiseProduct(c_weights_gradient));
                // element-wise division
                a_weights += step*(a_gradient.array() / g_a_weights.cwiseSqrt().array()).matrix();
                b_weights += step*(b_weights_gradient.array() / g_b_weights.cwiseSqrt().array()).matrix();
                c_weights += step*(c_weights_gradient.array() / g_c_weights.cwiseSqrt().array()).matrix();
#else
                a_weights += step*a_gradient;
                b_weights += step*b_weights_gradient;
                c_weights += step*c_weights_gradient;
#endif
                for (size_t i = 0; i < m_features; ++i) {
                    for (size_t j = 0; j < l_dimensions; ++j) {
                        if (b_weights(i, j) < 0) {
                            b_weights(i, j) = udouble_dist(random_engine);
                        }
                    }
                    for (size_t j = 0; j < k_dimensions; ++j) {
                        if (c_weights(i, j) < 0) {
                            c_weights(i, j) = udouble_dist(random_engine);
                        }
                    }
                }
            }
#ifdef ADAGRAD
            g_n_logz += tfactor_sq;
            n_logz += step / std::sqrt(g_n_logz);
#else
            n_logz += step;
#endif
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
                output_file << b_weights(i, j) << ",";
            }
            output_file << b_weights(i, l_dimensions - 1) << std::endl;
        }
    }

    for (size_t i = 0; i < m_features; ++i) {
        if (k_dimensions > 0) {
            for (size_t j = 0; j < k_dimensions - 1; ++j) {
                output_file << c_weights(i, j) << ",";
            }
            output_file << c_weights(i, k_dimensions - 1) << std::endl;
        }
    }

    output_file.close();
#ifdef COMPUTE_OBJ
    std::fstream objective_output_file(
            objective_output_file_path, std::ios::out);

    for (size_t i = 0; i < n_steps; ++i) {
        objective_output_file << objectives[i] << std::endl;
    }
    std::cout << "Fold final objective: " << objectives[n_steps - 1] << std::endl;
    objective_output_file.close();
#endif
    time_t end = std::time(0);
    double elapsed = std::difftime(end, start);
    *elapsed_time = elapsed;
#ifdef PRINT_DEBUG
    std::cout << "Fold finished, took: " << elapsed << "s." << std::endl;
#endif
}

static const int total_threads = 4;

int main(int argc, char* argv[]) {
    int fold_number = std::stoi(argv[1]);
    int l_dimensions = std::stoi(argv[2]);
    int k_dimensions = std::stoi(argv[3]);
    char* feature_set = argv[4];
    char* dataset_name = argv[5];
    double noise_factor = std::stod(argv[6]);
    int iterations = std::stoi(argv[7]);
    double eta_0 = std::stod(argv[8]);
    std::vector<double> times(fold_number);
    std::thread thread_pool[total_threads];
    int used_threads = 0;
#ifdef ADAGRAD
    short uses_adagrad = 1;
#else
    short uses_adagrad = 0;
#endif
    time_t start = std::time(0);
    for (int i = 1; i <= fold_number; ++i) {
        if (used_threads == total_threads) {
            for(size_t j = 0; j < used_threads; ++j) {
                thread_pool[j].join();
            }
            used_threads = 0;
        }
        thread_pool[used_threads] = std::thread(train_with_features,
            (boost::format(
                    "/home/diegob/workspace/master-thesis-2015/data/path_set_%1%_nce_data_features_%2%_fold_%3%_noise_%4%.csv") %
             dataset_name % feature_set % i % noise_factor).str(),
            (boost::format(
                    "/home/diegob/workspace/master-thesis-2015/data/path_set_%1%_nce_features_%2%.csv") %
             dataset_name % feature_set).str(),
            (boost::format(
                    "/home/diegob/workspace/master-thesis-2015/data/path_set_%1%_nce_noise_features_%2%_fold_%3%.csv") %
             dataset_name % feature_set % i).str(),
            iterations, eta_0, 0.1,
            l_dimensions,
            k_dimensions,
            (boost::format(
                    "/home/diegob/workspace/master-thesis-2015/data/models/path_set_%1%_nce_out_features_%2%_l_dim_%3%_k_dim_%4%_fold_%5%_iter_%6%_eta_%7%_adagrad_%8%_noise_%9%.csv") %
             dataset_name % feature_set % l_dimensions % k_dimensions % i % iterations % eta_0 % uses_adagrad % noise_factor).str(),
            (boost::format(
                    "/home/diegob/workspace/master-thesis-2015/data/models/path_set_%1%_nce_objective_features_%2%_l_dim_%3%_k_dim_%4%_fold_%5%_iter_%6%_eta_%7%_adagrad_%8%_noise_%9%.csv") %
             dataset_name % feature_set % l_dimensions % k_dimensions % i % iterations % eta_0 % uses_adagrad % noise_factor).str(),
            &times[i-1]
        );
        ++used_threads;
    }
    for(size_t j = 0; j < used_threads; ++j) {
        thread_pool[j].join();
    }
    time_t end = std::time(0);
    double total_time = std::difftime(end, start);
    std::fstream times_output_file(
            (boost::format(
                    "/home/diegob/workspace/master-thesis-2015/data/models/path_set_%1%_nce_timing_features_%2%_l_dim_%3%_k_dim_%4%_iter_%5%_eta_%6%_adagrad_%7%_noise_%8%.csv") %
             dataset_name % feature_set % l_dimensions % k_dimensions % iterations % eta_0 % uses_adagrad % noise_factor).str(),
            std::ios::out);
    times_output_file << total_time << std::endl;
    for (size_t i = 0; i < fold_number; ++i) {
        times_output_file << times[i] << std::endl;
    }
    times_output_file.close();
}
