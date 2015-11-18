#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <vector>
#include <iostream>
#include <time.h> 
#include <algorithm>
#include <random>

#include "train.h"

// #define use_lower_bound
// #define use_l2_reg
// #define use_structured_reg
// #define use_l1_reg
// #define max_only
// #define use_lp
// #define p 2.0

double expit(double x) {
    if (x > 0) {
        return 1. / (1. + exp(-x));
    } else {
        return 1. - 1. / (1. + exp(x));
    }
}

double log1exp(double x) {
    if (x > 0) {
        return x + log1p(std::exp(-x));
    } else {
        return log1p(std::exp(x));
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
// weights : the weights will be stored here. Will not be initialized and the
//           provided data will be used as the first iterate. Assumed to be
//           stored in *column-first* order (FORTRAN). Should be of size n x m.
// unaries : the unaries will be stored here. Should be of size n
void train(const long *data, size_t data_size, long n_steps,
           double eta_0, double power, int start_step,
           const double *unaries_noise,
           double *weights, double *unaries, double *n_logz,
           size_t n, size_t m) {
#define IDX(w, v) ((w) * m + (v))
    std::vector<long> start_indices;
    start_indices.reserve(data_size);
    start_indices.push_back(0);
    long n_noise = (data[0] == 0.);
    long n_model = (data[0] == 1.);
    for (size_t i = 1; i < data_size; i++) {
        if (data[i] == -1.) {
            assert(i + 1 < data_size);
            if (data[i+1] == 1.) {
                n_model++;
            } else {
                assert(data[i+1] == 0.);
                n_noise++;
            }
            start_indices.push_back(i + 1);
        }
    }

    double logz_noise = 0.;
    for (size_t i = 0; i < n; i++) {
        logz_noise += log1exp(unaries_noise[i]);
    }

    double log_nu = std::log(
        static_cast<double>(n_noise) / static_cast<double>(n_model));

    clock_t t;
    t = clock();

    std::vector<size_t> perm(start_indices.size(), 0);
    for (int i = 0; i < start_indices.size(); i++) {
        perm[i] = i;
    }

    std::srand(start_step);
    std::vector<size_t> positions(m, 0);
    std::vector<double> max(m, 0.);
    std::vector<double> sums(m, 0.);
    std::vector<size_t> max_idx(m ,0);
    size_t cidx;
    for (int i = 0; i < n_steps; i++) {
        if (i % start_indices.size() == 0) {
            // permute
            std::random_shuffle(perm.begin(), perm.end());
        }

        // double random = static_cast<double>(std::rand()) / RAND_MAX;
        // size_t idx = static_cast<size_t>(random * (start_indices.size() - 1));
        size_t idx = perm[i % start_indices.size()];

        double step = eta_0 * pow(start_step + i + 1, -power);
        // size_t idx = std::rand() % start_indices.size();
        assert (idx < start_indices.size());
        size_t start_idx = start_indices[idx];
        size_t end_idx;
        if (idx + 1 == start_indices.size()) {
            end_idx = data_size;
        } else {
            end_idx = start_indices[idx + 1] - 1;
        }

        // we cannot handle empty sets yet
        assert (start_idx < end_idx);

        double f_model = *n_logz;
        double f_noise = -logz_noise;

        for (size_t k = start_idx + 1; k < end_idx; k++) {
            f_model += unaries[data[k]];
            f_noise += unaries_noise[data[k]];
        }

        // Compute the maximum for each dimension.
        /*for (size_t j = 0; j < m; j++) {
            max[j] = -1;
        }
        for (size_t k = start_idx + 1; k < end_idx; k++) {
            cidx = IDX(data[k], 0);
        } */
        /*for (size_t j = 0; j < m; j++) {
            double max = -1;
            int max_idx = -1;
            for (size_t k = start_idx + 1; k < end_idx; k++) {
                assert(data[k] >= 0);
                assert(IDX(data[k], j) >= 0);
                assert(IDX(data[k], j) < n * m);
                f_model -= weights[IDX(data[k], j)];
                if (weights[IDX(data[k], j)] > max) {
                    max = weights[IDX(data[k], j)];
                    max_idx = IDX(data[k], j);
                }
            }
            assert(max_idx != -1);
            f_model += max;
            positions[j] = max_idx;
        }*/

        for (size_t j = 0; j < m; j++) {
            double max = -1;
            int max_idx = -1;
            double sum = 0.;
            for (size_t k = start_idx + 1; k < end_idx; k++) {
                assert(data[k] >= 0);
                assert(IDX(data[k], j) >= 0);
                assert(IDX(data[k], j) < n * m);
#ifndef max_only
                f_model -= weights[IDX(data[k], j)];
#endif
#ifdef use_lp
                sum += pow(weights[IDX(data[k], j)], p);
#endif
                if (weights[IDX(data[k], j)] > max) {
                    max = weights[IDX(data[k], j)];
                    max_idx = IDX(data[k], j);
                }
            }
            assert(max_idx != -1);
#ifndef use_lp
            f_model += max;
#endif
#ifdef use_lp
            f_model += pow(sum, 1. / p);
            sums[j] = sum;
#endif
            positions[j] = max_idx;
        }

        // We can now take the gradient step.
        double label = data[start_idx];
        double factor = step * (label - expit(f_model - f_noise - log_nu));

        for (size_t k = start_idx + 1; k < end_idx; k++) {
            assert(data[k] >= 0);
            assert(data[k] < n);
            unaries[data[k]] += factor;
        }

        // TODO Fix behaviour for empty sets

        // factor = -2. * factor * sum;
        // std::cout << "factor: " << factor << std::endl;
        for (size_t j = 0; j < m; j++) {
#ifndef use_lp
            weights[positions[j]] += factor;
#endif
#ifdef use_lp
            for (size_t k = start_idx + 1; k < end_idx; k++) {
                weights[IDX(data[k], j)] += factor * 1. / p * pow(sums[j], 1. / p - 1.) * p * pow(weights[IDX(data[k], j)], p - 1);
            }
#endif

            for (size_t k = start_idx + 1; k < end_idx; k++) {
                // weights[IDX(data[k], j)] = std::max(
                //   0., weights[IDX(data[k], j)] - factor);
                assert(IDX(data[k], j) >= 0);
                assert(IDX(data[k], j) < n * m);
#ifndef max_only
                weights[IDX(data[k], j)] -= factor;
#endif
                if (weights[IDX(data[k], j)] <= 0) {
                     weights[IDX(data[k], j)] = 1e-3 * (
                        static_cast<double>(std::rand()) / RAND_MAX);
                }
            }
        }

#ifdef use_structured_reg
        double norm;
        double add;
        double cstep = 1e-3 * step;
        for (size_t k = start_idx + 1; k < end_idx; k++) {
            cidx = IDX(data[k], 0);
            norm = 0;
            for (size_t j = 0; j < m; j++) {
                // we update the index IDX(data[k], j)
                norm += weights[cidx];
                cidx++;
            }

            if (norm < 1.0) {
                add = cstep;
            } else {
                add = - cstep;
            }

            cidx = IDX(data[k], 0);
            for (size_t j = 0; j < m; j++) {
                // we update the index IDX(data[k], j)
                weights[cidx] += add;
                if (weights[cidx] <= 0) {
                     weights[cidx] = 1e-3 * (
                        static_cast<double>(std::rand()) / RAND_MAX);
                }
                cidx++;
            }
        }
#endif

#ifdef use_lower_bound
        for (size_t k = start_idx + 1; k < end_idx; k++) {
            double norm = 0;
            for (size_t j = 0; j < m; j++) {
                norm += weights[IDX(data[k], j)];
            }

            double add = (1.0 - norm) / static_cast<double>(m);

            // norm = sqrt(norm);
            if (norm < 1.0) {
                for (size_t j = 0; j < m; j++) {
                    // weights[IDX(data[k], j)] += add;
                    weights[IDX(data[k], j)] /= norm;
                }
            }
        }
#endif

#ifdef use_l1_reg
        for (size_t k = 0; k < n; k++) {
            for (size_t j = 0; j < m; j++) {
                weights[IDX(k, j)] -= 1e-3 * step;
                if (weights[IDX(k, j)] <= 0) {
                    // weights[IDX(k, j)] = 1e-3 * (
                    //    static_cast<double>(std::rand()) / RAND_MAX);
                    weights[IDX(k, j)] = 0;
                }
            }
        }
#endif


#ifdef use_l2_reg
        for (size_t k = 0; k < n; k++) {
            for (size_t j = 0; j < m; j++) {
                weights[IDX(k, j)] -= 2. * weights[IDX(k, j)] * 1e-3 * step;
                if (weights[IDX(k, j)] <= 0) {
                    weights[IDX(k, j)] = 1e-3 * (
                        static_cast<double>(std::rand()) / RAND_MAX);
                    // weights[IDX(k, j)] = 0;
                }
            }
        }
#endif


        *n_logz += factor;
    }

    t = clock() - t;
    std::cout << "It took me "  << (((float)t)/CLOCKS_PER_SEC) << "seconds" << std::endl;
}


int* sample(const double *probabilities, int n, int n_samples, int* out_size) {
    std::vector<int> probabilities_int;
    probabilities_int.reserve(n);
    double total = 0.;
    for (int i = 0; i < n; i++) {
        probabilities_int.push_back(static_cast<int>(
                probabilities[i] * RAND_MAX));
        total += probabilities[i];
    }
    // Reserve double the expected size;
    int expected_size = static_cast<int>(total * n_samples * 2 + n_samples);
    int* samples = static_cast<int *>(malloc(sizeof(int) * expected_size));
    int next = 0;

    std::srand(time(0));
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n; j++) {
            if (next + 1 < expected_size) {
                int* new_samples = static_cast<int*>(
                    realloc(samples, sizeof(int) * 2 * expected_size));
                if (new_samples == NULL) {
                    new_samples = static_cast<int*>(
                        malloc(sizeof(int) * 2 * expected_size));
                    assert(new_samples != NULL);
                    memcpy(new_samples, samples, sizeof(int) * expected_size);
                    free(samples);
                    expected_size *= 2;
                }
                samples = new_samples;
            }
            if (rand() <= probabilities_int[j]) {
                samples[next++] = j;
            }
        }
        samples[next++] = -1;
    }

    *out_size = next;
    return samples;
}
