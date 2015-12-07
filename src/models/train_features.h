#ifndef MASTER_THESIS_2015_TRAIN_FEATURES_H
#define MASTER_THESIS_2015_TRAIN_FEATURES_H

void train(const long *data, size_t data_size, long n_steps,
           double eta_0, double power, int start_step,
           const double *features,
           double *b_weights, double *a_weights, double *n_logz,
           size_t n_items, size_t l_dim, size_t m_feat);

void train_with_features(std::string data_file_path,
                         std::string features_file_path,
                         std::string weights_file_path,
                         int n_steps, double eta_0, double iter_power,
                         std::string output_file_path);

#endif //MASTER_THESIS_2015_TRAIN_FEATURES_H
