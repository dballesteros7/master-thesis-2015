#ifndef __TRAIN_H
#define __TRAIN_H

#ifdef __cplusplus
extern "C" {
    void train(const long *data, size_t data_size, long n_steps,
               double eta_0, double power, int start_step,
               const double *unaries_noise,
               double *weights, double *unaries, double *n_logz,
               size_t n, size_t m);

    int* sample(const double *probabilities, int n, int n_samples,
                int *out_size);
}
#endif

#endif  // __TRAIN_H


