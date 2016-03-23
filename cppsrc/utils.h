#ifndef CPPSRC_UTILS_H
#define CPPSRC_UTILS_H

#include <string>
#include <vector>
#include <Eigen/Dense>

namespace masterthesis {
    double expit(double x);

    double log1exp(double x);

    void readFeatureFile(std::string path,
                         size_t* const nItems, size_t* const mFeatures,
                         std::vector<double>* features);

    void readFeatureFileBoost(std::string path,
                              size_t &nItems, size_t &mFeatures,
                              Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &features);
}

#endif //CPPSRC_UTILS_H
