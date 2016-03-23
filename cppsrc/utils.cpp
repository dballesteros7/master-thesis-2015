#include "utils.h"

#include <cmath>
#include <fstream>

#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>

#include <Eigen/Dense>

namespace masterthesis {
    double expit(double x) {
        return x > 0 ? 1 / (1 + std::exp(-x)) : 1 - 1 / (1 + std::exp(x));
    }

    double log1exp(double x) {
        return x > 0 ? x + log1p(exp(-x)) : log1p(exp(x));
    }

    void readFeatureFile(std::string path,
                         size_t* const nItems, size_t* const mFeatures,
                         std::vector<double>* const features) {
        std::fstream features_file_input;
        features_file_input.open(path, std::ios::in);

        std::string line;
        std::vector<std::string> tokens;
        getline(features_file_input, line);
        boost::split(tokens, line, boost::is_any_of(","));
        *nItems = stoul(tokens[0]);
        *mFeatures = stoul(tokens[1]);

        features->reserve(*nItems * *mFeatures);
        for (size_t i = 0; i < *nItems; ++i) {
            getline(features_file_input, line);
            boost::split(tokens, line, boost::is_any_of(","));
            for (size_t j = 0; j < *mFeatures; ++j) {
                (*features)[(i * *mFeatures) + j] = stod(tokens[j]);
            }
        }
    }

    void readFeatureFileBoost(std::string path,
                              size_t &nItems, size_t &mFeatures,
                              Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &features) {
        std::fstream features_file_input;
        features_file_input.open(path, std::ios::in);

        std::string line;
        std::vector<std::string> tokens;
        getline(features_file_input, line);
        boost::split(tokens, line, boost::is_any_of(","));
        nItems = stoul(tokens[0]);
        mFeatures = stoul(tokens[1]);

        features.resize(nItems, mFeatures);
        for (size_t i = 0; i < nItems; ++i) {
            getline(features_file_input, line);
            boost::split(tokens, line, boost::is_any_of(","));
            for (size_t j = 0; j < mFeatures; ++j) {
                features(i, j) = stod(tokens[j]);
            }
        }
    }
}