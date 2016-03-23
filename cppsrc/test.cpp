#include <iostream>
#include <limits>

#include <Eigen/Dense>

using namespace Eigen;


struct Visitor {
    void init(const double& value, EIGEN_DEFAULT_DENSE_INDEX_TYPE i, EIGEN_DEFAULT_DENSE_INDEX_TYPE j) {

    }
    void operator() (const double &value, EIGEN_DEFAULT_DENSE_INDEX_TYPE i, EIGEN_DEFAULT_DENSE_INDEX_TYPE j) {
        std::cout << value << " " << i << " " << j  << std::endl;
    }
};

int main(int argc, char* argv[]) {
    std::cout << std::numeric_limits<double>::min() << std::endl;
    Matrix3d m = Matrix3d::Random();
    Array3i indices();
    std::cout << m << std::endl;
    Visitor v;
    m.visit(v);
}