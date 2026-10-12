#include <iostream>
#include "matrix.hpp"

int main() {
    Matrix2x<float> m({
        {1., 2., 3.},
        {4., 5., 6.}
    });

    std::cout << m << std::endl;
}
