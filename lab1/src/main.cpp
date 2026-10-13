#include <iostream>
#include <sstream>
#include <stdexcept>
#include "matrix.hpp"
#include "tests.hpp"

int main() {
    Matrix2x<float> m({
        {1., 2., 3.},
        {1., 2., 3.}
    });

    std::cout << m << std::endl;
    return 0;
}
