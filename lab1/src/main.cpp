#include <iostream>
#include "matrix.hpp"

int main() {
    Matrix2x<mfloat> m(2, 3);

    m[0][0] = 1.0f;
    m[0][1] = 2.0f;
    m[0][2] = 3.0f;
    m[1][0] = 4.0f;
    m[1][1] = 5.0f;
    m[1][2] = 6.0f;

    std::cout << "m[0][1] = " << m[0][1] << std::endl;
    std::cout << "m[1][2] = " << m[1][2] << std::endl;

    auto t = Matrix2x<mfloat>::transpose(m);
    std::cout << "t[0][1] = " << t[0][1] << std::endl;
    std::cout << "t[1][0] = " << t[1][0] << std::endl;

    return 0;
}
