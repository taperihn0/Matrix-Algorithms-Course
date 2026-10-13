#pragma once

#include <cstddef>
#include <vector>
#include <stdexcept>
#include <iostream>
#include "types.hpp"

template <typename T>
class Matrix2x {
public:
    using value_type = T;
    using size_type = std::size_t;

    Matrix2x() = default;
    explicit Matrix2x(size_type n, size_type m);

    size_type rows() const;
    size_type cols() const;

    Matrix2x operator+(const Matrix2x& rhs) const;
    Matrix2x operator-(const Matrix2x& rhs) const;
    Matrix2x operator*(const Matrix2x& rhs) const;
    Matrix2x operator/(const Matrix2x& rhs) const;

    Matrix2x operator+(T rhs) const;
    Matrix2x operator-(T rhs) const;
    Matrix2x operator*(T rhs) const;
    Matrix2x operator/(T rhs) const;

    Matrix2x& operator+=(const Matrix2x& rhs);
    Matrix2x& operator-=(const Matrix2x& rhs);
    Matrix2x& operator*=(const Matrix2x& rhs);
    Matrix2x& operator/=(const Matrix2x& rhs);

    Matrix2x& operator+=(T rhs);
    Matrix2x& operator-=(T rhs);
    Matrix2x& operator*=(T rhs);
    Matrix2x& operator/=(T rhs);

    Matrix2x(std::initializer_list<std::initializer_list<T>> init);

    bool operator==(const Matrix2x& rhs) const;
    bool operator!=(const Matrix2x& rhs) const;

    T& at(size_type i, size_type j);
    const T& at(size_type i, size_type j) const;

    T& operator()(size_type i, size_type j);
    const T& operator()(size_type i, size_type j) const;

    T* data();
    const T* data() const;

private:
    void check_index(size_type i, size_type j) const;

    size_type m_rows;
    size_type m_cols;
    std::vector<T> m_data;
};

template <typename T>
class Matrix2xView {
public:
    using size_type = std::size_t;

    Matrix2xView() = default;

    Matrix2xView(const T* data,
                 size_type orig_cols,
                 size_type start_row,
                 size_type start_col,
                 size_type view_rows,
                 size_type view_cols)
        : m_data(const_cast<T*>(data))
        , m_orig_cols(orig_cols)
        , m_start_row(start_row)
        , m_start_col(start_col)
        , m_rows(view_rows)
        , m_cols(view_cols)
    {}

    Matrix2xView(T* data, 
                 size_type orig_cols, 
                 size_type start_row, 
                 size_type start_col, 
                 size_type view_rows, 
                 size_type view_cols)
        : m_data(data)
        , m_orig_cols(orig_cols)
        , m_start_row(start_row)
        , m_start_col(start_col)
        , m_rows(view_rows)
        , m_cols(view_cols)
    {}

    Matrix2xView(const Matrix2xView& other, 
                 size_type orig_row, 
                 size_type orig_col, 
                 size_type view_rows, 
                 size_type view_cols)
        : m_data(other.m_data)
        , m_orig_cols(other.m_orig_cols)
        , m_start_row(other.m_start_row + orig_row)
        , m_start_col(other.m_start_col + orig_col)
        , m_rows(view_rows)
        , m_cols(view_cols)
    {}

    size_type rows() const { return m_rows; }
    size_type cols() const { return m_cols; }

    const T* data() const { return m_data; }

    const T& at(size_type i, size_type j) const {
        return *(m_data + (m_start_row + i) * m_orig_cols + (m_start_col + j));
    }

    T& at(size_type i, size_type j) {
        return *(m_data + (m_start_row + i) * m_orig_cols + (m_start_col + j));
    }

    const T& operator()(size_type i, size_type j) const { return at(i, j); }
    T& operator()(size_type i, size_type j) { return at(i, j); }

private:
    T* m_data = nullptr;
    size_type m_orig_cols = 0;
    size_type m_start_row = 0;
    size_type m_start_col = 0;
    size_type m_rows = 0;
    size_type m_cols = 0;
};

enum class enumMatMult {
    Default,
    Binet,
    Strassen,
    Laderman
};

template <typename T>
Matrix2x<T> matrix_mul(const Matrix2x<T>& a, const Matrix2x<T>& b, enumMatMult alg);

template <typename T>
inline Matrix2x<T>::Matrix2x(size_type n, size_type m)
    : m_rows(n)
    , m_cols(m)
    , m_data(n * m) 
{}

template <typename T>
inline Matrix2x<T>::Matrix2x(std::initializer_list<std::initializer_list<T>> init)
    : m_rows(init.size())
    , m_cols()
    , m_data()
{
    if (init.size() == 0) {
        m_cols = 0;
        return;
    }

    m_cols = init.begin()->size();
    m_data.reserve(m_rows * m_cols);

    for (const auto& row : init) {
        if (row.size() != m_cols) {
            throw std::invalid_argument("Matrix2x: invalid initializer list");
        }

        for (const auto& v : row) {
            m_data.push_back(v);
        }
    }
}

template <typename T>
inline typename Matrix2x<T>::size_type Matrix2x<T>::rows() const {
    return m_rows;
}

template <typename T>
inline typename Matrix2x<T>::size_type Matrix2x<T>::cols() const {
    return m_cols;
}

template <typename T>
inline void Matrix2x<T>::check_index(size_type i, size_type j) const {
#ifdef DEBUG
    if (i >= m_rows || j >= m_cols) {
        throw std::out_of_range("Index out of range");
    }
#else
    (void)i;
    (void)j;
#endif
}

template <typename T>
inline T& Matrix2x<T>::at(size_type i, size_type j) {
    check_index(i, j);
    return m_data[i * m_cols + j];
}

template <typename T>
inline const T& Matrix2x<T>::at(size_type i, size_type j) const {
    check_index(i, j);
    return m_data[i * m_cols + j];
}

template <typename T>
inline T& Matrix2x<T>::operator()(size_type i, size_type j) {
    return at(i, j);
}

template <typename T>
inline const T& Matrix2x<T>::operator()(size_type i, size_type j) const {
    return at(i, j);
}

template <typename T>
T* Matrix2x<T>::data() {
    return m_data.data();
}

template <typename T>
const T* Matrix2x<T>::data() const {
    return m_data.data();
}

template <typename T>
inline Matrix2x<T> Matrix2x<T>::operator+(const Matrix2x& rhs) const {
    if (rows() != rhs.rows() || cols() != rhs.cols()) {
        throw std::invalid_argument("Size mismatch for addition");
    }

    Matrix2x res(rows(), cols());

    for (size_type i = 0; i < m_data.size(); ++i) {
        res.m_data[i] = m_data[i] + rhs.m_data[i];
    }

    return res;
}

template <typename T>
inline Matrix2x<T> Matrix2x<T>::operator-(const Matrix2x& rhs) const {
    if (rows() != rhs.rows() || cols() != rhs.cols()) {
        throw std::invalid_argument("Matrix2x: matrix size mismatch");
    }

    Matrix2x res(rows(), cols());

    for (size_type i = 0; i < m_data.size(); ++i) {
        res.m_data[i] = m_data[i] - rhs.m_data[i];
    }

    return res;
}

template <typename T>
inline Matrix2x<T> Matrix2x<T>::operator*(const Matrix2x& rhs) const {
    if (rows() != rhs.rows() || cols() != rhs.cols()) {
        throw std::invalid_argument("Matrix2x: matrix size mismatch");
    }

    Matrix2x res(rows(), cols());

    for (size_type i = 0; i < m_data.size(); ++i) {
        res.m_data[i] = m_data[i] * rhs.m_data[i];
    }

    return res;
}

template <typename T>
inline Matrix2x<T> Matrix2x<T>::operator/(const Matrix2x& rhs) const {
    if (rows() != rhs.rows() || cols() != rhs.cols()) {
        throw std::invalid_argument("Matrix2x: matrix size mismatch");
    }

    Matrix2x res(rows(), cols());

    for (size_type i = 0; i < m_data.size(); ++i) {
        res.m_data[i] = m_data[i] / rhs.m_data[i];
    }

    return res;
}

template <typename T>
inline Matrix2x<T> Matrix2x<T>::operator+(T rhs) const {
    Matrix2x res(rows(), cols());

    for (size_type i = 0; i < m_data.size(); ++i) {
        res.m_data[i] = m_data[i] + rhs;
    }

    return res;
}

template <typename T>
inline Matrix2x<T> Matrix2x<T>::operator-(T rhs) const {
    Matrix2x res(rows(), cols());

    for (size_type i = 0; i < m_data.size(); ++i) {
        res.m_data[i] = m_data[i] - rhs;
    }

    return res;
}

template <typename T>
inline Matrix2x<T> Matrix2x<T>::operator*(T rhs) const {
    Matrix2x res(rows(), cols());

    for (size_type i = 0; i < m_data.size(); ++i) {
        res.m_data[i] = m_data[i] * rhs;
    }

    return res;
}

template <typename T>
inline Matrix2x<T> Matrix2x<T>::operator/(T rhs) const {
    Matrix2x res(rows(), cols());

    for (size_type i = 0; i < m_data.size(); ++i) {
        res.m_data[i] = m_data[i] / rhs;
    }

    return res;
}

template <typename T>
inline Matrix2x<T>& Matrix2x<T>::operator+=(const Matrix2x& rhs) {
    *this = *this + rhs;
    return *this;
}

template <typename T>
inline Matrix2x<T>& Matrix2x<T>::operator-=(const Matrix2x& rhs) {
    *this = *this - rhs;
    return *this;
}

template <typename T>
inline Matrix2x<T>& Matrix2x<T>::operator*=(const Matrix2x& rhs) {
    *this = *this * rhs;
    return *this;
}

template <typename T>
inline Matrix2x<T>& Matrix2x<T>::operator/=(const Matrix2x& rhs) {
    *this = *this / rhs;
    return *this;
}

template <typename T>
inline Matrix2x<T>& Matrix2x<T>::operator+=(T rhs) {
    *this = *this + rhs;
    return *this;
}

template <typename T>
inline Matrix2x<T>& Matrix2x<T>::operator-=(T rhs) {
    *this = *this - rhs;
    return *this;
}

template <typename T>
inline Matrix2x<T>& Matrix2x<T>::operator*=(T rhs) {
    *this = *this * rhs;
    return *this;
}

template <typename T>
inline Matrix2x<T>& Matrix2x<T>::operator/=(T rhs) {
    *this = *this / rhs;
    return *this;
}

template <typename T>
inline bool Matrix2x<T>::operator==(const Matrix2x& rhs) const {
    return rows() == rhs.rows() && cols() == rhs.cols() && m_data == rhs.m_data;
}

template <typename T>
inline bool Matrix2x<T>::operator!=(const Matrix2x& rhs) const {
    return !(*this == rhs);
}

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const Matrix2x<T>& m) {
    for (typename Matrix2x<T>::size_type i = 0; i < m.rows(); ++i) {
        for (typename Matrix2x<T>::size_type j = 0; j < m.cols(); ++j) {
            if (j > 0) os << ' ';
            os << m.at(i, j);
        }
        os << '\n';
    }
    return os;
}

using Matrix2xmf = Matrix2x<mfloat>;
using Matrix2xmd = Matrix2x<mdouble>;
