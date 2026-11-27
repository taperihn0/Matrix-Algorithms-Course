#pragma once

#include "../../lab1/src/matrix.hpp"
#include "../../lab1/src/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>

template <typename T>
class Vector {
public:
  using value_type = T;
  using size_type = std::size_t;

  Vector() = default;
  explicit Vector(size_type n);
  Vector(std::initializer_list<T> init);
  Vector(const Vector &other);
  Vector(Vector &&other) noexcept;
  ~Vector() = default;

  Vector &operator=(const Vector &other);
  Vector &operator=(Vector &&other) noexcept;

  size_type len() const;

  // vector filled with zeros
  static Vector zeros(size_type n);

  // vector with random values in [-1, 1)
  static Vector random(size_type n);

  T &at(size_type i);
  const T &at(size_type i) const;

  T &operator[](size_type i);
  const T &operator[](size_type i) const;

  T *data();
  const T *data() const;

  // dot product with another vector
  T dot(const Vector &rhs) const;

  // euclidean norm
  T norm() const;

  Vector operator*(T rhs) const;
  Vector operator/(T rhs) const;
  Vector operator-(const Vector &rhs) const;

  // row vector times matrix
  Vector operator*(const Matrix2x<T> &rhs) const;

  Vector &operator*=(T rhs);
  Vector &operator/=(T rhs);
  Vector &operator-=(const Vector &rhs);

  Matrix2x<T> outer(const Vector<T> &v) const;

private:
  void check_index(size_type i) const;

  size_type m_len = 0;
  std::unique_ptr<T[]> m_data = nullptr;
};

template <typename T>
inline Vector<T> operator*(T lhs, const Vector<T> &rhs) {
  return rhs * lhs;
}

template <typename T>
inline Vector<T>::Vector(size_type n) : m_len(n) {
  if (m_len == 0) {
    return;
  }
  m_data = std::make_unique<T[]>(m_len);
}

template <typename T>
inline Vector<T>::Vector(std::initializer_list<T> init) : m_len(init.size()) {
  if (m_len == 0) {
    return;
  }
  m_data = std::make_unique<T[]>(m_len);
  std::copy(init.begin(), init.end(), m_data.get());
}

template <typename T>
inline Vector<T>::Vector(const Vector &other)
    : m_len(other.m_len), m_data() {
  if (m_len == 0) {
    return;
  }
  m_data = std::make_unique<T[]>(m_len);
  std::copy_n(other.m_data.get(), m_len, m_data.get());
}

template <typename T>
inline Vector<T>::Vector(Vector &&other) noexcept
    : m_len(std::exchange(other.m_len, 0)),
      m_data(std::exchange(other.m_data, nullptr)) {}

template <typename T>
inline Vector<T> &Vector<T>::operator=(const Vector &other) {
  if (this == &other) {
    return *this;
  }
  Vector tmp(other);
  std::swap(m_len, tmp.m_len);
  std::swap(m_data, tmp.m_data);
  return *this;
}

template <typename T>
inline Vector<T> &Vector<T>::operator=(Vector &&other) noexcept {
  if (this == &other) {
    return *this;
  }
  m_len = std::exchange(other.m_len, 0);
  m_data = std::exchange(other.m_data, nullptr);
  return *this;
}

template <typename T>
inline typename Vector<T>::size_type Vector<T>::len() const {
  return m_len;
}

template <typename T>
inline Vector<T> Vector<T>::zeros(size_type n) {
  Vector res(n);

  for (size_type i = 0; i < n; ++i) {
    res.m_data[i] = static_cast<T>(0);
  }

  return res;
}

template <typename T>
inline Vector<T> Vector<T>::random(size_type n) {
  static thread_local std::mt19937 engine(std::random_device{}());
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  Vector res(n);

  for (size_type i = 0; i < n; ++i) {
    res.m_data[i] = static_cast<T>(dist(engine));
  }

  return res;
}

template <typename T>
inline void Vector<T>::check_index(size_type i) const {
#ifdef DEBUG
  if (i >= m_len) {
    throw std::out_of_range("Index out of range");
  }
#else
  (void)i;
#endif
}

template <typename T> inline T &Vector<T>::at(size_type i) {
  check_index(i);
  return m_data[i];
}

template <typename T> inline const T &Vector<T>::at(size_type i) const {
  check_index(i);
  return m_data[i];
}

template <typename T> inline T &Vector<T>::operator[](size_type i) {
  return at(i);
}

template <typename T>
inline const T &Vector<T>::operator[](size_type i) const {
  return at(i);
}

template <typename T> inline T *Vector<T>::data() { return m_data.get(); }

template <typename T> inline const T *Vector<T>::data() const {
  return m_data.get();
}

template <typename T> inline T Vector<T>::dot(const Vector &rhs) const {
  if (len() != rhs.len()) {
    throw std::invalid_argument("Vector: size mismatch for dot product");
  }

  T sum = static_cast<T>(0);

  for (size_type i = 0; i < m_len; ++i) {
    sum += m_data[i] * rhs.m_data[i];
  }

  return sum;
}

template <typename T> inline T Vector<T>::norm() const {
  return static_cast<T>(std::sqrt(static_cast<double>(dot(*this))));
}

template <typename T> inline Vector<T> Vector<T>::operator*(T rhs) const {
  Vector res(m_len);

  for (size_type i = 0; i < m_len; ++i) {
    res.m_data[i] = m_data[i] * rhs;
  }

  return res;
}

template <typename T> inline Vector<T> Vector<T>::operator/(T rhs) const {
  Vector res(m_len);

  for (size_type i = 0; i < m_len; ++i) {
    res.m_data[i] = m_data[i] / rhs;
  }

  return res;
}

template <typename T>
inline Vector<T> Vector<T>::operator-(const Vector &rhs) const {
  if (len() != rhs.len()) {
    throw std::invalid_argument("Vector: size mismatch for subtraction");
  }

  Vector res(m_len);

  for (size_type i = 0; i < m_len; ++i) {
    res.m_data[i] = m_data[i] - rhs.m_data[i];
  }

  return res;
}

template <typename T>
inline Vector<T> Vector<T>::operator*(const Matrix2x<T> &rhs) const {
  if (len() != rhs.rows()) {
    throw std::invalid_argument("Vector: size mismatch for matrix multiplication");
  }

  Vector res(rhs.cols());

  for (size_type j = 0; j < rhs.cols(); ++j) {
    T sum = static_cast<T>(0);

    for (size_type i = 0; i < m_len; ++i) {
      sum += m_data[i] * rhs(i, j);
    }

    res.m_data[j] = sum;
  }

  return res;
}

template <typename T> inline Vector<T> &Vector<T>::operator*=(T rhs) {
  *this = *this * rhs;
  return *this;
}

template <typename T> inline Vector<T> &Vector<T>::operator/=(T rhs) {
  *this = *this / rhs;
  return *this;
}

template <typename T>
inline Vector<T> &Vector<T>::operator-=(const Vector &rhs) {
  *this = *this - rhs;
  return *this;
}

template <typename T>
Matrix2x<T> Vector<T>::outer(const Vector<T> &v) const
{
  Matrix2x<T> res(len(), v.len());

  for (auto i = 0uz; i < len(); ++i) {
    for (auto j = 0uz; j < v.len(); ++j) {
      res(i, j) = this->operator[](i) * v[j];
    }
  }

  return res;
}

using Vectormf = Vector<mfloat>;
using Vectormd = Vector<mdouble>;
