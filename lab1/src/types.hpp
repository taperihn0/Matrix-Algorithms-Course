#pragma once

#include <cassert>
#include <istream>
#include <stdint.h>
#include <string>
#include <type_traits>

template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
class WrappedFloat {
public:
  inline static std::size_t add = 0;
  inline static std::size_t sub = 0;
  inline static std::size_t divs = 0;
  inline static std::size_t mul = 0;

  using value_type = T;

  explicit WrappedFloat(T fl) : m_val(fl) {}
  WrappedFloat() = default;

  operator T() const { return m_val; }

  WrappedFloat operator+(WrappedFloat rhs) const {
    ++add;
    return WrappedFloat(m_val + rhs.m_val);
  }
  WrappedFloat operator-(WrappedFloat rhs) const {
    ++sub;
    return WrappedFloat(m_val - rhs.m_val);
  }
  WrappedFloat operator*(WrappedFloat rhs) const {
    ++mul;
    return WrappedFloat(m_val * rhs.m_val);
  }
  WrappedFloat operator/(WrappedFloat rhs) const {
    ++divs;
    return WrappedFloat(m_val / rhs.m_val);
  }

  WrappedFloat &operator+=(WrappedFloat rhs) {
    ++add;
    m_val += rhs.m_val;
    return *this;
  }
  WrappedFloat &operator-=(WrappedFloat rhs) {
    ++sub;
    m_val -= rhs.m_val;
    return *this;
  }
  WrappedFloat &operator*=(WrappedFloat rhs) {
    ++mul;
    m_val *= rhs.m_val;
    return *this;
  }
  WrappedFloat &operator/=(WrappedFloat rhs) {
    ++divs;
    m_val /= rhs.m_val;
    return *this;
  }

  bool operator==(WrappedFloat rhs) const { return m_val == rhs.m_val; }
  bool operator!=(WrappedFloat rhs) const { return m_val != rhs.m_val; }
  bool operator<(WrappedFloat rhs) const { return m_val < rhs.m_val; }
  bool operator>(WrappedFloat rhs) const { return m_val > rhs.m_val; }
  bool operator<=(WrappedFloat rhs) const { return m_val <= rhs.m_val; }
  bool operator>=(WrappedFloat rhs) const { return m_val >= rhs.m_val; }

  WrappedFloat operator-() const { return WrappedFloat(-m_val); }
  WrappedFloat operator+() const { return WrappedFloat(+m_val); }

  T value() const { return m_val; }

  WrappedFloat &operator=(T v) {
    m_val = v;
    return *this;
  }
  WrappedFloat &operator=(const WrappedFloat &other) {
    m_val = other.m_val;
    return *this;
  }

  void static reset_counters(){
    add = 0;
    sub = 0;
    mul = 0;
    divs = 0;
  }

private:
  T m_val;
};

template <typename T, typename Enable>
std::istream &operator>>(std::istream &is, WrappedFloat<T, Enable> &value) {
  T raw;
  if (is >> raw) {
    value = raw;
  }
  return is;
}

using mfloat = WrappedFloat<float>;
using mdouble = WrappedFloat<double>;

static_assert(sizeof(mfloat) == sizeof(float));
static_assert(sizeof(mdouble) == sizeof(double));
