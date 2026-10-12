#pragma once

#include <stdint.h>
#include <type_traits>
#include <string>
#include <cassert>

template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
class WrappedFloat {
public:
    using value_type = T;

    explicit WrappedFloat(T fl);
    WrappedFloat() = default;

    operator T() const { return m_val; }

    WrappedFloat operator+(WrappedFloat rhs) const { return WrappedFloat(m_val + rhs.m_val); }
    WrappedFloat operator-(WrappedFloat rhs) const { return WrappedFloat(m_val - rhs.m_val); }
    WrappedFloat operator*(WrappedFloat rhs) const { return WrappedFloat(m_val * rhs.m_val); }
    WrappedFloat operator/(WrappedFloat rhs) const { return WrappedFloat(m_val / rhs.m_val); }

    WrappedFloat& operator+=(WrappedFloat rhs) { m_val += rhs.m_val; return *this; }
    WrappedFloat& operator-=(WrappedFloat rhs) { m_val -= rhs.m_val; return *this; }
    WrappedFloat& operator*=(WrappedFloat rhs) { m_val *= rhs.m_val; return *this; }
    WrappedFloat& operator/=(WrappedFloat rhs) { m_val /= rhs.m_val; return *this; }

    bool operator==(WrappedFloat rhs) const { return m_val == rhs.m_val; }
    bool operator!=(WrappedFloat rhs) const { return m_val != rhs.m_val; }
    bool operator<(WrappedFloat rhs)  const { return m_val < rhs.m_val; }
    bool operator>(WrappedFloat rhs)  const { return m_val > rhs.m_val; }
    bool operator<=(WrappedFloat rhs) const { return m_val <= rhs.m_val; }
    bool operator>=(WrappedFloat rhs) const { return m_val >= rhs.m_val; }

    WrappedFloat operator-() const { return WrappedFloat(-m_val); }
    WrappedFloat operator+() const { return WrappedFloat(+m_val); }

    T value() const { return m_val; }

    WrappedFloat& operator=(T v) { m_val = v; return *this; }
private:
    T m_val;
};

using mfloat = WrappedFloat<float>;
using mdouble = WrappedFloat<double>;

static_assert(sizeof(mfloat) == sizeof(float));
static_assert(sizeof(mdouble) == sizeof(double));
