#pragma once

#include <cstddef>
#include <memory>
#include <new>
#include <utility>

template <typename T> class Matrix2xView;

template <typename T> class IntermediateBuffer {
public:
  IntermediateBuffer() = delete;

  explicit IntermediateBuffer(std::size_t buf_size)
      : m_size(buf_size) {
    m_data.reset(static_cast<T *>(::operator new(buf_size * sizeof(T),
                                           std::align_val_t(
                                               std::hardware_destructive_interference_size))));
    std::uninitialized_value_construct_n(m_data.get(), buf_size);
  }

  Matrix2xView<T> query_view(std::size_t view_rows, std::size_t view_cols);

  std::size_t mark() const { return m_pointer; }
  void revert_to(std::size_t mark);

private:
#ifdef DEBUG
  void check_overflow(std::size_t req) {
    if (m_pointer + req > m_size) {
      throw std::runtime_error("IntermediateBuffer: pointer overflow");
    }
  }
#endif

  std::unique_ptr<T[]> m_data = nullptr;
  std::size_t m_size = 0uz;
  std::size_t m_pointer = 0uz;
};

template <typename T>
inline Matrix2xView<T>
IntermediateBuffer<T>::query_view(std::size_t view_rows, std::size_t view_cols) {
  const std::size_t req = view_rows * view_cols;

#ifdef DEBUG
  check_overflow(req);
#endif

  const auto mv = Matrix2xView<T>(m_data.get() + m_pointer, view_cols, 0, 0,
                                  view_rows, view_cols);
  m_pointer += req;
  return mv;
}

template <typename T>
inline void IntermediateBuffer<T>::revert_to(std::size_t mark) {
  assert(mark <= m_pointer);
  m_pointer = mark;
}
