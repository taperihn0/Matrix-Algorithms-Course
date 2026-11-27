#pragma once

#include "lab1/src/matrix.hpp"

#include <filesystem>

namespace mutils::picture
{

// red, green and blue channels of a loaded picture.
// every stored value is in range [0, 255].
template <typename T>
struct Channels
{
  Matrix2x<T> red;
  Matrix2x<T> green;
  Matrix2x<T> blue;
};

// loads a picture and splits it into rgb channels.
// supported formats: uncompressed 24/32 bit BMP and binary PPM (P6);
// the format is detected from the file contents.
// throws std::runtime_error when the file cannot be read or is not supported.
template <typename T = float>
Channels<T> load(const std::filesystem::path &path);

} // namespace mutils::picture
