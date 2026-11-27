#include "picture.hpp"

#include <cstdint>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace mutils::picture
{
namespace
{

using Byte = std::uint8_t;

constexpr std::size_t FileHeaderSize = 14;
constexpr std::size_t InfoHeaderSize = 40;
constexpr std::uint32_t MaxHeaderNumber = 1u << 24;

// picture with the rows stored top to bottom.
struct Pixels
{
  std::size_t width = 0;
  std::size_t height = 0;
  std::vector<Byte> rgb; // width * height * 3 bytes
};

std::vector<Byte> read_file(const std::filesystem::path &path)
{
  std::ifstream file(path, std::ios::binary);

  if (!file) {
    throw std::runtime_error("picture: cannot open file '" + path.string() + "'");
  }

  return std::vector<Byte>(std::istreambuf_iterator<char>(file),
                           std::istreambuf_iterator<char>());
}

std::uint16_t le_u16(const Byte *p)
{
  return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
}

std::uint32_t le_u32(const Byte *p)
{
  return static_cast<std::uint32_t>(p[0]) |
         (static_cast<std::uint32_t>(p[1]) << 8) |
         (static_cast<std::uint32_t>(p[2]) << 16) |
         (static_cast<std::uint32_t>(p[3]) << 24);
}

Pixels decode_bmp(const std::vector<Byte> &bytes)
{
  if (bytes.size() < FileHeaderSize + InfoHeaderSize) {
    throw std::runtime_error("picture: truncated BMP header");
  }

  const auto data_offset = static_cast<std::size_t>(le_u32(&bytes[10]));
  const auto header_size = static_cast<std::size_t>(le_u32(&bytes[14]));
  const auto raw_width = static_cast<std::int32_t>(le_u32(&bytes[18]));
  const auto raw_height = static_cast<std::int32_t>(le_u32(&bytes[22]));
  const auto planes = le_u16(&bytes[26]);
  const auto bpp = le_u16(&bytes[28]);
  const auto compression = le_u32(&bytes[30]);

  if (header_size < InfoHeaderSize) {
    throw std::runtime_error("picture: unsupported BMP header");
  }
  if (raw_width <= 0 || raw_height == 0) {
    throw std::runtime_error("picture: invalid BMP size");
  }
  if (planes != 1 || (bpp != 24 && bpp != 32)) {
    throw std::runtime_error("picture: only 24 and 32 bit BMP files are supported");
  }
  if (compression != 0) {
    throw std::runtime_error("picture: compressed BMP files are not supported");
  }

  const auto w = static_cast<std::size_t>(raw_width);
  const auto h = static_cast<std::size_t>(
      raw_height < 0 ? -static_cast<std::int64_t>(raw_height) : raw_height);
  const bool top_down = raw_height < 0;

  const std::size_t bytes_pp = bpp / 8;
  // every row is padded to a multiple of 4 bytes.
  const std::size_t stride = ((w * bytes_pp + 3) / 4) * 4;

  if (data_offset > bytes.size() ||
      data_offset < FileHeaderSize + header_size) {
    throw std::runtime_error("picture: invalid BMP pixel data offset");
  }
  if (h != 0 && stride > (bytes.size() - data_offset) / h) {
    throw std::runtime_error("picture: truncated BMP pixel data");
  }

  Pixels px;
  px.width = w;
  px.height = h;
  px.rgb.resize(w * h * 3);

  for (std::size_t row = 0; row < h; ++row) {
    const auto *src = &bytes[data_offset + row * stride];
    const auto y = top_down ? row : h - 1 - row;

    for (std::size_t x = 0; x < w; ++x) {
      const auto *pixel = src + x * bytes_pp;
      auto *dst = &px.rgb[(y * w + x) * 3];

      // bmp stores the channels in bgr(a) order.
      dst[0] = pixel[2];
      dst[1] = pixel[1];
      dst[2] = pixel[0];
    }
  }

  return px;
}

std::size_t skip_space(const std::vector<Byte> &bytes, std::size_t pos)
{
  while (pos < bytes.size()) {
    if (bytes[pos] == '#') {
      while (pos < bytes.size() && bytes[pos] != '\n' && bytes[pos] != '\r') {
        ++pos;
      }
    } else if (bytes[pos] == ' ' || bytes[pos] == '\t' || bytes[pos] == '\n' ||
               bytes[pos] == '\r' || bytes[pos] == '\v' || bytes[pos] == '\f') {
      ++pos;
    } else {
      break;
    }
  }

  return pos;
}

std::uint32_t read_header_number(const std::vector<Byte> &bytes,
                                 std::size_t &pos,
                                 const char *what)
{
  pos = skip_space(bytes, pos);

  if (pos >= bytes.size() || bytes[pos] < '0' || bytes[pos] > '9') {
    throw std::runtime_error(std::string("picture: invalid PPM header (") + what +
                             " expected)");
  }

  std::uint32_t value = 0;

  while (pos < bytes.size() && bytes[pos] >= '0' && bytes[pos] <= '9') {
    value = value * 10 + static_cast<std::uint32_t>(bytes[pos] - '0');
    if (value > MaxHeaderNumber) {
      throw std::runtime_error("picture: PPM header value is too large");
    }
    ++pos;
  }

  return value;
}

Pixels decode_ppm(const std::vector<Byte> &bytes)
{
  std::size_t pos = 2; // past the "P6" magic

  const auto width = read_header_number(bytes, pos, "width");
  const auto height = read_header_number(bytes, pos, "height");
  const auto maxval = read_header_number(bytes, pos, "maximum value");

  if (width == 0 || height == 0) {
    throw std::runtime_error("picture: empty PPM picture");
  }
  if (maxval != 255) {
    throw std::runtime_error("picture: only 8 bit PPM pictures are supported");
  }

  // exactly one whitespace byte separates the header from the pixel data.
  if (pos >= bytes.size()) {
    throw std::runtime_error("picture: truncated PPM header");
  }
  ++pos;

  const auto nelem =
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3;

  if (bytes.size() - pos < nelem) {
    throw std::runtime_error("picture: truncated PPM pixel data");
  }

  Pixels px;
  px.width = width;
  px.height = height;
  px.rgb.assign(bytes.begin() + static_cast<std::ptrdiff_t>(pos),
                bytes.begin() + static_cast<std::ptrdiff_t>(pos + nelem));

  return px;
}

Pixels decode(const std::vector<Byte> &bytes)
{
  if (bytes.size() >= 2 && bytes[0] == 'B' && bytes[1] == 'M') {
    return decode_bmp(bytes);
  }

  if (bytes.size() >= 2 && bytes[0] == 'P' && bytes[1] == '6') {
    return decode_ppm(bytes);
  }

  throw std::runtime_error(
      "picture: unsupported picture format, expected BMP or PPM (P6)");
}

} // namespace

template <typename T>
Channels<T> load(const std::filesystem::path &path)
{
  const auto pixels = decode(read_file(path));

  Channels<T> res{
      .red = Matrix2x<T>(pixels.height, pixels.width),
      .green = Matrix2x<T>(pixels.height, pixels.width),
      .blue = Matrix2x<T>(pixels.height, pixels.width),
  };

  for (std::size_t i = 0; i < pixels.height; ++i) {
    for (std::size_t j = 0; j < pixels.width; ++j) {
      const auto *pixel = &pixels.rgb[(i * pixels.width + j) * 3];

      res.red(i, j) = static_cast<T>(pixel[0]);
      res.green(i, j) = static_cast<T>(pixel[1]);
      res.blue(i, j) = static_cast<T>(pixel[2]);
    }
  }

  return res;
}

template Channels<unsigned char> load(const std::filesystem::path &);
template Channels<float> load(const std::filesystem::path &);
template Channels<double> load(const std::filesystem::path &);
template Channels<mfloat> load(const std::filesystem::path &);
template Channels<mdouble> load(const std::filesystem::path &);

} // namespace mutils::picture
