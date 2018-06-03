#pragma once
#include <vector>
#include <string>
#include <fstream>
#include "type.hh"
#include "tensor.hh"
#include "util/util.hh"

namespace yonn
{
namespace io
{
// http://yann.lecun.com/exdb/mnist/
namespace mnist
{
namespace detail
{

struct header
{
    uint32_t magic_number;
    uint32_t nitems;
    uint32_t nrows;
    uint32_t ncols;
};

inline auto& operator>>(std::ifstream& ifs, header& h)
{
    ifs.read(reinterpret_cast<char*>(&h.magic_number), 4);
    ifs.read(reinterpret_cast<char*>(&h.nitems), 4);
    ifs.read(reinterpret_cast<char*>(&h.nrows), 4);
    ifs.read(reinterpret_cast<char*>(&h.ncols), 4);

    if (is_little_endian()) {
        reverse_endian(&h.magic_number);
        reverse_endian(&h.nitems);
        reverse_endian(&h.nrows);
        reverse_endian(&h.ncols);
    }

    // TODO throw;
    if (h.magic_number != 0x00000803 || h.nitems <= 0)
        throw;
    // TODO throw;
    if (ifs.fail() || ifs.bad())
        throw;
    return ifs;
}

inline void parse_image(
    std::ifstream& ifs,
    header const& h,
    value_type scale_min,
    value_type scale_max,
    size_t x_padding,
    size_t y_padding,
    vec_t& dst
)
{
    auto const width  = h.ncols + 2 * x_padding;
    auto const height = h.nrows + 2 * y_padding;
    std::vector<uint8_t> image_vec(h.nrows * h.ncols);
    ifs.read(reinterpret_cast<char*>(image_vec.data()), h.nrows * h.ncols);
    dst.resize(width * height, scale_min);

    for (uint32_t y{0}; y < h.nrows; y++)
        for (uint32_t x{0}; x < h.ncols; x++)
            dst[width * (y + y_padding) + x + x_padding]
                = (image_vec[y * h.ncols + x] / value_type(255))
                    * (scale_max - scale_min) + scale_min;
}

} // namespace detail

inline void parse_images(
    std::string const& path,
    tensor& images,
    value_type scale_min,
    value_type scale_max,
    size_t x_padding,
    size_t y_padding
)
{
    // TODO same error case
    if (x_padding < 0 || y_padding < 0)
        throw;
    if (scale_max < scale_min)
        throw;

    std::ifstream ifs{path, std::ios::in | std::ios::binary};

    // TODO error open file
    if (ifs.bad() || ifs.fail())
        throw;

    detail::header h;
    ifs >> h;

    images.resize(h.nitems);
    for (uint32_t i{0}; i < h.nitems; i++) {
        detail::parse_image(
            ifs, h, scale_min, scale_max, x_padding, y_padding,
            images[i]
        );
    }
}

inline void parse_labels(
    std::string const& path,
    std::vector<label_t>& labels
)
{
    std::ifstream ifs{path, std::ios::in | std::ios::binary};

    // TODO error open file
    if (ifs.bad() || ifs.fail())
        throw;

    uint32_t magic_number, nitems;
    ifs.read(reinterpret_cast<char*>(&magic_number), 4);
    ifs.read(reinterpret_cast<char*>(&nitems), 4);

    // mnist data is big-endian format
    if (is_little_endian()) {
        reverse_endian(&magic_number);
        reverse_endian(&nitems);
    }

    // TODO throw
    if (magic_number != 0x00000801 || nitems <= 0)
        throw;

    labels.resize(nitems);
    for (uint32_t i{0}; i < nitems; i++) {
        uint8_t label;
        ifs.read(reinterpret_cast<char *>(&label), 1);
        labels[i] = static_cast<label_t>(label);
    }
}

} // namespace mnist
} // namespace io
} // namespace yonn

