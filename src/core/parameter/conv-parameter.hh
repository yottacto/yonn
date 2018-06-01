#pragma once
#include <vector>
#include "type.hh"
#include "util.hh"
#include "core/parameter/parameter.hh"

namespace yonn
{
namespace core
{

struct connection_table
{
    connection_table() = default;

    template <unsigned int N>
    connection_table(bool const arr[N], size_t rows, size_t cols)
    : rows{rows}, cols{cols}, connected(rows * cols)
    {
        for (auto i = 0u; i < connected.size(); i++)
            connected[i] = arr[i];
    }

    auto is_empty() const -> bool
    {
        return rows == 0 && cols == 0;
    }

    auto is_connected(size_t x, size_t y) const -> bool
    {
        return is_empty()
            ? true
            : connected[y * cols + x];
    }

    size_t rows{0};
    size_t cols{0};
    std::vector<char> connected;
};

struct conv_parameter : parameter
{
    conv_parameter(
        shape3d_t const& in,
        size_t w_width,
        size_t w_height,
        size_t out_channels,
        padding pad_type,
        bool has_bias,
        size_t w_stride,
        size_t h_stride,
        core::connection_table const& tb = core::connection_table()
    ) :
        in{in}, has_bias{has_bias}, pad_type{pad_type},
        w_stride{w_stride}, h_stride{h_stride},
        tb{tb}
    {
        in_padded = {
            in_length(in.width, w_width, pad_type),
            in_length(in.height, w_height, pad_type),
            in.depth
        };
        out = {
            out_length(in.width, w_width, w_stride, pad_type),
            out_length(in.height, w_height, h_stride, pad_type),
            out_channels
        };
        weight = {
            w_width, w_height, in.depth * out_channels
        };
    }

    shape3d_t in;
    shape3d_t in_padded;
    shape3d_t out;
    shape3d_t weight;
    bool has_bias;
    padding pad_type;
    size_t w_stride;
    size_t h_stride;
    connection_table tb;
};

} // namespace core
} // namespace yonn

