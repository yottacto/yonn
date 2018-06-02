#pragma once
#include "type.hh"
#include "core/parameter/parameter.hh"

namespace yonn
{
namespace core
{

struct avg_pool_parameter : parameter
{
    avg_pool_parameter(
        shape3d_t const& in,
        size_t pool_width,
        size_t pool_height,
        size_t stride
    ) : in{in}, pool_width{pool_width}, pool_height{pool_height}, stride{stride}
    {
        out = {
            1 + (in.width - pool_width) / stride,
            1 + (in.height - pool_height) / stride,
            in.depth
        };
    }

    shape3d_t in;
    shape3d_t out;
    size_t pool_width;
    size_t pool_height;
    size_t stride;
};

} // namespace core
} // namespace yonn

