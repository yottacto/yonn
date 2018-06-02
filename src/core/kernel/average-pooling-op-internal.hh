#pragma once
#include <iterator>
#include <algorithm>
#include "tensor.hh"
#include "util.hh"
#include "core/parameter/avg-pool-parameter.hh"

namespace yonn
{
namespace core
{
namespace kernel
{

inline void average_pooling_op_internal(
    tensor const& in_data,
    tensor& out_data,
    avg_pool_parameter const& params
)
{
    // TODO parallelize
    for (size_t sample{0}; sample < in_data.size(); sample++) {
        auto const& in = in_data[sample];
        auto & out = out_data[sample];
        for (size_t od{0}; od < params.out.depth; od++)
        for (size_t ox{0}; ox < params.out.width; ox++)
        for (size_t oy{0}; oy < params.out.height; oy++) {
            auto& avg = out[params.out.get_index(ox, oy, od)];
            avg = 0;
            auto x_remain = std::min(
                params.pool_width,
                params.in.width - ox * params.stride
            );
            auto y_remain = std::min(
                params.pool_height,
                params.in.height - oy * params.stride
            );
            for (size_t ix{0}; ix < x_remain; ix++)
            for (size_t iy{0}; iy < y_remain; iy++) {
                avg += in[params.in.get_index(
                    ox * params.stride + ix,
                    oy * params.stride + iy,
                    od
                )];
            }
            avg /= params.pool_height * params.pool_width;
        }
    }
}

inline void average_pooling_op_internal(
    tensor const& in_data,
    tensor const& dout,
    tensor& dx,
    avg_pool_parameter const& params
)
{
    // TODO parallelize
    for (size_t sample{0}; sample < in_data.size(); sample++) {
        auto const& dout_sample = dout[sample];
        auto& dx_sample = dx[sample];
        std::fill(std::begin(dx_sample), std::end(dx_sample), 0);
        for (size_t od{0}; od < params.out.depth; od++)
        for (size_t ox{0}; ox < params.out.width; ox++)
        for (size_t oy{0}; oy < params.out.height; oy++) {
            auto delta = dout_sample[params.out.get_index(ox, oy, od)];
            delta /= params.pool_width * params.pool_height;
            auto x_remain = std::min(
                params.pool_width,
                params.in.width - ox * params.stride
            );
            auto y_remain = std::min(
                params.pool_height,
                params.in.height - oy * params.stride
            );
            for (size_t ix{0}; ix < x_remain; ix++)
            for (size_t iy{0}; iy < y_remain; iy++) {
                dx_sample[params.in.get_index(
                    ox * params.stride + ix,
                    oy * params.stride + iy,
                    od
                )] += delta;
            }
        }
    }
}

} // namespace kernel
} // namespace coer
} // namespace yonn

