#pragma once
#include <iterator>
#include <algorithm>
#include "tensor.hh"
#include "util/util.hh"
#include "core/parameter/avg-pool-parameter.hh"

namespace yonn
{
namespace core
{
namespace kernel
{

inline void average_pooling_op_internal(
    tensor const& in_data,
    vec_t const& w,
    vec_t const& bias,
    tensor& out_data,
    avg_pool_parameter const& params
)
{
    // TODO parallelize
    #if USE_OPENMP
    #pragma omp for
    #endif
    for (size_t sample = 0; sample < in_data.size(); sample++) {
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
            avg *= w[od];
            avg /= params.pool_height * params.pool_width;
            avg += bias[od];
        }
    }
}

// TODO change dw, db type to vec_t
inline void average_pooling_op_internal(
    tensor const& in_data,
    tensor const& w,
    tensor& dw,
    tensor& db,
    tensor const& dout,
    tensor& dx,
    avg_pool_parameter const& params
)
{
    std::fill(std::begin(dw[0]), std::end(dw[0]), 0);
    std::fill(std::begin(db[0]), std::end(db[0]), 0);
    auto const& area = params.pool_width * params.pool_height;

    // TODO parallelize
    #if USE_OPENMP
    #pragma omp for
    #endif
    for (size_t sample = 0; sample < in_data.size(); sample++) {
        auto const& in          = in_data[sample];
        auto const& dout_sample = dout[sample];
        auto& dx_sample         = dx[sample];

        std::fill(std::begin(dx_sample), std::end(dx_sample), 0);
        for (size_t od{0}; od < params.out.depth; od++)
        for (size_t ox{0}; ox < params.out.width; ox++)
        for (size_t oy{0}; oy < params.out.height; oy++) {
            auto delta = dout_sample[params.out.get_index(ox, oy, od)];
            auto deltax = delta * w[0][od] / area;
            value_type sum{0};
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
                )] += deltax;
                sum += in[params.in.get_index(
                    ox * params.stride + ix,
                    oy * params.stride + iy,
                    od
                )];
            }

            db[0][od] += delta;
            dw[0][od] += delta * sum / area;
        }
    }
}

} // namespace kernel
} // namespace coer
} // namespace yonn

