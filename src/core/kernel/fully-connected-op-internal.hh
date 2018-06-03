#pragma once
#include <algorithm>
#include <iterator>
#include "tensor.hh"
#include "util/util.hh"
#include "core/parameter/fully-parameter.hh"

namespace yonn
{
namespace core
{
namespace kernel
{

inline void fully_connected_op_internal(
    tensor const& in_data,
    vec_t const& w,
    vec_t const& bias,
    tensor& out_data,
    fully_parameter const& params
)
{
    // TODO parallelize
    for (size_t sample{0}; sample < in_data.size(); sample++) {
        auto const& in = in_data[sample];
        auto& out      = out_data[sample];
        auto const& in_size = params.in_size;
        auto const& out_size = params.out_size;
        auto const& has_bias = params.has_bias;
        // FIXME params out_size and below in_size
        for (size_t i{0}; i < out_size; i++) {
            out[i] = 0;
            for (size_t c = 0; c < in_size; c++)
                out[i] += in[c] * w[c * out_size + i];

            // FIXME
            if (has_bias)
            {
                out[i] += bias[i];
            }
        }
    }
}

inline void fully_connected_op_internal(
    tensor const& in_data,
    vec_t const& w,
    tensor& dw,
    tensor& db,
    tensor const& dout,
    tensor& dx,
    fully_parameter const& params
)
{
    auto const& in_size = params.in_size;
    auto const& out_size = params.out_size;
    auto const& has_bias = params.has_bias;
    std::fill(std::begin(dw[0]), std::end(dw[0]), 0);
    std::fill(std::begin(db[0]), std::end(db[0]), 0);
    // TODO clear grads or just assign the newvalue
    // TODO parallelize
    for (size_t sample{0}; sample < in_data.size(); sample++) {
        // derivatives for input data, heere dx
        // FIXME params in_size and out_size
        for (size_t i{0}; i < in_size; i++)
            // TODO dot product and vectorization
            dx[sample][i] = compute::dot(
                std::begin(dout[sample]),
                std::next(std::begin(w), i * out_size),
                out_size
            );

        // derivatives for w, here dw
        for (size_t i{0}; i < in_size; i++)
            for (size_t j{0}; j < out_size; j++)
                dw[0][i * out_size + j] += in_data[sample][i] * dout[sample][j];

        // derivatives for bias, here db
        if (has_bias)
            for (size_t i{0}; i < out_size; i++)
                db[0][i] += dout[sample][i];
    }
}

} // namespace kernel
} // namespace coer
} // namespace yonn

