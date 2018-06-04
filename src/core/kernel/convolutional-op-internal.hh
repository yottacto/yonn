#pragma once
#include <algorithm>
#include <iterator>
#include <numeric>
#include "tensor.hh"
#include "util/util.hh"
#include "core/parameter/conv-parameter.hh"

namespace yonn
{
namespace core
{
namespace kernel
{

inline void convolutional_op_internal(
    tensor const& in_data,
    vec_t const& w,
    vec_t const& bias,
    tensor& out_data,
    conv_parameter const& params
)
{
    auto out_area = params.out.area();
    auto in_w     = params.in_padded.width;
    auto in_d     = params.in_padded.depth;
    auto out_w    = params.out.width;
    auto out_h    = params.out.height;
    auto out_d    = params.out.depth;
    auto w_w      = params.weight.width;
    auto w_h      = params.weight.height;
    auto w_s      = params.w_stride;
    auto line_s   = in_w * params.h_stride;

    // TODO parallelize
    #if USE_OPENMP
    #pragma omp for
    #endif
    for (size_t sample = 0; sample < in_data.size(); sample++) {
        auto const& in = in_data[sample];
        auto& out      = out_data[sample];

        for (size_t i{0}; i < out_d; i++) {
            auto* pa = &out[params.out.get_index(0, 0, i)];
            for (size_t j{0}; j < in_d; j++) {
                if (!params.tb.is_connected(i, j))
                    continue;

                auto idx = params.weight.get_index(0, 0, in_d * i + j);
                auto const* pw = &w[idx];
                idx = params.in_padded.get_index(0, 0, j);
                auto const* pin = &in[idx];
                auto* pout = pa;

                for (size_t y{0}; y < out_h; y++) {
                    auto const* pin_line = pin;

                    for (size_t x{0}; x < out_w; x++) {
                        auto const* pin_element = pin_line;
                        auto const* pw_element = pw;
                        value_type sum{0};

                        for (size_t yi{0}; yi < w_h; yi++) {
                            for (size_t xi{0}; xi < w_w; xi++) {
                                sum += pw_element[xi] * pin_element[xi];
                            }
                            pw_element  += w_w;
                            pin_element += in_w;
                        }
                        pout[x] = sum;
                        pin_line += w_s;
                    }
                    pout += out_w;
                    pin += line_s;
                }
            }
            if (params.has_bias)
                compute::add(bias[i], out_area, pa);
        }
    }
}

// TODO change dw, db type to vec_t
inline void convolutional_op_internal(
    tensor const& in_data,
    vec_t const& w,
    tensor& dw,
    tensor& db,
    tensor const& dout,
    tensor& dx,
    conv_parameter const& params
)
{
    // TODO clear grads or just assign the newvalue
    // TODO parallelize

    std::fill(std::begin(dw[0]), std::end(dw[0]), 0);
    std::fill(std::begin(db[0]), std::end(db[0]), 0);

    #if USE_OPENMP
    #pragma omp for
    #endif
    for (size_t sample = 0; sample < in_data.size(); sample++) {
        std::fill(std::begin(dx[sample]), std::end(dx[sample]), 0);

        for (size_t i{0}; i < params.in.depth; i++)
        for (size_t j{0}; j < params.out.depth; j++) {
            if (!params.tb.is_connected(j, i))
                continue;

            auto idx       = params.in.depth * j + i;
            idx            = params.weight.get_index(0, 0, idx);
            auto const* pw = &w[idx];

            idx                    = params.out.get_index(0, 0, j);
            auto const* pdelta_src = &dout[sample][idx];

            idx = params.in_padded.get_index(0, 0, i);
            auto* pdelta_dst = &dx[sample][idx];

            for (size_t y{0}; y < params.out.height; y++) {
                for (size_t x{0}; x < params.out.width; x++) {
                    value_type const* ppw = pw;

                    idx = y * params.out.width + x;
                    auto const ppdelta_src = pdelta_src[idx];

                    auto* ppdelta_dst = pdelta_dst
                        + y * params.h_stride * params.in_padded.width
                        + x * params.w_stride;

                    for (size_t wy{0}; wy < params.weight.height; wy++) {
                        for (size_t wx{0}; wx < params.weight.width; wx++) {
                            idx = wy * params.in_padded.width + wx;
                            ppdelta_dst[idx] += *ppw++ * ppdelta_src;
                        }
                    }
                }
            }
        }


        // accumulate dw
        for (size_t i{0}; i < params.in.depth; i++)
        for (size_t j{0}; j < params.out.depth; j++) {
            if (!params.tb.is_connected(j, i))
                continue;

            for (size_t yi{0}; yi < params.weight.height; yi++)
            for (size_t xi{0}; xi < params.weight.width; xi++) {
                value_type dst{0};

                auto idx = params.in_padded.get_index(xi, yi, i);
                auto const* prevo = &in_data[sample][idx];

                idx               = params.out.get_index(0, 0, j);
                auto const* delta = &dout[sample][idx];

                if (params.w_stride > 1) {
                    for (size_t y{0}; y < params.out.height; y++) {
                        size_t prevo_idx =
                            y * params.in_padded.width * params.h_stride;
                        size_t delta_idx = y * params.out.width;

                        for (size_t x{0}; x < params.out.width; x++) {
                            dst += prevo[prevo_idx + x * params.w_stride] *
                                delta[delta_idx + x];
                        }
                    }
                } else {
                    for (size_t y{0}; y < params.out.height; y++) {
                        dst += compute::dot(
                            prevo + y * params.in_padded.width * params.h_stride,
                            delta + y * params.out.width, params.out.width
                        );
                    }
                }

                idx = params.in.depth * j + i;
                dw[0][params.weight.get_index(xi, yi, idx)] += dst;
            }
        }

        // accumulate db
        if (params.has_bias) {
            for (size_t j{0}; j < params.out.depth; j++) {
                auto idx = params.out.get_index(0, 0, j);
                auto const* delta  = &dout[sample][idx];
                auto const* deltaa = delta + params.out.width * params.out.height;
                db[0][j] += std::accumulate(delta, deltaa, value_type{0});
            }
        }
    }
}

} // namespace kernel
} // namespace coer
} // namespace yonn

