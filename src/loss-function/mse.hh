#pragma once
#include <utility>
#include <CL/cl.hpp>
#include "type.hh"
#include "util/util.hh"

namespace yonn
{
namespace loss_function
{

// mean-squared-error loss function
struct mse
{
    using kernel_type = cl::make_kernel<
        int, int,
        cl::Buffer&, cl::Buffer&, cl::Buffer&
    >;

    static auto name() { return "mse"; }

    static auto f(vec_t const& scores, label_t y) -> value_type
    {
        // FIXME value range in (min, max), this depends on the last layer's
        // output value range, typically depends on activation functions.
        // refactor this
        std::pair<value_type, value_type> range(0, 1);

        value_type d{0.0};
        for (size_t i{0}; i < scores.size(); i++)
            // d += sqr(scores[i] - (i == y ? range.second : range.first));
            d += (scores[i] - (i == y ? range.second : range.first))
                * (scores[i] - (i == y ? range.second : range.first));
            // d += sqr(scores[i] - (i == y ? range.second : range.first))
            //     / value_type{2.};

        return d / static_cast<value_type>(scores.size());
        // return d;
    }

    static auto df(const vec_t &scores, label_t y) -> vec_t
    {
        // FIXME value range in (min, max), this depends on the last layer's
        // output value range, typically depends on activation functions.
        // refactor this
        std::pair<value_type, value_type> range(0, 1);

        vec_t d(scores.size());
        auto factor = value_type{2} / static_cast<value_type>(scores.size());
        for (size_t i{0}; i < scores.size(); i++)
            d[i] = factor * (scores[i] - (i == y ? range.second : range.first));
            // d[i] = (scores[i] - (i == y ? range.second : range.first));
        return d;
    }
};

} // namespace yonn
} // namespace loss_function

