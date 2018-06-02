#pragma once
#include "type.hh"

namespace yonn
{
namespace loss_function
{

// mean-squared-error loss function
struct mse
{
    static auto f(const vec_t &y, const vec_t &t) -> value_type
    {
        value_type d{0.0};
        for (size_t i{0}; i < y.size(); i++)
            d += (y[i] - t[i]) * (y[i] - t[i]);

        return d / static_cast<value_type>(y.size());
    }

    static auto df(const vec_t &y, const vec_t &t) -> vec_t
    {
        vec_t d(t.size());
        auto factor = value_type(2) / static_cast<value_type>(t.size());
        for (size_t i{0}; i < y.size(); i++)
            d[i] = factor * (y[i] - t[i]);
        return d;
    }
};

} // namespace yonn
} // namespace loss_function

