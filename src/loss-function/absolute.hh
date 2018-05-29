#pragma once
#include <utility>
#include <cmath>
#include "type.hh"
#include "tensor.hh"

namespace yonn
{
namespace loss_function
{

struct absolute
{
    static auto f(vec_t const& scores, label_t y) -> value_type
    {
        // FIXME value range in (min, max), this depends on the last layer's
        // output value range, typically depends on activation functions.
        // refactor this
        std::pair<value_type, value_type> range(0, 1);
        // TODO assert score.size() == y.size()
        value_type loss{0};
        for (size_t i{0}; i < scores.size(); i++)
            loss += std::abs(scores[i] - (i == y ? range.second : range.first));
        return loss / static_cast<value_type>(scores.size());
    }

    static auto df(vec_t const& scores, label_t y) -> vec_t
    {
        // FIXME value range in (min, max), this depends on the last layer's
        // output value range, typically depends on activation functions.
        // refactor this
        std::pair<value_type, value_type> range(0, 1);
        // TODO assert score.size() == y.size()
        vec_t d(scores.size());
        auto factor = value_type{1} / static_cast<value_type>(scores.size());
        for (size_t i{0}; i < scores.size(); i++) {
            auto sign = scores[i] - (i == y ? range.second : range.first);
            if (sign == value_type{0})
                continue;
            if (sign < value_type{0})
                d[i] = -factor;
            else
                d[i] = +factor;
        }
        return d;
    }
};

} // namespace loss_function
} // namespace yonn

