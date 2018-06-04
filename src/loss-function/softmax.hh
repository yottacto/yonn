#pragma once
#include <algorithm>
#include <iterator>
#include <numeric>
#include <utility>
#include <cmath>
#include "type.hh"

namespace yonn
{
namespace loss_function
{

struct softmax
{
    static auto f(vec_t scores, label_t y) -> value_type
    {
        using std::begin;
        using std::end;
        auto& s = scores;
        auto max = std::max_element(begin(s), end(s));
        std::transform(begin(s), end(s), begin(s), [&](auto x) {
            return x - *max;
        });
        auto z = std::accumulate(begin(s), end(s), value_type{0}, [](auto const& lhs, auto const& rhs) {
            return lhs + std::exp(rhs);
        });
        std::transform(begin(s), end(s), begin(s), [&](auto x) {
            return x - std::log(z);
        });
        return -s[y];
    }

    static auto df(vec_t scores, label_t y) -> vec_t
    {
        using std::begin;
        using std::end;
        auto& s = scores;
        auto max = std::max_element(begin(s), end(s));
        std::transform(begin(s), end(s), begin(s), [&](auto x) {
            return x - *max;
        });
        auto z = std::accumulate(begin(s), end(s), value_type{0}, [](auto const& lhs, auto const& rhs) {
            return lhs + std::exp(rhs);
        });
        std::transform(begin(s), end(s), begin(s), [&](auto x) {
            return std::exp(x - std::log(z));
        });
        s[y] -= 1;
        return s;
    }
};

} // namespace yonn
} // namespace loss_function

