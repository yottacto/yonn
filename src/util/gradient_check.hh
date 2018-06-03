#pragma once
#include <vector>
#include <cmath>
#include "tensor.hh"
#include "util.hh"

namespace yonn
{

inline auto sum(tensor const& t) -> value_type
{
    value_type s{0};
    for (auto const& v : t)
        for (auto i : v)
            s += i;
    return s;
}

inline auto relative_error(tensor const& std, tensor const& num)
{
    value_type max_diff{0};
    value_type max{1e-8};
    for (auto vi = 0u; vi < std.size(); vi++)
    for (auto i  = 0u; i < std[vi].size(); i++) {
        max_diff = std::max(
            max_diff,
            std::abs(std[vi][i] - num[vi][i])
        );
        max = std::max(
            max,
            std::abs(std[vi][i]) + std::abs(num[vi][i])
        );
    }
    return max_diff / max;
}

template <class Layer>
inline auto gradient_check(
    Layer& l, std::vector<tensor> in, tensor const& dout,
    value_type h = 1e-6
)
{
    l.set_input_data(in);
    l.forward();
    // auto std_out = sum(l.get_output_data());

    l.set_output_grad(dout);
    l.backward();
    auto std_grad = l.get_input_grad();
    auto numeric_grad = std_grad;

    for (auto ti = 0u; ti < in.size(); ti++)
    for (auto vi = 0u; vi < in[ti].size(); vi++)
    for (auto i  = 0u; i < in[ti][vi].size(); i++) {
        auto& v = in[ti][vi][i];
        auto old = v;
        v = old + h;
        l.set_input_data(in);
        l.forward();
        auto pos_out = l.get_output_data();

        v = old - h;
        l.set_input_data(in);
        l.forward();
        auto neg_out = l.get_output_data();

        auto res = ((pos_out - neg_out) / (2. * h));
        value_type g = 0;
        for (auto i = 0u; i < dout.size(); i++)
            g += compute::dot(dout[i].begin(), res[i].begin(), dout[i].size());

        numeric_grad[ti][vi][i] = g;

        v = old;
    }

    auto error = relative_error(std_grad[0], numeric_grad[0]);
    for (auto i = 1u; i < std_grad.size(); i++)
        error = std::max(error, relative_error(std_grad[i], numeric_grad[i]));
    return error;
    // return numeric_grad;
}

} // namespace yonn

