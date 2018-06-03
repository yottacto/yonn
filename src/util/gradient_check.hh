#pragma once

#include <iostream>
#include <iomanip>

#include <vector>
#include <cmath>
#include "tensor.hh"
#include "util.hh"

namespace yonn
{

// TODO remove this
inline void print(tensor const& t)
{
    std::cout << "----\n";
    for (auto v : t) {
        for (auto i : v)
            std::cout << std::fixed << std::setprecision(12) << i << " ";
        std::cout << "\n";
    }
    std::cout << "----\n";
    std::cout << "\n";
}


inline auto sum(tensor const& t) -> value_type
{
    value_type s{0};
    for (auto const& v : t)
        for (auto i : v)
            s += i;
    return s;
}

inline auto relative_error(
    std::vector<tensor> const& std,
    std::vector<tensor> const& num
)
{
    value_type max_diff{0};
    value_type max{1e-8};
    for (auto ti = 0u; ti < std.size(); ti++)
    for (auto vi = 0u; vi < std[0].size(); vi++)
    for (auto i  = 0u; i < std[0][0].size(); i++) {
        max_diff = std::max(
            max_diff,
            std::abs(std[ti][vi][i] - num[ti][vi][i])
        );
        max = std::max(
            max,
            std::abs(std[ti][vi][i]) + std::abs(num[ti][vi][i])
        );
    }
    return max_diff / max;
}

template <class Layer>
inline auto gradient_check(
    Layer& l, std::vector<tensor> in, tensor const& dout,
    value_type h = 1e-2
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

    return relative_error(std_grad, numeric_grad);
    // return numeric_grad;
}

} // namespace yonn

