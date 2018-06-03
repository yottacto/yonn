#pragma once
#include <vector>
#include "type.hh"

namespace yonn
{

// inner vector for h*w*depth, outer vector for sample (mini batch)
using tensor = std::vector<vec_t>;
// using tensor = std::vector<float>;

inline auto operator-(tensor const& lhs, tensor const& rhs) -> tensor
{
    tensor res = lhs;
    for (auto vi = 0u; vi < res.size(); vi++)
    for (auto i  = 0u; i < res[vi].size(); i++)
        res[vi][i] -= rhs[vi][i];
    return res;
}

inline auto operator/(tensor const& lhs, value_type x) -> tensor
{
    tensor res = lhs;
    for (auto& v : res)
    for (auto& i : v)
        i /= x;
    return res;
}

} // namespace yonn

