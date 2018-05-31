#pragma once
#include <vector>
#include <algorithm>
#include "type.hh"

namespace yonn
{

template <class T>
T* reverse_endian(T* p)
{
    std::reverse(
        reinterpret_cast<char*>(p),
        reinterpret_cast<char*>(p) + sizeof(T)
    );
    return p;
}

inline auto is_little_endian()
{
    auto x = 1;
    return *reinterpret_cast<char*>(&x) != 0;
}

inline auto std_input_types(bool has_bias = true) -> std::vector<data_type>
{
    if (has_bias)
        return {data_type::data, data_type::weight, data_type::bias};
    return {data_type::data, data_type::weight};
}

namespace compute
{

// TODO this is a bad design
template <class Iter1, class Iter2>
auto dot(Iter1 it1, Iter2 it2, size_t n) -> value_type
{
    value_type res{0};
    for (size_t i{0}; i < n; i++, ++it1, ++it2)
        res += *it1 * *it2;
    return res;
}

} // namespace compute

} // namespace yonn

