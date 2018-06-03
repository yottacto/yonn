#pragma once
#include <iterator>
#include <algorithm>
#include <vector>
#include <cmath>
#include <random>
#include "type.hh"
#include "tensor.hh"

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

template <class Vec>
auto max_index(Vec const& vec) -> size_t
{
    using std::begin;
    using std::end;
    return std::max_element(begin(vec), end(vec)) - begin(vec);
}

template <class T>
auto sqr(T x)
{
    return x * x;
}

inline auto in_length(
    size_t in_length,
    size_t window_size,
    padding pad_type
)
{
    return pad_type == padding::same
        ? in_length + window_size - 1
        : in_length;
}

inline auto out_length(
    size_t in_length,
    size_t window_size,
    size_t stride,
    padding pad_type
)
{
    size_t out_length{0};
    if (pad_type == padding::same)
        out_length = in_length;
    else if (pad_type == padding::valid)
        out_length = in_length - window_size + 1;

    // TODO the result must be integer
    return (out_length + stride - 1) / stride;
}

// TODO init_weight
inline void init_weight(vec_t& a, size_t fan_in, size_t fan_out)
{
    value_type weight_base = std::sqrt(6. / (fan_in + fan_out));

    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::uniform_int_distribution<> dis(-weight_base, weight_base);
    std::generate(std::begin(a), std::end(a), [&]() { return dis(gen); });
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

template <class T>
void add(T src, size_t size, T* dst)
{
    for (size_t i{0}; i < size; i++)
        dst[i] += src;
}

} // namespace compute

} // namespace yonn

