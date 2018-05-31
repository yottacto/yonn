#pragma once
#include <vector>

namespace yonn
{

using value_type = float;
using vec_t = std::vector<value_type>;

// TODO unsigned int or unsigned long
using size_t = unsigned int;

using label_t = size_t;

enum class data_type
{
    data,
    weight,
    bias,
    label,
};

template <class T>
struct shape3d
{
    using size_type = T;

    shape3d() = default;
    shape3d(size_type height, size_type weight, size_type depth)
        : height{height}, weight{weight}, depth{depth} {}

    auto area() const
    {
        return height * weight;
    }

    auto size() const
    {
        return area() * depth;
    }

    size_type height;
    size_type weight;
    size_type depth;
};

using shape3d_t = shape3d<size_t>;

} // namespace yonn

