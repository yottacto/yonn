#pragma once

namespace yonn
{

using size_t = unsigned int;

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

