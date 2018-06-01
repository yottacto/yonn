#pragma once
#include <vector>
#include <unordered_map>

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

enum class padding
{
    valid,
    same,  // add zero-padding to keep the image size
};

template <class T>
struct shape3d
{
    using size_type = T;

    shape3d() = default;
    shape3d(size_type width, size_type height, size_type depth)
        : width{width}, height{height}, depth{depth} {}

    auto area() const -> size_t
    {
        return width * height;
    }

    auto size() const -> size_t
    {
        return area() * depth;
    }

    auto get_index(size_t x, size_t y, size_t z) const -> size_t
    {
        // TODO check x, y, z in range
        return (z * height + y) * width + x;
    }

    size_type width;
    size_type height;
    size_type depth;
};

using shape3d_t = shape3d<size_t>;

struct result
{
    auto accuracy() const
    {
        return success * 100. / total;
    }

    void insert(label_t predicted, label_t actual)
    {
        confusion_matrix[predicted][actual]++;
        total++;
        if (predicted == actual)
            success++;
    }

    size_t success{0};
    size_t total{0};
    std::unordered_map<
        label_t,
        std::unordered_map<label_t, size_t>
    > confusion_matrix;
};

} // namespace yonn

