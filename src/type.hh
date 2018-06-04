#pragma once
#include <iomanip>
#include <algorithm>
#include <iterator>
#include <vector>
#include <set>
#include <map>
#include <unordered_map>

namespace yonn
{

using value_type = double;
using vec_t = std::vector<value_type>;

auto inline operator/(vec_t const& lhs, value_type rhs)
{
    auto res = lhs;
    std::transform(std::begin(res), std::end(res), std::begin(res),
        [&](auto x) {
            return std::divides<value_type>{}(x, rhs);
        });
    return res;
}

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

    template <class OStream>
    void print_accuracy(OStream& os) const
    {
        os << "accuracy: " << std::fixed << std::setprecision(2)
            << accuracy()
            << " (" << success << "/" << total << ")\n";
    }

    template <class OStream>
    void print_detail(OStream& os) const
    {
        print_accuracy(os);
        auto all = labels();
        os << std::setw(6) << '*';
        for (auto i : all)
            os << std::setw(6) << i;
        os << "\n";

        for (auto i : all) {
            os << std::setw(6) << i;
            for (auto const& j : all) {
                size_t count{0};
                if (confusion_matrix.count(i) && confusion_matrix.at(i).count(j))
                    count = confusion_matrix.at(i).at(j);
                os << std::setw(6) << count;
            }
            os << "\n";
        }
    }

    auto labels() const -> std::set<label_t>
    {
        std::set<label_t> all;
        for (auto const& i : confusion_matrix) {
            all.insert(i.first);
            for (auto const& j : i.second)
                all.insert(j.first);
        }
        return all;
    }

    size_t success{0};
    size_t total{0};
    std::map<
        label_t,
        std::unordered_map<label_t, size_t>
    > confusion_matrix;
};

} // namespace yonn

