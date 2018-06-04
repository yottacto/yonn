#pragma once
#include <cstddef>
#include <iomanip>

namespace yonn
{
namespace util
{

struct progress_display
{
    using size_type = std::size_t;

    explicit progress_display(size_type total, size_type len = 60, size_type count = 0)
        : total{total}, len{len}, count{count} {}

    void tick() { count++; }
    void tick(size_t c) { count += c; }

    template <class Stream>
    void display(Stream& os) const
    {
        os << "[";
        auto progress = static_cast<double>(count) / static_cast<double>(total);
        size_type pos = len * progress;
        os << std::string(pos, '#') << std::string(len - pos, '-') << "]";
        os << std::setw(4) << static_cast<size_type>(progress * 100.) << "%\r";
        os.flush();
    }

    size_type total;
    size_type len;
    size_type count;
};

} // namespace util
} // namespace yonn

