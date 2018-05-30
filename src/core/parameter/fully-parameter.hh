#pragma once
#include "type.hh"
#include "core/parameter/parameter.hh"

namespace yonn
{
namespace core
{

struct fully_parameter : parameter
{
    fully_parameter(size_t in_size, size_t out_size, bool has_bias)
        : in_size{in_size}, out_size{out_size}, has_bias{has_bias} {}
    size_t in_size;
    size_t out_size;
    bool has_bias;
};

} // namespace core
} // namespace yonn

