#pragma once
#include "type.hh"
#include "core/parameter/parameter.hh"

namespace yonn
{
namespace core
{

struct fully_parameter : parameter
{
    size_t in_size;
    size_t out_size;
    bool has_bias;
};

} // namespace core
} // namespace yonn

