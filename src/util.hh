#pragma once
#include <vector>
#include "type.hh"

namespace yonn
{

inline auto std_input_type(bool has_bias = true) -> std::vector<data_type>
{
    if (has_bias)
        return {data_type::data, data_type::weight, data_type::bias};
    return {data_type::data, data_type::weight};
}

} // namespace yonn

