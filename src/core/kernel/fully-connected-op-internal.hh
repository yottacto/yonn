#pragma once
#include "tensor.hh"

namespace yonn
{
namespace core
{
namespace kernel
{

inline void fully_connected_op_internal(
    tensor const& in_data,
    vec_t const& w,
    vec_t const& bias,
    tensor& out_data
)
{
    // TODO parallelize
    for (size_t sample{0}; sample < in_data.size(); sample++) {
        // TODO
    }
}


} // namespace kernel
} // namespace coer
} // namespace yonn

