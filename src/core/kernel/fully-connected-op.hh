#pragma once
#include "type.hh"
#include "core/framework/op-kernel.hh"
#include "core/backend.hh"

namespace yonn
{
namespace core
{

struct fully_connected_op : framework::op_kernel
{
    fully_connected_op(
        size_t in_dim,
        size_t out_dim,
        bool has_bias = true,
        backend backend = default_engine()
    )
    {
    }

private:
};

} // namespace core
} // namespace yonn

