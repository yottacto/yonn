#pragma once
#include "type.hh"
#include "core/framework/op-kernel.hh"
#include "core/backend.hh"

namespace yonn
{
namespace core
{
namespace kernel
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

    void compute(framework::op_kernel_context& context)
    {
        auto const engine = context.engine();

        if (engine == core::backend::internal) {
            tensor const& in_data = context.input(0);
            tensor const& w = context.input(1);
            // TODO params to specify has_bias, using pointer and nullptr
            tensor const& bias = context.input(2);
            tensor& out_data = context.output(0);

            fully_connected_op_internal(
                in_data, w[0], bias[0], out_data
            );
        } else if (engine == core::backend::opencl) {
        } else {
            // TODO not support backend engine
        }
    }

private:
};

} // namespace kernel
} // namespace core
} // namespace yonn

