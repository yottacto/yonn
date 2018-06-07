#pragma once
#include "type.hh"
#include "core/backend.hh"
#include "core/framework/op-kernel.hh"
#include "core/parameter/avg-pool-parameter.hh"
#include "core/kernel/average-pooling-op-internal.hh"

namespace yonn
{
namespace core
{
namespace kernel
{

struct average_pooling_op : framework::op_kernel
{
    average_pooling_op(avg_pool_parameter const& params)
        : params{params}
    {
    }

    void compute(framework::op_kernel_context& context) override
    {
        auto const engine = context.engine();

        if (engine == core::backend_type::internal) {
            tensor const& in_data = context.input(0);
            tensor const& w       = context.input(1);
            tensor const& bias    = context.input(2);
            tensor& out_data      = context.output(0);

            average_pooling_op_internal(
                in_data, w[0], bias[0], out_data, params
            );
        } else if (engine == core::backend_type::opencl) {
        } else {
            // TODO not support backend engine
        }
    }

private:
    avg_pool_parameter params;
};

struct average_pooling_grad_op : framework::op_kernel
{
    average_pooling_grad_op(avg_pool_parameter const& params)
        : params{params}
    {
    }

    void compute(framework::op_kernel_context& context) override
    {
        auto const engine = context.engine();

        if (engine == core::backend_type::internal) {
            tensor const& in_data = context.input(0);
            tensor const& w       = context.input(1);
            tensor& dw            = context.input_grad(1);
            tensor& db            = context.input_grad(2);
            tensor& dx            = context.input_grad(0);
            tensor const& dout    = context.output_grad(0);

            average_pooling_op_internal(
                in_data, w, dw, db, dout, dx, params
            );
        } else if (engine == core::backend_type::opencl) {
        } else {
            // TODO not support backend engine
        }
    }

private:
    avg_pool_parameter params;
};

} // namespace kernel
} // namespace core
} // namespace yonn

