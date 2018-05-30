#pragma once
#include "type.hh"
#include "core/framework/op-kernel.hh"
#include "core/backend.hh"
#include "core/parameter/fully-parameter.hh"

namespace yonn
{
namespace core
{
namespace kernel
{

struct fully_connected_op : framework::op_kernel
{
    fully_connected_op(fully_parameter const& params)
        : params{params}
    {
    }

    void compute(framework::op_kernel_context& context) override
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
    fully_parameter params;
};

struct fully_connected_grad_op : framework::op_kernel
{
    fully_connected_grad_op(fully_parameter const& params)
        : params{params}
    {
    }

    void compute(framework::op_kernel_context& context) override
    {
        auto const engine = context.engine();

        if (engine == core::backend::internal) {
            tensor const& in_data = context.input(0);
            tensor const& w = context.input(1);
            tensor& dw = context.input_grad(1);
            // TODO params to specify has_bias, using pointer and nullptr
            tensor& db = context.input_grad(2);
            tensor& dx = context.input_grad(0);
            tensor& dout = context.output_grad(0);

            fully_connected_op_internal(
                in_data, w[0], dw, bias[0], out_data, dout, dx
            );
        } else if (engine == core::backend::opencl) {
        } else {
            // TODO not support backend engine
        }
    }

private:
    fully_parameter params;
};

} // namespace kernel
} // namespace core
} // namespace yonn

