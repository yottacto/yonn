#pragma once
#include "type.hh"
#include "core/backend.hh"
#include "core/framework/op-kernel.hh"
#include "core/parameter/conv-parameter.hh"
#include "core/kernel/convolutional-op-internal.hh"

namespace yonn
{
namespace core
{
namespace kernel
{

struct convolutional_op : framework::op_kernel
{
    convolutional_op(conv_parameter const& params)
        : params{params}
    {
    }

    void compute(framework::op_kernel_context& context) override
    {
        auto const engine = context.engine();

        if (engine == core::backend::internal) {
            tensor const& in_data = context.input(0);
            tensor const& w       = context.input(1);
            // TODO params to specify has_bias, using pointer and nullptr
            tensor const& bias    = context.input(2);
            tensor& out_data      = context.output(0);

            convolutional_op_internal(
                in_data, w[0], bias[0], out_data, params
            );
        } else if (engine == core::backend::opencl) {
        } else {
            // TODO not support backend engine
        }
    }

// TODO uncomment
// private:
    conv_parameter params;
};

struct convolutional_grad_op : framework::op_kernel
{
    convolutional_grad_op(conv_parameter const& params)
        : params{params}
    {
    }

    void compute(framework::op_kernel_context& context) override
    {
        auto const engine = context.engine();

        if (engine == core::backend::internal) {
            tensor const& in_data = context.input(0);
            tensor const& w       = context.input(1);
            tensor& dw            = context.input_grad(1);
            // FIXME params to specify has_bias, using pointer and nullptr
            tensor& db            = context.input_grad(2);
            tensor& dx            = context.input_grad(0);
            tensor& dout          = context.output_grad(0);

            convolutional_op_internal(
                in_data, w[0], dw, db, dout, dx, params
            );
        } else if (engine == core::backend::opencl) {
        } else {
            // TODO not support backend engine
        }
    }

// TODO uncomment
// private:
    conv_parameter params;
};

} // namespace kernel
} // namespace core
} // namespace yonn

