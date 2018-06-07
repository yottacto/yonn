#pragma once
#include <variant>
#include <string>
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
    convolutional_op(conv_parameter const& params, std::string const& name)
        : params{params}, name{name}
    {
    }

    void compute(framework::op_kernel_context& context, core::engine::engine_type& eng) override
    {
        auto const engine = context.engine();

        if (engine == core::backend_type::internal) {
            using data_type = tensor;
            data_type const& in_data = *std::get<data_type*>(context.input(0));
            data_type const& w       = *std::get<data_type*>(context.input(1));
            // FIXME params to specify has_bias, using pointer and nullptr
            data_type const& bias    = *std::get<data_type*>(context.input(2));
            data_type& out_data      = *std::get<data_type*>(context.output(0));

            convolutional_op_internal(
                in_data, w[0], bias[0], out_data, params
            );
        } else if (engine == core::backend_type::opencl) {
            using data_type = cl::Buffer;
            data_type& in_data  = *std::get<data_type*>(context.input(0));
            data_type& w        = *std::get<data_type*>(context.input(1));
            data_type& bias     = *std::get<data_type*>(context.input(2));
            data_type& out_data = *std::get<data_type*>(context.output(0));

            auto& e = std::get<core::engine::opencl>(eng);
            auto const& eargs = e.forward_eargs.at(name);
            auto const& kernel = e.forward_kernels.at(name);

            // TODO
            // kernel(eargs, )
        } else {
            // TODO not support backend engine
        }
    }

// TODO uncomment
// private:
    conv_parameter params;
    std::string name;
};

struct convolutional_grad_op : framework::op_kernel
{
    convolutional_grad_op(conv_parameter const& params, std::string const& name)
        : params{params}, name{name}
    {
    }

    void compute(framework::op_kernel_context& context, core::engine::engine_type& eng) override
    {
        auto const engine = context.engine();

        if (engine == core::backend_type::internal) {
            using data_type = tensor;
            data_type const& in_data = *std::get<data_type*>(context.input(0));
            data_type const& w       = *std::get<data_type*>(context.input(1));
            data_type& dw            = *std::get<data_type*>(context.input_grad(1));
            // FIXME params to specify has_bias, using pointer and nullptr
            data_type& db            = *std::get<data_type*>(context.input_grad(2));
            data_type& dx            = *std::get<data_type*>(context.input_grad(0));
            data_type& dout          = *std::get<data_type*>(context.output_grad(0));

            convolutional_op_internal(
                in_data, w[0], dw, db, dout, dx, params
            );
        } else if (engine == core::backend_type::opencl) {
            using data_type = cl::Buffer;
        } else {
            // TODO not support backend engine
        }
    }

// TODO uncomment
// private:
    conv_parameter params;
    std::string name;
};

} // namespace kernel
} // namespace core
} // namespace yonn

