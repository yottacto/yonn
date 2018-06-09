#pragma once
#include <string>
#include <variant>
#include "type.hh"
#include "core/backend.hh"
#include "core/framework/op-kernel.hh"
#include "core/parameter/fully-parameter.hh"
#include "core/kernel/fully-connected-op-internal.hh"

namespace yonn
{
namespace core
{
namespace kernel
{

struct fully_connected_op : framework::op_kernel
{
    fully_connected_op(fully_parameter const& params, std::string const& name)
        : params{params}, name{name}
    {
    }

    void compute(framework::op_kernel_context& context, core::engine::engine_type& eng, bool united_backend) override
    {
        ignore(eng);
        ignore(united_backend);

        auto const engine = context.engine();

        if (engine == core::backend_type::internal) {
            using data_type = tensor;
            data_type const& in_data = *std::get<data_type*>(context.input(0));
            data_type const& w       = *std::get<data_type*>(context.input(1));
            // TODO params to specify has_bias, using pointer and nullptr
            data_type const& bias    = *std::get<data_type*>(context.input(2));
            data_type& out_data      = *std::get<data_type*>(context.output(0));

            fully_connected_op_internal(
                in_data, w[0], bias[0], out_data, params
            );
        } else if (engine == core::backend_type::opencl) {
        } else {
            // TODO not support backend engine
        }
    }

private:
    fully_parameter params;
    std::string name;
};

struct fully_connected_grad_op : framework::op_kernel
{
    fully_connected_grad_op(fully_parameter const& params, std::string const& name)
        : params{params}, name{name}
    {
    }

    void compute(framework::op_kernel_context& context, core::engine::engine_type& eng, bool united_backend) override
    {
        ignore(eng);
        ignore(united_backend);

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

            fully_connected_op_internal(
                in_data, w[0], dw, db, dout, dx, params
            );
        } else if (engine == core::backend_type::opencl) {
        } else {
            // TODO not support backend engine
        }
    }

private:
    fully_parameter params;
    std::string name;
};

} // namespace kernel
} // namespace core
} // namespace yonn

