#pragma once
#include <string>
#include <variant>
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
    average_pooling_op(avg_pool_parameter const& params, std::string const& name)
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
    std::string name;
};

struct average_pooling_grad_op : framework::op_kernel
{
    average_pooling_grad_op(avg_pool_parameter const& params, std::string const& name)
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
    std::string name;
};

} // namespace kernel
} // namespace core
} // namespace yonn

