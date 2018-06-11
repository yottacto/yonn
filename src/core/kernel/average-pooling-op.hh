#pragma once
#include <iostream>

#include <string>
#include <variant>
#include "type.hh"
#include "core/backend.hh"
#include "core/framework/op-kernel.hh"
#include "core/parameter/avg-pool-parameter.hh"
#include "core/kernel/average-pooling-op-internal.hh"

#include "core/kernel/opencl/average-pooling.hh"

namespace yonn
{
namespace core
{
namespace kernel
{

struct average_pooling_op : framework::op_kernel
{
    using fk_type = cl::make_kernel<
        int, int, int, int, int, int, int, int, int,
        cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&
    >;

    average_pooling_op(avg_pool_parameter const& params, std::string const& name)
        : params{params}, name{name}
    {
    }

    void init_opencl_kernel(core::engine::opencl& eng)
    {
        if (!opencl_kernel_initialized) {
            sources.emplace_back(opencl_kernel::avg_pool_kernel_code.c_str(), opencl_kernel::avg_pool_kernel_code.size());
            program = cl::Program{eng.context, sources};
            if (program.build({eng.default_device}) != CL_SUCCESS) {
                // FIXME
                std::cerr << "Error building: "
                    << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(eng.default_device) << "\n";
                throw;
            }

            opencl_kernel_initialized = true;
        }
        fk = std::make_unique<fk_type>(program, "forward");
    }

    void init_opencl(core::engine::opencl& eng, size_t size, size_t sample_count)
    {
        fk_eargs = std::make_unique<cl::EnqueueArgs>(eng.queue, cl::NDRange(size));
        this->sample_count = sample_count;
    }

    void compute(framework::op_kernel_context& context, core::engine::engine_type& eng, bool united_backend = true) override
    {
        ignore(eng);
        ignore(united_backend);

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
            using data_type = cl::Buffer;
            data_type& in_data  = *std::get<data_type*>(context.input(0));
            data_type& w        = *std::get<data_type*>(context.input(1));
            data_type& bias     = *std::get<data_type*>(context.input(2));
            data_type& out_data = *std::get<data_type*>(context.output(0));

            (*fk)(*fk_eargs,
                params.in.width,
                params.in.height,
                params.in.depth,
                params.out.width,
                params.out.height,
                params.out.depth,
                params.pool_width,
                params.pool_height,
                params.stride,
                // params.has_bias,
                in_data, w, bias, out_data
            ).wait();
        } else {
            // TODO not support backend engine
        }
    }

private:
    avg_pool_parameter params;
    std::string name;

    size_t sample_count;

    bool opencl_kernel_initialized{false};
    std::unique_ptr<fk_type> fk;
    std::unique_ptr<cl::EnqueueArgs> fk_eargs;
    cl::Program::Sources sources;
    cl::Program program;
};

struct average_pooling_grad_op : framework::op_kernel
{
    using bk_dx_type = cl::make_kernel<
        int, int, int, int, int, int, int, int, int,
        cl::Buffer&, cl::Buffer&, cl::Buffer&
    >;

    using bk_dw_type = cl::make_kernel<
        int, int, int, int, int, int, int, int, int, int,
        cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::LocalSpaceArg
    >;

    using bk_db_type = cl::make_kernel<
        int, int, int, int,
        cl::Buffer&, cl::Buffer&
    >;

    average_pooling_grad_op(avg_pool_parameter const& params, std::string const& name)
        : params{params}, name{name}
    {
    }

    void init_opencl_kernel(core::engine::opencl& eng)
    {
        if (!opencl_kernel_initialized) {
            sources.emplace_back(opencl_kernel::avg_pool_kernel_code.c_str(), opencl_kernel::avg_pool_kernel_code.size());
            program = cl::Program{eng.context, sources};
            if (program.build({eng.default_device}) != CL_SUCCESS) {
                // FIXME
                std::cerr << "Error building: "
                    << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(eng.default_device) << "\n";
                throw;
            }

            opencl_kernel_initialized = true;
        }
        bk_dx = std::make_unique<bk_dx_type>(program, "backward_dx");
        bk_dw = std::make_unique<bk_dw_type>(program, "backward_dw");
        bk_db = std::make_unique<bk_db_type>(program, "backward_db");
    }

    void init_opencl(core::engine::opencl& eng, std::vector<size_t> const& sizes, size_t sample_count)
    {
        bk_dx_eargs = std::make_unique<cl::EnqueueArgs>(eng.queue, sizes[0]);

        auto group_size = sample_count * params.out.height * params.out.width;
        bk_dw_eargs = std::make_unique<cl::EnqueueArgs>(
            eng.queue,
            cl::NDRange(sizes[1] * group_size),
            cl::NDRange(group_size)
        );

        local_mem = cl::Local(sizeof(value_type) * group_size);

        bk_db_eargs = std::make_unique<cl::EnqueueArgs>(eng.queue, sizes[2]);
        this->sample_count = sample_count;
    }


    void compute(framework::op_kernel_context& context, core::engine::engine_type& eng, bool united_backend = true) override
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

            average_pooling_op_internal(
                in_data, w, dw, db, dout, dx, params
            );
        } else if (engine == core::backend_type::opencl) {
            using data_type = cl::Buffer;
            data_type& in_data = *std::get<data_type*>(context.input(0));
            data_type& w       = *std::get<data_type*>(context.input(1));
            data_type& dw      = *std::get<data_type*>(context.input_grad(1));
            data_type& db      = *std::get<data_type*>(context.input_grad(2));
            data_type& dx      = *std::get<data_type*>(context.input_grad(0));
            data_type& dout    = *std::get<data_type*>(context.output_grad(0));

            (*bk_dx)(*bk_dx_eargs,
                params.in.width,
                params.in.height,
                params.in.depth,
                params.out.width,
                params.out.height,
                params.out.depth,
                params.pool_width,
                params.pool_height,
                params.stride,
                w, dout, dx
            ).wait();

            (*bk_dw)(*bk_dw_eargs,
                sample_count,
                params.in.width,
                params.in.height,
                params.in.depth,
                params.out.width,
                params.out.height,
                params.out.depth,
                params.pool_width,
                params.pool_height,
                params.stride,
                in_data, dout, dw, local_mem
            ).wait();

            (*bk_db)(*bk_db_eargs,
                sample_count,
                params.out.width,
                params.out.height,
                params.out.depth,
                dout, db
            ).wait();

        } else {
            // TODO not support backend engine
        }
    }

// TODO uncomment it
// private:
    avg_pool_parameter params;
    std::string name;

    size_t sample_count;

    bool opencl_kernel_initialized{false};
    std::unique_ptr<bk_dx_type> bk_dx;
    std::unique_ptr<bk_dw_type> bk_dw;
    std::unique_ptr<bk_db_type> bk_db;

    std::unique_ptr<cl::EnqueueArgs> bk_dx_eargs;
    std::unique_ptr<cl::EnqueueArgs> bk_dw_eargs;
    std::unique_ptr<cl::EnqueueArgs> bk_db_eargs;

    cl::Program::Sources sources;
    cl::Program program;

    cl::LocalSpaceArg local_mem;
};

} // namespace kernel
} // namespace core
} // namespace yonn

