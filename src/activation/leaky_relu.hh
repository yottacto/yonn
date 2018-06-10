#pragma once
#include <iostream>

#include <cmath>
#include <CL/cl.hpp>
#include "layer/layer.hh"
#include "util/util.hh"
#include "core/backend.hh"
#include "tensor.hh"

#include "core/kernel/opencl/leaky-relu.hh"

namespace yonn
{
namespace activation
{

struct leaky_relu : layer
{
    using fk_type = cl::make_kernel<
        value_type,
        cl::Buffer&, cl::Buffer&
    >;

    using bk_type = cl::make_kernel<
        value_type,
        cl::Buffer&, cl::Buffer&, cl::Buffer&
    >;

    explicit leaky_relu(
        value_type epsilon = 0.01,
        core::backend_type backend = core::layer_default_engine()
    ) :
        layer({data_type::data}, {data_type::data}, backend), epsilon{epsilon}
    {}

    explicit leaky_relu(core::backend_type backend, value_type epsilon = 0.01)
        : leaky_relu(epsilon, backend)
    {}


    // TODO explicit specify the dims

    auto name() const -> std::string override
    {
        return "leaky relu layer";
    }

    auto kernel_code() const -> std::string
    {
        return opencl_kernel::leaky_relu_kernel_code;
    }

    auto nd_size() const -> size_t
    {
        return output_shape(0).size() * batch_size;
    }

    auto fan_in_size() const -> size_t override
    {
        return input_shape(0).size();
    }

    auto fan_out_size() const -> size_t override
    {
        return output_shape(0).size();
    }

    void init_engine(
        core::backend_type const& backend,
        core::engine::engine_type& eng
    ) override
    {
        ignore(eng);

        // this backend cannot be network_default
        if (this->backend == core::backend_type::network_default)
            layer::set_engine(backend);
        if (backend == core::backend_type::opencl) {
            auto& e = std::get<core::engine::opencl>(eng);
            init_opencl_kernel(e);
        }
    }

    void init_opencl_kernel(core::engine::opencl& eng);
    void init_opencl(core::engine::opencl& eng, size_t size);

    void allocate_nsamples_opencl(size_t batch_size, core::engine::opencl& e) override;

    void forward_propagation(core::engine::engine_type& eng, bool united_backend) override;
    void backward_propagation(core::engine::engine_type& eng, bool united_backend) override;

    void forward_activation(vec_t const& in, vec_t& out);
    void backward_activation(vec_t const& x, vec_t& dx, vec_t const& y, vec_t const& dy);


    value_type epsilon;

    bool opencl_kernel_initialized{false};
    std::unique_ptr<fk_type> fk;
    std::unique_ptr<bk_type> bk;
    std::unique_ptr<cl::EnqueueArgs> fk_eargs;
    std::unique_ptr<cl::EnqueueArgs> bk_eargs;
    cl::Program::Sources sources;
    cl::Program program;
};

// implementation of leaky_relu
void leaky_relu::allocate_nsamples_opencl(size_t batch_size, core::engine::opencl& e)
{
    this->batch_size = batch_size;
    if (backend == core::backend_type::opencl) {
        input[0]->allocate_nsamples(batch_size, input_shape(0), e.context);
        output[0]->allocate_nsamples(batch_size, output_shape(0), e.context);

        init_opencl(e, batch_size * input_shape(0).size());
    } else {
        // TODO error or currently not supportted backend
    }
}

void leaky_relu::init_opencl_kernel(core::engine::opencl& eng)
{
    if (!opencl_kernel_initialized) {
        sources.emplace_back(opencl_kernel::leaky_relu_kernel_code.c_str(), opencl_kernel::leaky_relu_kernel_code.size());
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
    bk = std::make_unique<bk_type>(program, "backward");
}

void leaky_relu::init_opencl(core::engine::opencl& eng, size_t size)
{
    fk_eargs = std::make_unique<cl::EnqueueArgs>(eng.queue, cl::NDRange(size));
    bk_eargs = std::make_unique<cl::EnqueueArgs>(eng.queue, cl::NDRange(size));
}

void leaky_relu::forward_propagation(core::engine::engine_type& eng, bool united_backend)
{
    auto const& backend = layer::engine();
    if (backend == core::backend_type::internal) {
        tensor const& in_data  = *(input[0] ->get_data());
        tensor&       out_data = *(output[0]->get_data());

        for (size_t sample{0}; sample < in_data.size(); sample++)
            forward_activation(in_data[sample], out_data[sample]);

    } else if (backend == core::backend_type::opencl) {
        if (!united_backend) {
            auto& e = std::get<core::engine::opencl>(eng);
            input[0]->set_data(tensor_to_vector(input[0]->data), e);
        }

        cl::Buffer& in_data  = *(input[0] ->get_data_buffer());
        cl::Buffer& out_data = *(output[0]->get_data_buffer());
        (*fk)(*fk_eargs, epsilon, in_data, out_data);

        if (!united_backend) {
            auto& e = std::get<core::engine::opencl>(eng);
            vector_to_tensor(output[0]->get_data(e), output[0]->data);
        }
    }
}

void leaky_relu::backward_propagation(core::engine::engine_type& eng, bool united_backend)
{
    auto const& backend = layer::engine();
    if (backend == core::backend_type::internal) {
        tensor const& in_data  = *(input[0] ->get_data());
        tensor&       in_grad  = *(input[0] ->get_grad());
        tensor const& out_data = *(output[0]->get_data());
        tensor const& out_grad = *(output[0]->get_grad());

        for (size_t sample{0}; sample < in_data.size(); sample++)
            backward_activation(
                in_data[sample],  in_grad[sample],
                out_data[sample], out_grad[sample]
            );
    } else if (backend == core::backend_type::opencl) {
        if (!united_backend) {
            auto& e = std::get<core::engine::opencl>(eng);
            output[0]->set_data(tensor_to_vector(output[0]->data), e);
            output[0]->set_grad(tensor_to_vector(output[0]->grad), e);
        }

        cl::Buffer& in_grad  = *(input[0] ->get_grad_buffer());
        cl::Buffer& out_data = *(output[0]->get_data_buffer());
        cl::Buffer& out_grad = *(output[0]->get_grad_buffer());
        (*bk)(*bk_eargs, epsilon, out_data, out_grad, in_grad);

        if (!united_backend) {
            auto& e = std::get<core::engine::opencl>(eng);
            // vector_to_tensor(input[i]->get_data(e), input[i]->data);
            vector_to_tensor(input[0]->get_grad(e), input[0]->grad);
        }
    }
}

void leaky_relu::forward_activation(vec_t const& in, vec_t& out)
{
    for (size_t i = 0; i < in.size(); i++)
        out[i] = in[i] > value_type{0} ? in[i] : epsilon * in[i];
}

void leaky_relu::backward_activation(vec_t const& x, vec_t& dx, vec_t const& y, vec_t const& dy)
{
    for (size_t i = 0; i < x.size(); i++)
        dx[i] = dy[i] * (y[i] > value_type{0} ? value_type(1) : epsilon);
}

} // namespace activation
} // namespace yonn

