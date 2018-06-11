#pragma once
#include <iostream>

#include <vector>
#include <memory>
#include <CL/cl.hpp>
#include "type.hh"
#include "absolute.hh"
#include "mse.hh"
#include "softmax.hh"
#include "core/backend.hh"

#include "opencl/gradient.hh"

namespace yonn
{
namespace loss_function
{

// TODO opencl backend

template <class Error>
auto gradient(vec_t const& scores, label_t y)
{
    return Error::df(scores, y);
}

template <class Error>
auto gradient(tensor const& scores, std::vector<label_t> const& y) -> tensor
{
    tensor grads(scores.size());
    for (size_t i{0}; i < scores.size(); i++)
        grads[i] = gradient<Error>(scores[i], y[i]);
    return grads;
}

template <class Error>
struct opencl_gradient
{
    using kernel_type = typename Error::kernel_type;

    opencl_gradient() = default;

    void allocate_output(size_t batch_size, core::engine::opencl& eng)
    {
        if (!opencl_kernel_initialized) {
            sources.emplace_back(
                opencl_kernel::gradient_kernel_code.c_str(),
                opencl_kernel::gradient_kernel_code.size()
            );
            program = cl::Program{eng.context, sources};
            if (program.build({eng.default_device}) != CL_SUCCESS) {
                // FIXME
                std::cerr << "Error building: "
                    << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(eng.default_device) << "\n";
                throw;
            }
            kernel = std::make_shared<kernel_type>(program, Error::name());
            opencl_kernel_initialized = true;
        }

        auto bsize = batch_size * sizeof(label_t);
        labels_buffer = cl::Buffer(eng.context, CL_MEM_READ_WRITE, bsize);
        this->batch_size = batch_size;
    }

    auto gradient(
        cl::Buffer& output,
        size_t out_size,
        std::vector<label_t> const& desired_output,
        cl::Buffer& grad_buffer,
        core::engine::opencl& e
    )
    {
        auto bsize = batch_size * sizeof(label_t);
        e.queue.enqueueWriteBuffer(
            labels_buffer,
            CL_TRUE,
            0,
            bsize,
            desired_output.data()
        );

            if (kernel == nullptr)
                std::cerr << "------------- amazing!\n";
        (*kernel)(
            cl::EnqueueArgs{e.queue, cl::NDRange(batch_size * out_size)},
            batch_size,
            out_size,
            output,
            labels_buffer,
            grad_buffer
        ).wait();
    }

    bool opencl_kernel_initialized{false};
    size_t batch_size;

    // change to std::shared_ptr, bacause the outter need std::any, and
    // std::any has a copy ctor
    std::shared_ptr<kernel_type> kernel;
    cl::Program::Sources sources;
    cl::Program program;

    cl::Buffer labels_buffer;
};

} // namespace loss_function
} // namespace yonn

