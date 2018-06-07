#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <variant>
#include <any>
#include <CL/cl.hpp>
#include "type.hh"
#include "tensor.hh"

namespace yonn
{
namespace core
{
namespace engine
{

// TODO
namespace framework
{
struct op_kernel;
} // namespace framework

struct opencl
{
    using buffer_key_type = std::variant<vec_t const*, tensor const*>;

    opencl()
    {
        cl::Platform::get(&all_platforms);
        default_platform = all_platforms[0];

        default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
        default_device = all_devices[0];

        context = cl::Context{default_device};

        queue = cl::CommandQueue{context, default_device};
    }

    void init_kernel(
        std::string const& name,
        std::string const& kernel_code,
        size_t nd_size
    )
    {
        cl::Program::Sources sources;

        sources.emplace_back(kernel_code.c_str(), kernel_code.size());

        cl::Program program{context, sources};
        if (program.build({default_device}) != CL_SUCCESS) {
            // TODO throw error
        }

        forward_kernels.emplace(std::make_pair(
            name,
            cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&>(program, "forward")
        ));

        backward_kernels.emplace(std::make_pair(
            name,
            cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&>(program, "backward")
        ));

        forward_eargs.emplace(std::make_pair(
            name,
            cl::EnqueueArgs{queue, cl::NDRange(nd_size)}
        ));
        backward_eargs.emplace(std::make_pair(
            name,
            cl::EnqueueArgs{queue, cl::NDRange(nd_size)}
        ));
    }

    std::vector<cl::Platform> all_platforms;
    cl::Platform default_platform;
    std::vector<cl::Device> all_devices;
    cl::Device default_device;
    cl::Context context;
    cl::CommandQueue queue;

    std::unordered_map<std::string, std::any> forward_kernels;
    std::unordered_map<std::string, std::any> backward_kernels;
    std::unordered_map<std::string, cl::EnqueueArgs> forward_eargs;
    std::unordered_map<std::string, cl::EnqueueArgs> backward_eargs;
};


} // namespace engine
} // namespace core
} // namespace yonn

