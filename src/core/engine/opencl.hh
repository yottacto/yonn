#pragma once
#include <any>
#include <variant>
#include <unordered_map>
#include <vector>
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
    }

    std::vector<cl::Platform> all_platforms;
    cl::Platform default_platform;
    std::vector<cl::Device> all_devices;
    cl::Device default_device;
    cl::Context context;

    std::unordered_map<framework::op_kernel*, std::any>   kernels;
    std::unordered_map<buffer_key_type,       cl::Buffer> buffers;
};


} // namespace engine
} // namespace core
} // namespace yonn

