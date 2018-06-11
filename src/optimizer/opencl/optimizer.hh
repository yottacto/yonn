#pragma once
#include <string>

namespace yonn
{
namespace opencl_kernel
{

inline std::string optimizer_kernel_code{R"(

// #include "../../core/kernel/opencl/typedef.hh"
typedef double value_type;

kernel void naive(
    value_type alpha,
    global value_type const* dw,
    global value_type* w
)
{
    int gid = get_global_id(0);
    w[gid] -= alpha * dw[gid];
}

)"};

} // namespace opencl_kernel
} // namespace yonn

