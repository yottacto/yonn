#pragma once
#include <string>

namespace yonn
{
namespace opencl_kernel
{

inline std::string tanh_kernel_code{R"(

// #include "typedef.hh"
typedef float value_type;

kernel void forward(
    global value_type const* in,
    global value_type* out
)
{
    // (in_size, sample_count)
    // (i,       sample)
    int gid = get_global_id(0);

    out[gid] = tanh(in[gid]);
}

kernel void backward(
    value_type epsilon,
    global value_type const* out,
    global value_type const* dout,
    global value_type* dx
)
{
    // (in_size, sample_count)
    // (i,       sample)
    int gid = get_global_id(0);

    dx[gid] = dout[gid] * ((value_type)(1) - out[i] * out[i]);
}

)"};

} // namespace opencl_kernel
} // namespace yonn

