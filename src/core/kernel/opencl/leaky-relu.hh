#pragma once
#include <string>

namespace yonn
{
namespace opencl_kernel
{

std::string leaky_relu_kernel_code{R"(

typedef double value_type;

kernel void forward(
    value_type epsilon,
    global value_type const* in,
    global value_type* out
)
{
    // (in_size, sample_count)
    // (i,       sample)
    int gid = get_global_id(0);

    out[gid] = in[gid] > 0 ? in[gid] : epsilon * in[gid];
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

    dx[gid] = dout[gid] * (out[gid] > 0 ? 1 : epsilon);
}

)"};

} // namespace opencl_kernel
} // namespace yonn

