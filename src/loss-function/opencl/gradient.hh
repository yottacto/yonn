#pragma once
#include <string>

namespace yonn
{
namespace opencl_kernel
{

inline std::string gradient_kernel_code{R"(

// #include "../../core/kernel/opencl/typedef.hh"
typedef double value_type;

kernel void mse(
    int sample_count,
    int out_size,
    global value_type const* scores,
    global unsigned int const* labels,
    global value_type* d
)
{
    // (out_size, sample_count)
    // (i,        sample)
    int gid = get_global_id(0);
    int tid = gid;
    int sample = tid / out_size;
    int i = tid % out_size;

    value_type factor = (value_type)(2) / (value_type)(out_size);
    global value_type const* score = scores + sample * out_size;
    d[gid] = factor
        * (score[i] - (i == labels[sample]
            ? (value_type)(1)
            : (value_type)(0)));
}

)"};

} // namespace opencl_kernel
} // namespace yonn

