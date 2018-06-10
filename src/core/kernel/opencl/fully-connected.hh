#pragma once
#include <string>

namespace yonn
{
namespace opencl_kernel
{

std::string fully_kernel_code{R"(

kernel void forward(
    int in_size,
    int out_size,
    int has_bias,
    global value_type const* in_data,
    global value_type const* w,
    global value_type const* bias,
    global value_type* out
)
{
    // (out_size, sample_count)
    // (oi,       sample)
    int gid = get_global_id(0);
    int tid = gid;
    int sample = tid / out_size;
    tid %= out_size;
    int oi = tid;

    global value_type const* in_data = in + sample * in_size;
    value_type sum = 0;
    for (int i = 0; i < in_size; i++)
        sum += in[i] * w[i * out_size + oi];
    if (has_bias)
        sum += bias[oi];
    out[gid] = sum;
}

kernel void backward_dx(
    int sample_count,
    int in_size,
    int out_size,
    global value_type const* w,
    global value_type const* dout,
    global value_type* dx
)
{
    // (in_size, sample_count)
    // (i,       sample)
    int gid = get_global_id(0);
    int tid = gid;
    int sample = tid / in_size;
    tid %= in_size;
    int i = tid;

    value_type sum = 0;
    global value_type const* dout_now = dout + sample * out_size;
    global value_type const* weight = w + i * out_size;
    for (int j = 0; j < out_size; j++)
        sum += dout_now[j] * weight[j];
    dx[gid] = sum;
}

kernel void backward_dw(
    int sample_count,
    int in_size,
    int out_size,
    global value_type const* in_data,
    global value_type const* dout,
    global value_type* dw
)
{
    // (out_size, in_size)
    // (wi,       hi)
    int gid = get_global_id(0);
    int tid = gid;
    int hi = tid / out_size;
    tid %= w_area;
    int wi = tid;

    value_type sum = 0;
    for (int sample = 0; sample < sample_count; sample++) {
        global value_type const* in = in_data + sample * in_size;
        global value_type const* dout_now = dout + sample * out_size;

        sum += in[hi] * dout_now[wi];
    }
    dw[gid] = sum;
}

kernel void backward_db(
    int sample_count,
    int out_size,
    global value_type const* dout,
    global value_type* db
)
{
    // out_size
    int oi = get_global_id(0);

    value_type sum = 0;
    global value_type const* dout_now = dout;
    for (int sample = 0; sample < sample_count; sample++) {
        sum += dout_now[oi];
        dout_now += out_size;
    }

    db[oi] = sum;
}

)"};

} // namespace opencl_kernel
} // namespace yonn

