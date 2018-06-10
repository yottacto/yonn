#pragma once
#include <string>

namespace yonn
{
namespace opencl_kernel
{

std::string avg_pool_kernel_code{R"(

typedef double value_type;

kernel void forward(
    int in_w,
    int in_h,
    int in_d,
    int out_w,
    int out_h,
    int out_d,
    int pool_w,
    int pool_h,
    int stride,
    // int has_bias,
    global value_type const* in_data,
    global value_type const* w,
    global value_type const* bias,
    global value_type* out
)
{
    int in_size = in_d * in_h * in_w;
    int in_area = in_h * in_w;
    int out_size = out_d * out_h * out_w;
    int out_area = out_h * out_w;
    int pool_area = pool_w * pool_h;

    // (out_h, out_w, out_d, sample_count)
    // (oh,    ow,    od,    sample)
    int gid = get_global_id(0);
    int tid = gid;
    int sample = tid / out_size;
    tid %= out_size;
    int od = tid / out_area;
    tid %= out_area;
    int oh = tid / out_w;
    tid %= out_w;
    int ow = tid;

    int id = od;
    int ih = oh * stride;
    int iw = ow * stride;
    global value_type const* in = in_data + sample * in_size
        + id * in_area + ih * in_w + iw;

    value_type sum = 0;
    for (int wh = 0; wh < pool_h && ih + wh < in_h; wh++) {
        for (int ww = 0; ww < pool_w && iw + ww < in_w; ww++)
            sum += in[ww];
        in += in_w;
    }
    sum *= w[od];
    sum /= value_type(pool_area);
    // TODO if has_bias
    sum += bias[od];

    out[gid] = sum;
}

kernel void backward_dx(
    int in_w,
    int in_h,
    int in_d,
    int out_w,
    int out_h,
    int out_d,
    int pool_w,
    int pool_h,
    int stride,
    global value_type const* w,
    global value_type const* dout,
    global value_type* dx
)
{
    int in_size  = in_d * in_h * in_w;
    int in_area  = in_h * in_w;
    int out_size = out_d * out_h * out_w;
    int out_area = out_h * out_w;
    int pool_area = pool_w * pool_h;
    // (in_w, in_h, in_d, sample_count)
    // (iw,   ih,   id,   sample)
    int gid = get_global_id(0);
    int tid = gid;
    int sample = tid / in_size;
    tid %= in_size;
    int id = tid / in_area;
    tid %= in_area;
    int ih = tid / in_w;
    tid %= in_w;
    int iw = tid;


    int od = id;
    int ow = (iw + 1 - pool_w + stride - 1) / stride;
    int oh = (ih + 1 - pool_h + stride - 1) / stride;
    int ww = iw - ow * stride;
    int wh = ih - oh * stride;

    global value_type const* dout_now = dout + sample * out_size
        + od * out_area;

    value_type sum = 0;
    for (int hi = wh, ohi = oh; hi >= 0 && ohi < out_h; hi -= stride, ohi++)
        for (int wi = ww, owi = ow; wi >= 0 && owi < out_w; wi -= stride, owi++) {
            if (ohi >= 0 && owi >= 0) {
                sum += w[od] * dout_now[ohi * out_w + owi] / pool_area;
            }
        }

    dx[gid] = sum;
}

kernel void backward_dw(
    int sample_count,
    int in_w,
    int in_h,
    int in_d,
    int out_w,
    int out_h,
    int out_d,
    int pool_w,
    int pool_h,
    int stride,
    global value_type const* in_data,
    global value_type const* dout,
    global value_type* dw
)
{
    int pool_area   = pool_h * pool_w;
    int in_size  = in_d * in_h * in_w;
    int in_area  = in_h * in_w;
    int out_size = out_d * out_h * out_w;
    int out_area = out_h * out_w;

    int gid = get_global_id(0);
    int tid = gid;
    int id = gid;
    int od = id;

    value_type sum = 0;
    for (int sample = 0; sample < sample_count; sample++) {
        global value_type const* in = in_data + sample * in_size
            + id * in_area;
        global value_type const* dout_now = dout + sample * out_size
            + od * out_area;

        for (int oh = 0; oh < out_h; oh++)
        for (int ow = 0; ow < out_w; ow++) {
            value_type tsum = 0;
            for (int ih = oh * stride, w = 0; w < pool_h && ih < in_h; w++, ih++)
            for (int iw = ow * stride, h = 0; h < pool_h && iw < in_w; h++, iw++)
                tsum += in[ih * in_w + iw];
            tsum /= pool_area;
            sum += tsum * dout_now[oh * out_w + ow];
        }
    }
    dw[gid] = sum;
}

kernel void backward_db(
    int sample_count,
    int out_w,
    int out_h,
    int out_d,
    global value_type const* dout,
    global value_type* db
)
{
    int out_size = out_d * out_h * out_w;
    int out_area = out_h * out_w;

    int gid = get_global_id(0);
    int tid = gid;
    int id = gid;
    int od = id;

    value_type sum = 0;
    for (int sample = 0; sample < sample_count; sample++) {
        global value_type const* dout_now = dout + sample * out_size
            + od * out_area;

        for (int oh = 0; oh < out_h; oh++)
        for (int ow = 0; ow < out_w; ow++)
            sum += dout_now[oh * out_w + ow];
    }
    db[gid] = sum;
}

)"};

} // namespace opencl_kernel
} // namespace yonn

