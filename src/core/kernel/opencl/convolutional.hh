#pragma once
#include <string>

namespace yonn
{
namespace opencl_kernel
{

inline std::string const conv_kernel_code{R"(

// #include "typedef.hh"
typedef float value_type;

int get_index(int w, int h, int d, int x, int y, int z)
{
    return (z * h + y) * w + x;
}

kernel void forward(
    int in_w,
    int in_h,
    int in_d,
    int out_w,
    int out_h,
    int out_d,
    int w_w,
    int w_h,
    int w_s,
    int h_s,
    int has_bias,
    global int const* table,
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
    int w_d = in_d * out_d;
    int w_size = w_w * w_h * w_d;
    int w_area = w_w * w_h;

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

    global value_type const* in = in_data + sample * in_size;
    value_type sum = 0;
    for (int id = 0; id < in_d; id++) {
        if (!table[od + id * out_d])
            continue;

        int iw = ow * w_s;
        int ih = oh * h_s;

        global value_type const* pw = w + (in_d * od + id) * w_area;
        global value_type const* pin = in + id * in_area
            + ih * in_w + iw;

        value_type tsum = 0;
        for (int yi = 0; yi < w_h; yi++) {
            for (int xi = 0; xi < w_w; xi++)
                tsum += pw[xi] * pin[xi];
            pw  += w_w;
            pin += in_w;
        }
        sum += tsum;
    }
    if (has_bias)
        sum += bias[od];
    out[gid] = sum;
}

kernel void backward_dx(
    int sample_count,
    int in_w,
    int in_h,
    int in_d,
    int out_w,
    int out_h,
    int out_d,
    int w_w,
    int w_h,
    int w_s,
    int h_s,
    global int const* table,
    global value_type const* w,
    global value_type const* dout,
    global value_type* dx
)
{
    int in_size  = in_d * in_h * in_w;
    int in_area  = in_h * in_w;
    int out_size = out_d * out_h * out_w;
    int out_area = out_h * out_w;
    int w_d      = in_d * out_d;
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


    value_type sum = 0;
    for (int od = 0; od < out_d; od++) {
        if (!table[od + id * out_d])
            continue;

        global value_type const* weight = w
            + get_index(w_w, w_h, w_d, 0, 0, in_d * od + id);

        global value_type const* dout_now = dout + sample * out_size
            + od * out_area;

        int ow = (iw + 1 - w_w + w_s - 1) / w_s;
        int oh = (ih + 1 - w_h + h_s - 1) / h_s;
        int ww = iw - ow * w_s;
        int wh = ih - oh * h_s;

        for (int hi = wh, ohi = oh; hi >= 0 && ohi < out_h; hi -= h_s, ohi++)
            for (int wi = ww, owi = ow; wi >= 0 && owi < out_w; wi -= w_s, owi++) {
                if (ohi >= 0 && owi >= 0) {
                    sum += weight[hi * w_w + wi] * dout_now[ohi * out_w + owi];
                }
            }


    }
    dx[gid] = sum;

    // if (gid == 43) {
    //     printf("(iw, w_w, ih, w_h) = (%d %d %d %d) \n", iw, w_w, ih, w_h);
    //     printf("-> %lf\n", sum);
    // }

}

kernel void backward_dw(
    int sample_count,
    int in_w,
    int in_h,
    int in_d,
    int out_w,
    int out_h,
    int out_d,
    int w_w,
    int w_h,
    int w_s,
    int h_s,
    global int const* table,
    global value_type const* in_data,
    global value_type const* dout,
    global value_type* dw
)
{
    int w_d      = in_d * out_d;
    int w_size   = w_d * w_h * w_w;
    int w_area   = w_h * w_w;
    int in_size  = in_d * in_h * in_w;
    int in_area  = in_h * in_w;
    int out_size = out_d * out_h * out_w;
    int out_area = out_h * out_w;
    // (w_w, w_h, in_d * out_d)
    // (ww,  wh,  wd)
    int gid = get_global_id(0);
    int tid = gid;
    int wd = tid / w_area;
    tid %= w_area;
    int wh = tid / w_w;
    tid %= w_w;
    int ww = tid;

    int id = wd % in_d;
    int od = wd / in_d;

    value_type sum = 0;
    for (int sample = 0; sample < sample_count; sample++) {
        global value_type const* in = in_data + sample * in_size
            + id * in_area;
        global value_type const* dout_now = dout + sample * out_size
            + od * out_area;

        for (int i = wh, oi = 0; i < in_h && oi < out_w; i += h_s, oi++)
            for (int j = ww, oj = 0; j < in_w && oj < out_h; j += w_s, oj++)
                sum += in[i * in_w + j] * dout_now[oi * out_w + oj];
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

    int od = get_global_id(0);

    value_type sum = 0;
    for (int sample = 0; sample < sample_count; sample++) {
        global value_type const* dout_now = dout + sample * out_size
            + od * out_area;

        for (int i = 0; i < out_area; i++)
            sum += dout_now[i];
    }

    db[od] = sum;

    // if (od == 0) {
    //     printf("-> %lf\n", db[od]);
    // }

}

)"};

} // namespace opencl_kernel
} // namespace yonn

