#pragma once
#include <string>

namespace yonn
{
namespace opencl_kernel
{

std::string const conv_kernel_code{R"(

typedef unsigned int size_t;
typedef float value_type;

kernel size_t get_index(size_t w, size_t h, size_t d, size_t x, size_t y, size_t z)
{
    return (z * h + y) * w + x;
}

kernel void forward(
    size_t sample_count,
    size_t in_w,
    size_t in_h,
    size_t in_d,
    size_t out_w,
    size_t out_h,
    size_t out_d,
    size_t w_w,
    size_t w_h,
    size_t w_s,
    bool has_bias,
    global int const* table,
    global value_type const* in_data,
    global value_type const* w,
    global value_type const* bias,
    global value_type* out_data
)
{
    // (sample_count, out_d, out_h, out_w)
    // (sample,       od,    y,     x)
    size_t out_size = out_d * out_h * out_w;
    size_t out_area = out_h * out_w;
    size_t id = get_global_id(0);
    size_t sample = id / out_size;
    id %= out_size;
    size_t od = id / area;
    id %= area;
    size_t y = id / out_w;
    id %= out_w;
    size_t x = id;

    global value_type const* in = in_data + sample * (in_w * in_h * in_d);
    global value_type* out = out_data + sample * out_size
        + get_index(out_w, out_h, out_d, x, y, od);

    out[0] = 0;

    for (size_t i = 0; i < in_d; i++) {
        if (!table[od + i * out_d])
            continue;

        size_t idx = get_index(w_w, w_h, in_d * out_d, 0, 0, in_d * i + od);
        global value_type const* pw = w + idx;
        idx = get_index(in_w, in_h, in_d, 0, 0, j);
        global value_type const* pin = in + idx;

        global value_type const* pin_element = pin_line;
        global value_type const* pw_element = pw;
        value_type sum{0};

        for (size_t yi{0}; yi < w_h; yi++) {
            for (size_t xi{0}; xi < w_w; xi++) {
                sum += pw_element[xi] * pin_element[xi];
            }
            pw_element  += w_w;
            pin_element += in_w;
        }
        out[0] += sum;
    }
    if (has_bias)
        out[0] += bias[od];
}

kernel void backward_dx(
    size_t sample_count,
    size_t in_w,
    size_t in_h,
    size_t in_d,
    size_t out_w,
    size_t out_h,
    size_t out_d,
    size_t w_w,
    size_t w_h,
    size_t w_s,
    size_t h_s,
    global int const* table,
    global value_type const* w,
    global value_type const* dout,
    global value_type* dx
)
{
    size_t in_size  = in_d * in_h * in_w;
    size_t in_area  = in_h * in_w;
    size_t out_size = out_d * out_h * out_w;
    size_t out_area = out_h * out_w;
    size_t w_d      = in_d * out_d;
    // (sample_count, in_w, in_h, in_d)
    // (sample,       iw,   ih,   id)
    size_t tid = get_global_id(0);
    size_t sample = tid / in_size;
    tid %= in_size;
    size_t id = tid / in_area;
    tid %= in_area;
    size_t ih = tid / in_w;
    tid %= in_w;
    size_t iw = tid;

    value_type sum = 0;
    // for (size_t id{0}; id < params.in.depth; id++)
    if (iw + 1 - w_w >= 0 && ih + 1 - w_h >= 0) {
        for (size_t od{0}; od < out_d; od++) {
            if (!table[od + id * out_d])
                continue;

            global value_type const* weight = w
                + get_index(w_w, w_h, in_d * out_d, 0, 0, in_d * od + id);

            global value_type const* dout_now = dout + sample * out_size
                + od * out_area;

            size_t ow = (iw + 1 - w_w + w_s - 1) / w_s;
            size_t oh = (ih + 1 - w_h + h_s - 1) / h_s;
            size_t ww = iw - ow * w_s;
            size_t wh = ih - oh * h_s;

            for (; wh >= 0; wh -= h_s, oh++)
                for (; ww >= 0; ww -= w_s, ow++)
                    sum += w[wh * w_w + ww] * dout_now[oh * out_w + ow];
        }
    }
    dx[tid] = sum;
}

kernel void backward_dw(
    size_t sample_count,
    size_t in_w,
    size_t in_h,
    size_t in_d,
    size_t out_w,
    size_t out_h,
    size_t out_d,
    size_t w_w,
    size_t w_h,
    size_t w_s,
    size_t h_s,
    global int const* table,
    global value_type const* in_data,
    global value_type const* dout,
    global value_type* dw,
)
{
    size_t w_d      = in_d * out_d;
    size_t w_size   = w_d * w_h * w_w;
    size_t w_area   = w_h * w_w;
    size_t in_size  = in_d * in_h * in_w;
    size_t in_area  = in_h * in_w;
    size_t out_size = out_d * out_h * out_w;
    size_t out_area = out_h * out_w;
    // (w_w, w_h, in_d * out_d)
    // (ww,  wh,  wd)
    size_t tid = get_global_id(0);
    size_t wd = tid / w_area;
    tid %= w_area;
    size_t wh = tid / w_w;
    tid %= w_w;
    size_t ww = tid;

    size_t id = wd % in_d;
    size_t od = wd / in_d;

    value_type sum = 0;
    for (size_t sample = 0; sample < sample_count; sample++) {
        global value_type const* in = in_data + sample * in_size
            + id * in_area;
        global value_type const* dout_now = dout + sample * out_size
            + od * out_area;

        for (size_t i = wh, oi = 0; i < in_h; i += h_s, oi++)
            for (size_t j = ww, oj = 0; j < in_w; j += w_s, oj++)
                sum += in[i * in_w + j] * dout_now[oi * out_w + oj];
    }
    dw[tid] = sum;
}

kernel void backward_db(
    size_t sample_count,
    size_t out_w,
    size_t out_h,
    size_t out_d,
    global value_type const* dout,
    global value_type* db,
)
{
    size_t out_size = out_d * out_h * out_w;
    size_t out_area = out_h * out_w;

    size_t od = get_global_id(0);

    value_type sum = 0;
    for (size_t sample = 0; sample < sample_count; sample++) {
        global value_type const* dout_now = dout + sample * out_size
            + od * out_area;

        for (size_t i = 0; i < out_area; i++)
            sum += dout_now[i];
    }

    db[tid] = sum;

)"};

} // namespace opencl_kernel
} // namespace yonn

