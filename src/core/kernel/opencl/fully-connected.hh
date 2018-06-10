#pragma once
#include <string>

namespace yonn
{
namespace opencl_kernel
{

std::string fully_kernel_code{R"(

typedef unsigned int size_t;
typedef double value_type;

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

kernel void backward_x(
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
    global value_type* dw,
    global value_type* db,
    global value_type const* dout,
    global value_type const* bias,
    global value_type* dx
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

}

kernel void backward_w(
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
    global value_type* dw,
    global value_type* db,
    global value_type const* dout,
    global value_type const* bias,
    global value_type* dx
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

}

kernel void backward_b(
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
    global value_type* dw,
    global value_type* db,
    global value_type const* dout,
    global value_type const* bias,
    global value_type* dx
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

}

)"};

} // namespace opencl_kernel
} // namespace yonn

