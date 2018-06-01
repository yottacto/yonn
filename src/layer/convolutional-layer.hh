#pragma once
#include <memory>
#include "layer.hh"
#include "type.hh"
#include "util.hh"
#include "core/parameter/conv-parameter.hh"
#include "core/framework/op-kernel.hh"
#include "core/kernel/convolutional-op.hh"

namespace yonn
{

struct convolutional_layer : layer
{
    // TODO backend
    convolutional_layer(
        size_t in_width,
        size_t in_height,
        size_t window_width,
        size_t window_height,
        size_t in_channels,
        size_t out_channels,
        core::connection_table const& table,
        padding pad_type,
        bool has_bias,
        size_t w_stride,
        size_t h_stride,
        core::backend backend
    );

    convolutional_layer(
        size_t in_width,
        size_t in_height,
        size_t window_size,
        size_t in_channels,
        size_t out_channels,
        padding pad_type,
        bool has_bias,
        size_t w_stride,
        size_t h_stride,
        core::backend backend
    );

    void forward_propagation() override;
    void backward_propagation() override;

private:
    core::conv_parameter params;
    core::framework::op_kernel_context forward_context;
    core::framework::op_kernel_context backward_context;
    std::shared_ptr<core::framework::op_kernel> forward_kernel;
    std::shared_ptr<core::framework::op_kernel> backward_kernel;
};

// implementation of convolutional_layer
convolutional_layer::convolutional_layer(
    size_t in_width,
    size_t in_height,
    size_t window_width,
    size_t window_height,
    size_t in_channels,
    size_t out_channels,
    core::connection_table const& table,
    padding pad_type = padding::valid,
    bool has_bias = true,
    size_t w_stride = 1,
    size_t h_stride = 1,
    core::backend backend = core::default_engine()
) :
    layer(std_input_types(has_bias), {data_type::data}),
    params(
        shape3d_t{in_width, in_height, in_channels},
        window_width,
        window_height,
        out_channels,
        pad_type,
        has_bias,
        w_stride,
        h_stride,
        table
    ),
    // FIXME op_kernel need context to constrct, in fact in order to
    // specify device and layer's params
    forward_kernel(new core::kernel::convolutional_op(params)),
    backward_kernel(new core::kernel::convolutional_grad_op(params))
{
    in_shapes.emplace_back(
        in_length(in_width, window_width, pad_type),
        in_length(in_height,  window_height, pad_type),
        in_channels
    );
    in_shapes.emplace_back(
        window_width,
        window_height,
        in_channels * out_channels
    );
    if (has_bias)
        in_shapes.emplace_back(
            out_length(in_width, window_width, w_stride, pad_type),
            out_length(in_height, window_height, h_stride, pad_type),
            out_channels
        );

    out_shapes.emplace_back(
        out_length(in_width, window_width, w_stride, pad_type),
        out_length(in_height, window_height, h_stride, pad_type),
        out_channels
    );

    // invariant, all input channels allocated in constructor
    // TODO reasoning about this input_shape
    input[0] = std::make_shared<edge>();
    input[1] = std::make_shared<edge>();
    input[2] = std::make_shared<edge>();

    output[0] = std::make_shared<edge>();

    // TODO init different kernel
}

convolutional_layer::convolutional_layer(
    size_t in_width,
    size_t in_height,
    size_t window_size,
    size_t in_channels,
    size_t out_channels,
    padding pad_type = padding::valid,
    bool has_bias = true,
    size_t w_stride = 1,
    size_t h_stride = 1,
    core::backend backend = core::default_engine()
) :
    convolutional_layer(
        in_width,    in_height,
        window_size, window_size,
        in_channels, out_channels,
        core::connection_table(),
        pad_type,
        has_bias,
        w_stride, h_stride,
        backend
    )
{
}

void convolutional_layer::forward_propagation()
{
    // TODO not same size padding, need local storage

    // TODO init once
    // TODO const in data?
    std::vector<tensor*> in_data(in_channels);
    for (size_t i{0}; i < in_channels; i++)
        in_data[i] = input[i]->get_data();
    std::vector<tensor*> out_data(out_channels);
    for (size_t i{0}; i < out_channels; i++)
        out_data[i] = output[i]->get_data();

    forward_context.set_in_out(in_data, out_data);
    forward_context.set_engine(layer::engine());

    forward_kernel->compute(forward_context);
}

void convolutional_layer::backward_propagation()
{
    // TODO same size padding

    std::vector<tensor*> in_data(in_channels);
    std::vector<tensor*> in_grad(in_channels);
    for (size_t i{0}; i < in_channels; i++) {
        in_data[i] = input[i]->get_data();
        in_grad[i] = input[i]->get_grad();
    }
    std::vector<tensor*> out_data(out_channels);
    std::vector<tensor*> out_grad(out_channels);
    for (size_t i{0}; i < out_channels; i++) {
        out_data[i] = output[i]->get_data();
        out_grad[i] = output[i]->get_grad();
    }

    backward_context.set_in_out(in_data, in_grad, out_data, out_grad);
    backward_context.set_engine(layer::engine());

    backward_kernel->compute(backward_context);
}

} // namespace yonn

