#pragma once
#include <variant>
#include <memory>
#include <string>
#include "layer.hh"
#include "type.hh"
#include "util/util.hh"
#include "core/parameter/conv-parameter.hh"
#include "core/framework/op-kernel.hh"
#include "core/kernel/convolutional-op.hh"

#include "core/kernel/opencl/convolutional.hh"

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
        core::backend_type backend
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
        core::backend_type backend
    );

    convolutional_layer(
        size_t in_width,
        size_t in_height,
        size_t window_size,
        size_t in_channels,
        size_t out_channels,
        core::connection_table const& table,
        padding pad_type,       // = padding::valid,
        bool has_bias,          // = true,
        size_t w_stride,        // = 1,
        size_t h_stride,        // = 1,
        core::backend_type backend   // = core::default_engine()
    );

    auto name() const -> std::string override
    {
        return "convolutional layer";
    }

    auto kernel_code() const -> std::string
    {
        return opencl_kernel::conv_kernel_code;
    }

    auto nd_size() const -> size_t
    {
        return output_shape(0).size() * batch_size;
    }

    auto fan_in_size() const -> size_t override
    {
        return params.weight.width * params.weight.height * params.weight.depth;
    }

    auto fan_out_size() const -> size_t override
    {
        return (params.weight.width/params.w_stride)
            * (params.weight.height/params.h_stride) * params.out.depth;
    }

    void init_engine(
        core::backend_type const& backend,
        core::engine::engine_type& eng
    ) override;

    void forward_propagation(core::engine::engine_type& eng) override;
    void backward_propagation(core::engine::engine_type& eng) override;

// private:
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
    core::backend_type backend = core::default_engine()
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
    forward_kernel(new core::kernel::convolutional_op(params, name())),
    backward_kernel(new core::kernel::convolutional_grad_op(params, name()))
{
    in_shapes.emplace_back(
        in_length(in_width, window_width, pad_type),
        in_length(in_height, window_height, pad_type),
        in_channels
    );
    in_shapes.emplace_back(
        window_width,
        window_height,
        in_channels * out_channels
    );
    if (has_bias)
        in_shapes.emplace_back(
            1,
            1,
            out_channels
        );

    out_shapes.emplace_back(
        out_length(in_width, window_width, w_stride, pad_type),
        out_length(in_height, window_height, h_stride, pad_type),
        out_channels
    );

    // TODO init different kernel
    if (backend == core::backend_type::internal) {
        // invariant, all input channels allocated in constructor
        // TODO reasoning about this input_shape
        input[0] = std::make_shared<edge>();
        input[1] = std::make_shared<edge>(input_shape(1));
        input[2] = std::make_shared<edge>(input_shape(2));

        output[0] = std::make_shared<edge>();
    } else if (backend == core::backend_type::opencl) {
        // wait for opencl context, init in init_engine
    } else {
    }

    // TODO init weight
    for (size_t i{0}; i < in_types.size(); i++)
        if (in_types[i] == data_type::weight)
            init_weight(input[i]->data[0], fan_in_size(), fan_out_size());
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
    core::backend_type backend = core::default_engine()
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

convolutional_layer::convolutional_layer(
    size_t in_width,
    size_t in_height,
    size_t window_size,
    size_t in_channels,
    size_t out_channels,
    core::connection_table const& table,
    padding pad_type = padding::valid,
    bool has_bias = true,
    size_t w_stride = 1,
    size_t h_stride = 1,
    core::backend_type backend = core::default_engine()
) :
    convolutional_layer(
        in_width,    in_height,
        window_size, window_size,
        in_channels, out_channels,
        table,
        pad_type,
        has_bias,
        w_stride, h_stride,
        backend
    )
{
}

void convolutional_layer::init_engine(
    core::backend_type const& backend,
    core::engine::engine_type& eng
)
{
    // this backend cannot be network_default
    if (this->backend == core::backend_type::network_default)
        layer::set_engine(backend);

    // internal is inited in ctor
    if (backend == core::backend_type::opencl) {
        auto const& e = std::get<core::engine::opencl>(eng);
        input[0] = std::make_shared<edge>();
        input[1] = std::make_shared<edge>(input_shape(1), e.context);
        input[2] = std::make_shared<edge>(input_shape(2), e.context);

        output[0] = std::make_shared<edge>();
    }
}

void convolutional_layer::forward_propagation(core::engine::engine_type& eng)
{
    // TODO not same size padding, need local storage

    // TODO init once
    // TODO const in data?
    using data_type = std::variant<tensor*, cl::Buffer*>;
    std::vector<data_type> in_data(in_channels);
    std::vector<data_type> out_data(out_channels);
    auto const& backend = layer::engine();
    if (backend == core::backend_type::internal) {
        for (size_t i{0}; i < in_channels; i++)
            in_data[i].emplace<tensor*>(input[i]->get_data());

        for (size_t i{0}; i < out_channels; i++)
            out_data[i].emplace<tensor*>(output[i]->get_data());

    } else if (backend == core::backend_type::opencl) {
        for (size_t i{0}; i < in_channels; i++)
            in_data[i].emplace<cl::Buffer*>(input[i]->get_data_buffer());

        for (size_t i{0}; i < out_channels; i++)
            out_data[i].emplace<cl::Buffer*>(output[i]->get_data_buffer());
    }

    forward_context.set_in_out(in_data, out_data);
    forward_context.set_engine(layer::engine());

    forward_kernel->compute(forward_context, eng);
}

void convolutional_layer::backward_propagation(core::engine::engine_type& eng)
{
    // TODO same size padding

    using data_type = std::variant<tensor*, cl::Buffer*>;
    std::vector<data_type> in_data(in_channels);
    std::vector<data_type> in_grad(in_channels);
    std::vector<data_type> out_data(out_channels);
    std::vector<data_type> out_grad(out_channels);

    auto const& backend = layer::engine();
    if (backend == core::backend_type::internal) {
        for (size_t i{0}; i < in_channels; i++) {
            in_data[i].emplace<tensor*>(input[i]->get_data());
            in_grad[i].emplace<tensor*>(input[i]->get_grad());
        }

        for (size_t i{0}; i < out_channels; i++) {
            out_data[i].emplace<tensor*>(output[i]->get_data());
            out_grad[i].emplace<tensor*>(output[i]->get_grad());
        }

    } else if (backend == core::backend_type::opencl) {
        for (size_t i{0}; i < in_channels; i++) {
            in_data[i].emplace<cl::Buffer*>(input[i]->get_data_buffer());
            in_grad[i].emplace<cl::Buffer*>(input[i]->get_grad_buffer());
        }

        for (size_t i{0}; i < out_channels; i++) {
            out_data[i].emplace<cl::Buffer*>(output[i]->get_data_buffer());
            out_grad[i].emplace<cl::Buffer*>(output[i]->get_grad_buffer());
        }
    }

    backward_context.set_in_out(in_data, in_grad, out_data, out_grad);
    backward_context.set_engine(layer::engine());

    backward_kernel->compute(backward_context, eng);
}

} // namespace yonn

