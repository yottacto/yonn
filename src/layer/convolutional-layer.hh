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
        core::backend_type backend,
        padding pad_type,
        bool has_bias,
        size_t w_stride,
        size_t h_stride
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

    // TODO deprecated
    auto kernel_code() const -> std::string
    {
        return opencl_kernel::conv_kernel_code;
    }

    auto nd_size() const -> std::vector<size_t>
    {
        std::vector<size_t> sizes{
            output_shape(0).size() * batch_size,
            output_shape(0).size() * batch_size,
            output_shape(0).size() * batch_size,
        };
        if (params.has_bias)
            sizes.emplace_back(output_shape(0).size() * batch_size);

        return sizes;
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

    void allocate_nsamples_opencl(size_t batch_size, core::engine::opencl& e) override;

    void forward_propagation(core::engine::engine_type& eng, bool united_backend) override;
    void backward_propagation(core::engine::engine_type& eng, bool united_backend) override;

// private:
    core::conv_parameter params;
    core::framework::op_kernel_context forward_context;
    core::framework::op_kernel_context backward_context;
    std::unique_ptr<core::kernel::convolutional_op> forward_kernel;
    std::unique_ptr<core::kernel::convolutional_grad_op> backward_kernel;
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
    core::backend_type backend = core::layer_default_engine()
) :
    layer(std_input_types(has_bias), {data_type::data}, backend),
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

    // TODO deprecated
    // TODO init different kernel
    // if (backend == core::backend_type::internal) {
    //     // invariant, all input channels allocated in constructor
    //     // TODO reasoning about this input_shape
    //     input[0] = std::make_shared<edge>();
    //     input[1] = std::make_shared<edge>(input_shape(1));
    //     input[2] = std::make_shared<edge>(input_shape(2));

    //     output[0] = std::make_shared<edge>();

    //     for (size_t i{0}; i < in_types.size(); i++)
    //         if (in_types[i] == data_type::weight)
    //             init_weight(input[i]->data[0], fan_in_size(), fan_out_size());
    // } else if (backend == core::backend_type::opencl) {
    //     // wait for opencl context, init in init_engine
    // } else {
    // }

}

convolutional_layer::convolutional_layer(
    size_t in_width,
    size_t in_height,
    size_t window_size,
    size_t in_channels,
    size_t out_channels,
    core::backend_type backend = core::layer_default_engine(),
    padding pad_type = padding::valid,
    bool has_bias = true,
    size_t w_stride = 1,
    size_t h_stride = 1
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
    padding pad_type = padding::valid,
    bool has_bias = true,
    size_t w_stride = 1,
    size_t h_stride = 1,
    core::backend_type backend = core::layer_default_engine()
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
    core::backend_type backend = core::layer_default_engine()
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

    if (backend == core::backend_type::internal) {
        // invariant, all input channels allocated in constructor
        // TODO reasoning about this input_shape
        input[0] = std::make_shared<edge>();
        input[1] = std::make_shared<edge>(input_shape(1));
        input[2] = std::make_shared<edge>(input_shape(2));

        output[0] = std::make_shared<edge>();

        for (size_t i{0}; i < in_types.size(); i++)
            if (in_types[i] == data_type::weight)
                init_weight(input[i]->data[0], fan_in_size(), fan_out_size());
    } else if (backend == core::backend_type::opencl) {
        auto& e = std::get<core::engine::opencl>(eng);
        input[0] = std::make_shared<edge>();
        input[1] = std::make_shared<edge>(input_shape(1), e.context);
        input[2] = std::make_shared<edge>(input_shape(2), e.context);

        output[0] = std::make_shared<edge>();

        for (size_t i{0}; i < in_types.size(); i++)
            if (in_types[i] == data_type::weight)
                init_weight(input[i]->data[0], fan_in_size(), fan_out_size());

        for (auto i = 1u; i < input.size(); i++)
            this->input[i]->set_data(tensor_to_vector(input[i]->data), e);

        forward_kernel->init_opencl_kernel(e);
        backward_kernel->init_opencl_kernel(e);
    }
}

void convolutional_layer::allocate_nsamples_opencl(size_t batch_size, core::engine::opencl& e)
{
    this->batch_size = batch_size;
    if (backend == core::backend_type::opencl) {
        input[0]->allocate_nsamples(batch_size, input_shape(0), e.context);
        output[0]->allocate_nsamples(batch_size, output_shape(0), e.context);

        forward_kernel->init_opencl(e, batch_size * params.out.size(), batch_size);
        backward_kernel->init_opencl(e, {
            batch_size * params.in_padded.size(),
            params.weight.size(),
            params.out.depth,
        }, batch_size);
    } else {
        // TODO error or currently not supportted backend
    }
}

void convolutional_layer::forward_propagation(core::engine::engine_type& eng, bool united_backend = true)
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
        if (!united_backend) {
            auto& e = std::get<core::engine::opencl>(eng);
            for (size_t i{0}; i < in_channels; i++)
                input[i]->set_data(tensor_to_vector(input[i]->data), e);
        }

        for (size_t i{0}; i < in_channels; i++)
            in_data[i].emplace<cl::Buffer*>(input[i]->get_data_buffer());

        for (size_t i{0}; i < out_channels; i++)
            out_data[i].emplace<cl::Buffer*>(output[i]->get_data_buffer());
    }

    forward_context.set_in_out(in_data, out_data);
    forward_context.set_engine(layer::engine());

    forward_kernel->compute(forward_context, eng, united_backend);

    if (backend == core::backend_type::opencl) {
        if (!united_backend) {
            auto& e = std::get<core::engine::opencl>(eng);
            for (size_t i{0}; i < out_channels; i++)
                vector_to_tensor(output[i]->get_data(e), output[i]->data);
        }
    }
}

void convolutional_layer::backward_propagation(core::engine::engine_type& eng, bool united_backend = true)
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
        if (!united_backend) {
            auto& e = std::get<core::engine::opencl>(eng);
            for (size_t i{0}; i < out_channels; i++) {
                output[i]->set_data(tensor_to_vector(output[i]->data), e);
                output[i]->set_grad(tensor_to_vector(output[i]->grad), e);
            }
        }

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

    backward_kernel->compute(backward_context, eng, united_backend);

    if (backend == core::backend_type::opencl) {
        if (!united_backend) {
            auto& e = std::get<core::engine::opencl>(eng);
            for (size_t i{0}; i < in_channels; i++) {
                vector_to_tensor(input[i]->get_data(e), input[i]->data);
                vector_to_tensor(input[i]->get_grad(e), input[i]->grad);
            }
        }
    }
}

} // namespace yonn

