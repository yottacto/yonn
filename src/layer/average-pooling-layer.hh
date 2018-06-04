#pragma once
#include <memory>
#include "layer.hh"
#include "type.hh"
#include "util/util.hh"
#include "core/framework/op-kernel.hh"
#include "core/kernel/average-pooling-op.hh"
#include "core/parameter/avg-pool-parameter.hh"

namespace yonn
{

struct average_pooling_layer : layer
{
    // TODO backend
    average_pooling_layer(
        size_t width,
        size_t height,
        size_t in_channels,
        size_t pool_width,
        size_t pool_height,
        size_t stride,
        bool has_bias = true
    ) :
        layer(std_input_types(has_bias), {data_type::data}),
        params(
            shape3d_t{width, height, in_channels},
            pool_width,
            pool_height,
            stride
        ),
        // FIXME op_kernel need context to constrct, in fact in order to
        // specify device and layer's params
        forward_kernel(new core::kernel::average_pooling_op(params)),
        backward_kernel(new core::kernel::average_pooling_grad_op(params))
    {
        in_shapes.emplace_back(params.in.size(), 1, 1);
        in_shapes.emplace_back(params.in.size(), 1, 1);
        in_shapes.emplace_back(params.in.size(), 1, 1);

        out_shapes.emplace_back(params.out.size(), 1, 1);

        // invariant, all input channels allocated in constructor
        // TODO reasoning about this input_shape
        input[0] = std::make_shared<edge>(input_shape(0));
        input[1] = std::make_shared<edge>(input_shape(1));
        input[2] = std::make_shared<edge>(input_shape(2));

        output[0] = std::make_shared<edge>(output_shape(0));

        // TODO init different kernel

        for (size_t i{0}; i < in_types.size(); i++)
            if (in_types[i] == data_type::weight)
                init_weight(input[i]->data[0], fan_in_size(), fan_out_size());
    }

    average_pooling_layer(
        size_t width,
        size_t height,
        size_t in_channels,
        size_t pool_size
    ) : average_pooling_layer(
        width,
        height,
        in_channels,
        pool_size,
        pool_size,
        height == 1 ? 1 : pool_size
    ) {}

    average_pooling_layer(
        size_t width,
        size_t height,
        size_t in_channels,
        size_t pool_size,
        size_t stride
    ) : average_pooling_layer(
        width,
        height,
        in_channels,
        pool_size,
        pool_size,
        stride
    ) {}

    auto fan_in_size() const -> size_t override
    {
        return params.pool_width * params.pool_height;
    }

    // TODO actually without multiplying depth, but this seems better?
    auto fan_out_size() const -> size_t override
    {
        return (params.pool_width / params.stride)
            * (params.pool_height / params.stride);
    }

    void forward_propagation() override;
    void backward_propagation() override;

// TODO uncomment
// private:
    core::avg_pool_parameter params;
    core::framework::op_kernel_context forward_context;
    core::framework::op_kernel_context backward_context;
    std::shared_ptr<core::framework::op_kernel> forward_kernel;
    std::shared_ptr<core::framework::op_kernel> backward_kernel;
};

void average_pooling_layer::forward_propagation()
{
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

void average_pooling_layer::backward_propagation()
{
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

