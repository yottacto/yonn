#pragma once
#include <memory>
#include "layer.hh"
#include "type.hh"
#include "util.hh"
#include "core/framework/op-kernel.hh"
#include "core/kernel/fully-connected-op.hh"
#include "core/parameter/fully-parameter.hh"

namespace yonn
{

struct fully_connected_layer : layer
{
    // TODO backend
    fully_connected_layer(size_t in_dims, size_t out_dims, bool has_bias = true) :
        layer(std_input_types(has_bias), {data_type::data}),
        params(in_dims, out_dims, has_bias),
        // FIXME op_kernel need context to constrct, in fact in order to
        // specify device and layer's params
        forward_kernel(new core::kernel::fully_connected_op(params)),
        backward_kernel(new core::kernel::fully_connected_grad_op(params))
    {
        in_shapes.emplace_back(in_dims,  1,       1);
        in_shapes.emplace_back(out_dims, in_dims, 1);
        if (has_bias)
            in_shapes.emplace_back(out_dims, 1,       1);

        out_shapes.emplace_back(out_dims, 1, 1);

        // invariant, all input channels allocated in constructor
        // TODO reasoning about this input_shape
        input[0] = std::make_shared<edge>();
        input[1] = std::make_shared<edge>(input_shape(1));
        if (has_bias)
            input[2] = std::make_shared<edge>((input_shape(2)));

        output[0] = std::make_shared<edge>();

        // TODO init different kernel

        // TODO init weight
        for (size_t i{0}; i < in_types.size(); i++)
            if (in_types[i] == data_type::weight)
                init_weight(input[i]->data[0], fan_in_size(), fan_out_size());
    }

    auto fan_in_size() const -> size_t override
    {
        return params.in_size;
    }

    auto fan_out_size() const -> size_t override
    {
        return params.out_size;
    }

    void forward_propagation() override;
    void backward_propagation() override;

private:
    core::fully_parameter params;
    core::framework::op_kernel_context forward_context;
    core::framework::op_kernel_context backward_context;
    std::shared_ptr<core::framework::op_kernel> forward_kernel;
    std::shared_ptr<core::framework::op_kernel> backward_kernel;
};

void fully_connected_layer::forward_propagation()
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

void fully_connected_layer::backward_propagation()
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

