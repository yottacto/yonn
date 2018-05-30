#pragma once
#include <memory>
#include "layer.hh"
#include "type.hh"
#include "core/framework/op-kernel.hh"
#include "core/kernel/fully-connected-op.hh"
#include "core/parameter/fully-parameter.hh"

namespace yonn
{

struct fully_conneceted_layer : layer
{
    // TODO backend
    fully_conneceted_layer(size_t in_dims, size_t out_dims, bool has_bias = true) :
        layer(std_input_types(has_bias), {data_type::data}),
        params(in_dims, out_dims, has_bias),
        // FIXME op_kernel need context to constrct, in fact in order to
        // specify device and layer's params
        forward_kernel(new fully_connected_op(params)),
        backward_kernel(new fully_connected_grad_op(params)),
    {
        // invariant, all input channels allocated in constructor
        // TODO reasoning about this input_shape
        input[0] = std::make_shared<edge>(input_shape(0));
        input[1] = std::make_shared<edge>(input_shape(1));
        input[2] = std::make_shared<edge>(input_shape(2));

        // TODO init different kernel
    }

    void forward_propagation() override;
    void backward_propagation() override;
    auto input_shapes() -> std::vector<shape3d_t> override;
    auto input_shape(size_t) -> shape3d_t override;

private:
    core::fully_parameter params;
    core::framework::op_kernel_context forward_context;
    core::framework::op_kernel_context backward_context;
    std::shared_ptr<core::framework::op_kernel> forward_kernel;
    std::shared_ptr<core::framework::op_kernel> backward_kernel;
};

void fully_conneceted_layer::forward_propagation()
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

    forward_kernel.compute(forward_context);
}

void fully_conneceted_layer::backward_propagation()
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

    backward_kernel.compute(backward_context);
}

} // namespace yonn

