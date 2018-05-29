#pragma once
#include <memory>
#include "layer.hh"
#include "type.hh"
#include "core/framework/op-kernel.hh"

namespace yonn
{

struct fully_conneceted_layer : layer
{
    fully_conneceted_layer(size_t in_dims, size_t out_dims, bool has_bias = true)
        : layer(std_input_types(has_bias), {data_type::data})
    {
        // invariant, all input channels allocated in constructor
        // TODO reasoning about this input_shape
        input[0] = std::make_shared<edge>(input_shape(0));
        input[1] = std::make_shared<edge>(input_shape(1));
        input[2] = std::make_shared<edge>(input_shape(2));
    }

    void forward_propagation() override;
    void backward_propagation() override;
    auto input_shapes() -> std::vector<shape3d_t> override;
    auto input_shape(size_t) -> shape3d_t override;

private:
    core::framework::op_kernel_context forward_context;
    core::framework::op_kernel_context backward_context;
    std::shared_ptr<core::framework::op_kernel> forward_kernel;
    std::shared_ptr<core::framework::op_kernel> backward_kernel;
};

void fully_conneceted_layer::forward_propagation()
{
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

} // namespace yonn

