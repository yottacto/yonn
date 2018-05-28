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
        // TODO
    }

    void forward_propagation() override;
    void backward_propagation() override;

private:
    std::shared_ptr<core::framework::op_kernel> forward_kernel;
    std::shared_ptr<core::framework::op_kernel> backward_kernel;
};

} // namespace yonn

