#pragma once
#include <variant>
#include <memory>
#include "layer.hh"
#include "type.hh"
#include "util/util.hh"
#include "core/framework/op-kernel.hh"
#include "core/kernel/fully-connected-op.hh"
#include "core/parameter/fully-parameter.hh"

#include "core/kernel/opencl/fully-connected.hh"

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
        forward_kernel(new core::kernel::fully_connected_op(params, name())),
        backward_kernel(new core::kernel::fully_connected_grad_op(params, name()))
    {
        in_shapes.emplace_back(in_dims,  1,       1);
        in_shapes.emplace_back(out_dims, in_dims, 1);
        if (has_bias)
            in_shapes.emplace_back(out_dims, 1, 1);

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

    auto name() const -> std::string override
    {
        return "fully connected layer";
    }

    auto kernel_code() const -> std::string
    {
        return opencl_kernel::fully_kernel_code;
    }

    auto nd_size() const -> size_t
    {
        return output_shape(0).size() * batch_size;
    }

    auto fan_in_size() const -> size_t override
    {
        return params.in_size;
    }

    auto fan_out_size() const -> size_t override
    {
        return params.out_size;
    }

    void init_engine(
        core::backend_type const& backend,
        core::engine::engine_type& eng
    ) override;

    void forward_propagation(core::engine::engine_type& eng, bool united_backend) override;
    void backward_propagation(core::engine::engine_type& eng, bool united_backend) override;

// TODO uncomment
// private:
    core::fully_parameter params;
    core::framework::op_kernel_context forward_context;
    core::framework::op_kernel_context backward_context;
    std::unique_ptr<core::kernel::fully_connected_op> forward_kernel;
    std::unique_ptr<core::kernel::fully_connected_grad_op> backward_kernel;
};

void fully_connected_layer::init_engine(
    core::backend_type const& backend,
    core::engine::engine_type& eng
)
{
    // TODO this is copy from conv
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

        // TODO
    }
}


void fully_connected_layer::forward_propagation(core::engine::engine_type& eng, bool united_backend)
{
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


    forward_kernel->compute(forward_context, eng, united_backend);
}

void fully_connected_layer::backward_propagation(core::engine::engine_type& eng, bool united_backend)
{
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

    backward_kernel->compute(backward_context, eng, united_backend);
}

} // namespace yonn

