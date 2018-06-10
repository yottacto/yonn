#pragma once
#include <memory>
#include "layer.hh"
#include "type.hh"
#include "util/util.hh"
#include "core/framework/op-kernel.hh"
#include "core/kernel/average-pooling-op.hh"
#include "core/parameter/avg-pool-parameter.hh"

#include "../core/kernel/opencl/average-pooling.hh"

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
        bool has_bias = true,
        core::backend_type backend = core::layer_default_engine()
    ) :
        // FIXME layer need backend
        layer(std_input_types(has_bias), {data_type::data}),
        params(
            shape3d_t{width, height, in_channels},
            pool_width,
            pool_height,
            stride
        ),
        // FIXME op_kernel need context to constrct, in fact in order to
        // specify device and layer's params
        forward_kernel(new core::kernel::average_pooling_op(params, name())),
        backward_kernel(new core::kernel::average_pooling_grad_op(params, name()))
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
        size_t pool_size,
        core::backend_type backend = core::layer_default_engine(),
        bool has_bias = true
    ) : average_pooling_layer(
        width,
        height,
        in_channels,
        pool_size,
        pool_size,
        height == 1 ? 1 : pool_size,
        has_bias,
        backend
    ) {}

    average_pooling_layer(
        size_t width,
        size_t height,
        size_t in_channels,
        size_t pool_size,
        size_t stride,
        bool has_bias = true,
        core::backend_type backend = core::layer_default_engine()
    ) : average_pooling_layer(
        width,
        height,
        in_channels,
        pool_size,
        pool_size,
        stride,
        has_bias,
        backend
    ) {}

    auto name() const -> std::string override
    {
        return "average pooling layer";
    }

    auto kernel_code() const -> std::string
    {
        return opencl_kernel::avg_pool_kernel_code;
    }

    auto nd_size() const -> size_t
    {
        return output_shape(0).size() * batch_size;
    }

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

    void init_engine(
        core::backend_type const& backend,
        core::engine::engine_type& eng
    ) override;

    void forward_propagation(core::engine::engine_type& eng, bool united_backend) override;
    void backward_propagation(core::engine::engine_type& eng, bool united_backend) override;

// TODO uncomment
// private:
    core::avg_pool_parameter params;
    core::framework::op_kernel_context forward_context;
    core::framework::op_kernel_context backward_context;
    std::shared_ptr<core::framework::op_kernel> forward_kernel;
    std::shared_ptr<core::framework::op_kernel> backward_kernel;
};

void average_pooling_layer::init_engine(
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

void average_pooling_layer::forward_propagation(core::engine::engine_type& eng, bool united_backend)
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

    forward_context.set_in_out(in_data, out_data);
    forward_context.set_engine(layer::engine());

    forward_kernel->compute(forward_context, eng, united_backend);
}

void average_pooling_layer::backward_propagation(core::engine::engine_type& eng, bool united_backend)
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

