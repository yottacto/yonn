#pragma once
#include <cmath>
#include "layer/layer.hh"
#include "util/util.hh"
#include "tensor.hh"

#include "core/kernel/opencl/leaky-relu.hh"

namespace yonn
{
namespace activation
{

struct leaky_relu : layer
{
    explicit leaky_relu(value_type epsilon = 0.01)
        : layer({data_type::data}, {data_type::data}), epsilon{epsilon}
    {}

    // TODO explicit specify the dims

    auto name() const -> std::string override
    {
        return "leaky relu layer";
    }

    auto kernel_code() const -> std::string
    {
        return opencl_kernel::leaky_relu_kernel_code;
    }

    auto nd_size() const -> size_t
    {
        return output_shape(0).size() * batch_size;
    }

    auto fan_in_size() const -> size_t override
    {
        return input_shape(0).size();
    }

    auto fan_out_size() const -> size_t override
    {
        return output_shape(0).size();
    }

    void init_engine(
        core::backend_type const& backend,
        core::engine::engine_type& eng
    ) override;

    void forward_propagation(core::engine::engine_type& eng) override;
    void backward_propagation(core::engine::engine_type& eng) override;

    void forward_activation(vec_t const& in, vec_t& out);
    void backward_activation(vec_t const& x, vec_t& dx, vec_t const& y, vec_t const& dy);


    value_type epsilon;
};


// FIXME this is copy from conv
void leaky_relu::init_engine(
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


void leaky_relu::forward_propagation(core::engine::engine_type& eng)
{
    // TODO init once
    // TODO const in data?
    tensor const& in_data  = *(input[0] ->get_data());
    tensor&       out_data = *(output[0]->get_data());

    for (size_t sample{0}; sample < in_data.size(); sample++)
        forward_activation(in_data[sample], out_data[sample]);
}

void leaky_relu::backward_propagation(core::engine::engine_type& eng)
{
    tensor const& in_data  = *(input[0] ->get_data());
    tensor&       in_grad  = *(input[0] ->get_grad());
    tensor const& out_data = *(output[0]->get_data());
    tensor const& out_grad = *(output[0]->get_grad());

    for (size_t sample{0}; sample < in_data.size(); sample++)
        backward_activation(
            in_data[sample],  in_grad[sample],
            out_data[sample], out_grad[sample]
        );
}

void leaky_relu::forward_activation(vec_t const& in, vec_t& out)
{
    #if USE_OPENMP
    #pragma omp for
    #endif
    for (size_t i = 0; i < in.size(); i++)
        out[i] = in[i] > value_type{0} ? in[i] : epsilon * in[i];
}

void leaky_relu::backward_activation(vec_t const& x, vec_t& dx, vec_t const& y, vec_t const& dy)
{
    #if USE_OPENMP
    #pragma omp for
    #endif
    for (size_t i = 0; i < x.size(); i++)
        dx[i] = dy[i] * (y[i] > value_type{0} ? value_type(1) : epsilon);
}

} // namespace activation
} // namespace yonn

