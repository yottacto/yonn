#pragma once
#include <cmath>
#include "layer/layer.hh"
#include "tensor.hh"

namespace yonn
{
namespace activation
{

struct tanh : layer
{
    void forward_propagation() override;
    void backward_propagation() override;

    void forward_activation(vec_t const& in, vec_t& out);
    void backward_activation(vec_t const& x, vec_t& dx, vec_t const& y, vec_t const& dy);
};

void tanh::forward_propagation()
{
    // TODO init once
    // TODO const in data?
    tensor* in_data  = input[0] ->get_data();
    tensor* out_data = output[0]->get_data();

    for (size_t sample{0}; sample < in_data->size(); sample++)
        forward_activation(in_data[sample], out_data[sample]);
}

void tanh::backward_propagation()
{
    tensor* in_data  = input[0] ->get_data();
    tensor* in_grad  = input[0] ->get_grad();
    tensor* out_data = output[0]->get_data();
    tensor* out_grad = output[0]->get_grad();

    for (size_t sample{0}; sample < in_data->size(); sample++)
        backward_activation(
            in_data[sample],  in_grad[sample],
            out_data[sample], out_grad[sample]
        );
}

void tanh::forward_activation(vec_t const& in, vec_t& out)
{
    for (size_t i{0}; i < in.size(); i++)
        out[i] = std::tanh(in[i]);
}

void tanh::backward_activation(vec_t const& x, vec_t& dx, vec_t const& y, vec_t const& dy)
{
    for (size_t i{0}; i < x.size(); i++)
        dx[i] = dy[i] * (value_type(1) - sqr(y[i]));
}

} // namespace activation
} // namespace yonn

