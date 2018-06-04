#pragma once
#include "type.hh"
#include "absolute.hh"
#include "mse.hh"
#include "softmax.hh"

namespace yonn
{
namespace loss_function
{

// TODO opencl backend

template <class Error>
auto gradient(vec_t const& scores, label_t y)
{
    return Error::df(scores, y);
}

template <class Error>
auto gradient(tensor const& scores, std::vector<label_t> const& y) -> tensor
{
    tensor grads(scores.size());
    for (size_t i{0}; i < scores.size(); i++)
        grads[i] = gradient<Error>(scores[i], y[i]);
        // grads[i] = gradient<Error>(scores[i], y[i])
        //     / static_cast<value_type>(scores.size());
    return grads;
}

} // namespace loss_function
} // namespace yonn

