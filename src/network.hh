#pragma once
#include <vector>
#include <iterator>
#include <algorithm>
#include "layer/layer.hh"
#include "nodes.hh"

namespace yonn
{

template <class Net>
struct network
{
    using network_type = Net;

    void add(layer& l)
    {
        net.add(l);
    }

    // TODO callback function on each epoch and minibatch
    template <class Error, class Optimizer>
    auto train(
        Optimizer& optimizer,
        tensor const& inputs,
        vec_t const& desired_outputs,
        size_t batch_sze,
        int epoch
    ) -> bool;

    // TODO currently, output label was tansformed to tensor type
    // TODO TensorIterator is actually std::vector<tensor>::iterator, same for
    // OutIterator
    template <class Error, class Optimizer, class TensorIterator, class OutIterator>
    void train_once<(
        Optimizer& optimizer,
        TensorIterator inputs,
        OutIterator desired_outputs,
        size_t size
    );

    template <class Error, class Optimizer, class TensorIterator, class OutIterator>
    void train_onebatch(
        Optimizer& optimizer,
        TensorIterator inputs,
        OutIterator desired_outputs,
        size_t batch_size
    );

    // FIXME refactor tensor
    auto forward_propagation(std::vector<float> const& input) -> vec_t
    {
        return forward_propagation({input});
    }

    auto forward_propagation(tensor const& input) -> vec_t
    {
        return net.forward(input);
    }

    template <class Error>
    void backward_progapation(tensor const& output, vec_t const& desired_output)
    {
        tensor delta = gradient<Error>(output, desired_output);
        net.backward(delta);
    }

private:
    network_type net;
    tensor in_batch;
    vec_t desired_out_batch;
};

template <class Layer>
auto& operator<<(network<sequential>& net, Layer&& l)
{
    net.add(std::forward<Layer>(l));
    return net;
}


// implementation of network<Net>
// TODO callback function on each epoch and minibatch
template <class Net>
template <
    class Error,
    class Optimizer
>
auto network<Net>::train(
    Optimizer& optimizer,
    tensor const& inputs,
    vec_t const& desired_outputs,
    size_t batch_sze,
    int epoch
)
{
    // TODO network phase
    // TODO reset or init weight

    in_batch.resize(batch_size);
    desired_out_batch.resize(batch_size);

    for (auto round = 0; round < epoch; round++) {
        for (size_t i = 0; i < input.size(); i += batch_size) {
            train_once<Error>(
                optimizer,
                std::next(std::begin(inputs), i),
                std::next(std::begin(desired_outputs), i),
                std::min(batch_size, inputs.size() - i)
            );
        }
    }

    return true;
}

template <class Net>
template <
    class Error,
    class Optimizer,
    class TensorIterator,
    class OutIterator
>
void network<Net>::train_once<(
    Optimizer& optimizer,
    TensorIterator inputs,
    TensorIterator desired_outputs,
    size_t size
)
{
    if (size == 1) {
        back_propagation<Error>(
            forward_propagation({*inputs}),
            {*desired_outputs}
        );
        net.update_weights(optimizer);
    } else {
        train_onebatch<Error>(optimizer, inputs, desired_outputs, size);
    }
}

template <class Net>
template <class Error, class Optimizer, class TensorIterator, class OutIterator>
void network<Net>::train_onebatch(
    Optimizer& optimizer,
    TensorIterator inputs,
    OutIterator desired_outputs,
    size_t batch_size
)
{
    std::copy(inputs, std::next(inputs, batch_size), std::begin(in_batch));
    std::copy(
        desired_outputs,
        std::next(desired_outputs, batch_size),
        std::begin(desired_out_batch)
    );

    back_propagation<Error>(
        forward_propagation(in_bacth),
        desired_out_batch
    );
    net.update_weights(optimizer);
}

} // namespace yonn

