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
    template < class Error, class Optimizer>
    auto train(
        Optimizer& optimizer,
        std::vector<tensor> const& inputs,
        std::vector<tensor> const& desired_outputs,
        size_t batch_sze,
        int epoch
    ) -> bool;

    // TODO TensorIterator is actually std::vector<tensor>::iterator
    template < class Error, class Optimizer, class TensorIterator>
    void train_once<(
        Optimizer& optimizer,
        TensorIterator inputs,
        TensorIterator desired_outputs,
        size_t size
    );

    template <class Error, class Optimizer, class TensorIterator>
    void train_onebatch(
        Optimizer& optimizer,
        TensorIterator inputs,
        TensorIterator desired_outputs,
        size_t batch_size
    );

private:
    network_type net;
    std::vector<tensor> in_batch;
    std::vector<tensor> desired_out_batch;
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
    std::vector<tensor> const& inputs,
    std::vector<tensor> const& desired_outputs,
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
    class TensorIterator
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
            forward_propagation(*inputs),
            *desired_outputs
        );
        net.update_weights(optimizer);
    } else {
        train_onebatch<Error>(optimizer, inputs, desired_outputs, size);
    }
}

template <class Net>
template <class Error, class Optimizer, class TensorIterator>
void network<Net>::train_onebatch(
    Optimizer& optimizer,
    TensorIterator inputs,
    TensorIterator desired_outputs,
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

