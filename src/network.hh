#pragma once
#include <vector>
#include <iterator>
#include <algorithm>
#include "nodes.hh"
#include "layer/layer.hh"
#include "loss-function/loss-function.hh"
#include "topo/sequential.hh"
#include "util/util.hh"

namespace yonn
{

template <class Net>
struct network
{
    using network_type = Net;

    void allocate_nsamples(size_t batch_size)
    {
        net.allocate_nsamples(batch_size);
    }

    template <class Layer>
    void add(Layer&& l)
    {
        net.add(std::forward<Layer>(l));
    }

    // TODO callback function on each epoch and minibatch
    template <class Error, class Optimizer, class EachBatch, class EachEpoch>
    auto train(
        Optimizer& optimizer,
        tensor const& inputs,
        std::vector<label_t> const& desired_outputs,
        size_t batch_sze,
        int epoch,
        EachBatch each_batch,
        EachEpoch each_epoch
    ) -> bool;

    // TODO currently, output label was tansformed to tensor type
    // TODO TensorIterator is actually std::vector<tensor>::iterator, same for
    // OutIterator
    template <class Error, class Optimizer, class TensorIterator, class OutIterator>
    void train_once(
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
    auto forward_propagation(vec_t const& input) -> tensor
    {
        return forward_propagation(tensor{input});
    }

    auto forward_propagation(tensor const& input) -> tensor
    {
        return net.forward(input);
    }

    auto forward_prop_max_index(vec_t const& input) -> label_t
    {
        return max_index(forward_propagation(input)[0]);
    }

    template <class Error>
    void backward_propagation(
        tensor const& output,
        std::vector<label_t> const& desired_output
    )
    {
        tensor delta = loss_function::gradient<Error>(output, desired_output);
        net.backward(delta);
    }

    auto test(
        tensor const& inputs,
        std::vector<label_t> const& desired_outputs
    ) -> result
    {
        result res;
        for (size_t sample{0}; sample < inputs.size(); sample++) {
            auto const predicted = forward_prop_max_index(inputs[sample]);
            auto const actual    = desired_outputs[sample];
            res.insert(predicted, actual);
        }
        return res;
    }

    void print_out_shapes() const
    {
        net.print_out_shapes();
    }

// TODO uncomment
// private:
    network_type net;
    tensor in_batch;
    std::vector<label_t> desired_out_batch;
};

template <class Layer>
auto& operator<<(network<topo::sequential>& net, Layer&& l)
{
    net.add(std::forward<Layer>(l));
    return net;
}


// implementation of network<Net>
template <class Net>
template <
    class Error,
    class Optimizer,
    class EachBatch,
    class EachEpoch
>
auto network<Net>::train(
    Optimizer& optimizer,
    tensor const& inputs,
    std::vector<label_t> const& desired_outputs,
    size_t batch_size,
    int epoch,
    EachBatch each_batch,
    EachEpoch each_epoch
) -> bool
{
    // TODO network phase
    // TODO reset or init weight

    in_batch.resize(batch_size);
    desired_out_batch.resize(batch_size);

    allocate_nsamples(batch_size);

    for (auto round = 0; round < epoch; round++) {
        for (size_t i{0}; i < inputs.size(); i += batch_size) {
            train_once<Error>(
                optimizer,
                std::next(std::begin(inputs), i),
                std::next(std::begin(desired_outputs), i),
                std::min<size_t>(batch_size, inputs.size() - i)
            );
            each_batch();
        }
        each_epoch();
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
void network<Net>::train_once(
    Optimizer& optimizer,
    TensorIterator inputs,
    OutIterator desired_outputs,
    size_t size
)
{
    if (size == 1) {
        backward_propagation<Error>(
            forward_propagation({*inputs}),
            {*desired_outputs}
        );
        net.update_weight(&optimizer);
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

    backward_propagation<Error>(
        forward_propagation(in_batch),
        desired_out_batch
    );
    net.update_weight(&optimizer);
}

} // namespace yonn

