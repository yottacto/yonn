#pragma once
// TODO remove this
#include <iostream>

#include <any>
#include <variant>
#include <vector>
#include <iterator>
#include <algorithm>
#include "nodes.hh"
#include "layer/layer.hh"
#include "loss-function/loss-function.hh"
#include "topo/sequential.hh"
#include "util/util.hh"
#include "util/timer.hh"

namespace yonn
{

template <class Net>
struct network
{
    using network_type = Net;

    network(core::backend_type backend = core::default_engine())
        : backend{backend}, net{backend}
    {
    }

    void allocate_nsamples(size_t batch_size)
    {
        in_batch.resize(batch_size);
        desired_out_batch.resize(batch_size);

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

    // TODO refactor tensor
    auto forward_propagation(vec_t const& input) -> std::variant<tensor*, cl::Buffer*>
    {
        return forward_propagation(tensor{input});
    }

    auto forward_propagation(tensor const& input) -> std::variant<tensor*, cl::Buffer*>
    {
        forward_timer.start();
        auto ret = net.forward(input);
        forward_timer.stop();
        return ret;
    }

    auto forward_prop_max_index(vec_t const& input) -> label_t
    {
        if (backend == core::backend_type::internal) {
            return max_index((*std::get<tensor*>(forward_propagation(input)))[0]);
        } else if (backend == core::backend_type::opencl) {
            forward_propagation(input);
            return max_index(net.get_output_tensor()[0]);
        } else {
            // TODO
            throw;
        }
    }

    template <class Error>
    void backward_propagation(
        std::variant<tensor*, cl::Buffer*> const& output,
        std::vector<label_t> const& desired_output
    )
    {
        if (backend == core::backend_type::internal) {
            using data_type = tensor;
            auto& out = std::get<data_type*>(output);
            tensor delta = loss_function::gradient<Error>(*out, desired_output);
            backward_timer.start();
            net.backward(delta);
            backward_timer.stop();
        } else if (backend == core::backend_type::opencl) {
            using data_type = cl::Buffer;
            auto& out = *std::get<data_type*>(output);
            auto& e = std::get<core::engine::opencl>(net.eng);
            auto out_size = net.out_size();
            auto& grad_buffer = net.all_nodes.back()->output[0]->grad_buffer;
            backward_timer.start();
            std::any_cast<loss_function::opencl_gradient<Error>>(cl_gradient)
                .gradient(
                    out,
                    out_size,
                    desired_output,
                    grad_buffer,
                    e
                );
            net.backward(grad_buffer);
            backward_timer.stop();
        }
    }

    template <class EachTest>
    auto test(
        tensor const& inputs,
        std::vector<label_t> const& desired_outputs,
        EachTest each_test
    ) -> result
    {
        allocate_nsamples(1);

        result res;
        for (size_t sample{0}; sample < inputs.size(); sample++) {
            auto const predicted = forward_prop_max_index(inputs[sample]);
            auto const actual    = desired_outputs[sample];
            res.insert(predicted, actual);
            each_test(false);
        }
        each_test(true);
        return res;
    }

    void print_out_shapes() const
    {
        net.print_out_shapes();
    }

// TODO uncomment
// private:
    core::backend_type backend;
    network_type net;
    tensor in_batch;
    std::vector<label_t> desired_out_batch;
    std::any cl_gradient;

    util::timer forward_timer;
    util::timer backward_timer;
    util::timer update_weight_timer;
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

    allocate_nsamples(batch_size);

    auto& e = std::get<core::engine::opencl>(net.eng);

    loss_function::opencl_gradient<Error> cl_g{};
    cl_g.allocate_output(batch_size, e);

    cl_gradient.emplace<loss_function::opencl_gradient<Error>>(cl_g);

    for (auto round = 0; round < epoch; round++) {
        each_epoch(false);
        for (size_t i{0}; i < inputs.size(); i += batch_size) {
            train_once<Error>(
                optimizer,
                std::next(std::begin(inputs), i),
                std::next(std::begin(desired_outputs), i),
                std::min<size_t>(batch_size, inputs.size() - i)
            );
            each_batch(false);
        }
        each_batch(true);

        std::cerr << "forward:\t\t"
            << forward_timer.elapsed_seconds() << "s.\n";
        std::cerr << "backward:\t\t"
            << backward_timer.elapsed_seconds() << "s.\n";
        std::cerr << "weight update:\t\t"
            << update_weight_timer.elapsed_seconds() << "s.\n";
        forward_timer.reset();
        backward_timer.reset();
        update_weight_timer.reset();
    }
    each_epoch(true);

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
        update_weight_timer.start();
        net.update_weight(&optimizer);
        update_weight_timer.stop();
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
    update_weight_timer.start();
    net.update_weight(&optimizer);
    update_weight_timer.stop();
}

} // namespace yonn

