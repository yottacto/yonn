#pragma once
#include <vector>
#include "nodes.hh"
#include "tensor.hh"
#include "optimizer/optimizer.hh"

namespace yonn
{
namespace topo
{

struct sequential : nodes<sequential>
{
    template <class Layer>
    void add(Layer&& l)
    {
        emplace_back(std::forward<Layer>(l));
        if (all_nodes.size() != 1) {
            connect(all_nodes[all_nodes.size() - 2], all_nodes.back());
        }
    }

    void allocate_nsamples(size_t batch_size);
    auto forward(tensor const& first) -> tensor;
    void backward(tensor const& first);
    void update_weight(optimizer::optimizer* opt);
};

void sequential::allocate_nsamples(size_t batch_size)
{
    for (auto const& l : all_nodes)
        l->allocate_nsamples(batch_size);
}

auto sequential::forward(tensor const& first) -> tensor
{
    all_nodes.front()->set_input_data(first);

    for (auto const& l : all_nodes)
        l->forward();

    tensor outt;
    // FIXME
    // TODO output channel index?
    all_nodes.back()->output_data(outt);
    // TODO normalize output
    return outt;
}

void sequential::backward(tensor const& first)
{
    own_nodes.back()->set_output_grad(first);

    for (auto l = all_nodes.crbegin(); l != all_nodes.crend(); ++l)
        (*l)->backward();
}

void sequential::update_weight(optimizer::optimizer* opt)
{
    for (auto& l : all_nodes)
        l->update_weight(opt);
}

} // namespace topo
} // namespace yonn

