#pragma once
#include <vector>
#include <memory>
#include <utility>
#include <type_traits>
#include "layer/layer.hh"
#include "tensor.hh"
#include "optimizer/optimizer.hh"

namespace yonn
{

template <class Derived>
struct nodes
{

    void add(layer&& l) { Derived::add(l); }

    auto forward(tensor const& first) -> tensor
    {
        return Derived::forward(first);
    }

    // TODO
    void backward(tensor const& first)
    {
        Derived::backward(first);
    }

    void update_weight(optimizer::optimizer* opt)
    {
        for (auto& l : all_nodes)
            l->update_wegith(opt);
    }

protected:

    template <class Layer>
    void emplace_back_impl(Layer&& l, std::true_type)
    {
        own_nodes.emplace_back(std::make_shared<Layer>(
            std::forward<Layer>(l)
        ));
        all_nodes.emplace_back(own_nodes.back());
    }

    template <class Layer>
    void emplace_back_impl(Layer&& l, std::false_type)
    {
        all_nodes.emplace_back(
            std::make_shared<Layer>(l)
        );
    }

    template <class Layer>
    void emplace_back(Layer&& l)
    {
        emplace_back_impl(
            std::forward<Layer>(l),
            typename std::is_rvalue_reference<decltype(l)>::type{}
        );
    }

    template <class T>
    void emplace_back(std::shared_ptr<T>& l)
    {
        all_nodes.emplace_back(l);
    }

    // TODO if the network is being frequently changed, maybe std::list<> is
    // better
    std::vector<std::shared_ptr<layer>> own_nodes;
    // TODO all_nodes use raw pointer?
    std::vector<std::shared_ptr<layer>> all_nodes;
};


// implementation of nodes<Derived>

} // namespace yonn

