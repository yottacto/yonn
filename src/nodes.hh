#pragma once
#include <vector>
#include <memory>
#include <utility>
#include <type_traits>
#include "layer/layer.hh"

namespace yonn
{

template <class Derived>
struct nodes
{

    void add(layer&& l) { Derived::add(l); }
    void forward() { Derived::forward(); }
    void backward() { Derived::backward(); }

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
};

// implementation of nodes<Derived>


} // namespace yonn

