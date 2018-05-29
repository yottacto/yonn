#pragma once
#include <vector>
#include "nodes.hh"
#include "tensor.hh"

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

    auto forward(std::vector<tensor> const& first) -> std::vector<tensor>;
};

auto sequential::forward(std::vector<tensor> const& first)
{
}

} // namespace topo
} // namespace yonn

