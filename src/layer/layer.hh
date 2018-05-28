#pragma once
#include <vector>
#include "node.hh"
#include "type.hh"

namespace yonn
{

struct layer : node
{
    layer(
        std::vector<data_type> const& in_types,
        std::vector<data_type> const& out_typos
    ) : in_types{in_types}, out_types{out_types}
    {
    }

    virtual ~layer() = default;

    virtual void forward_propagation()  = 0;
    virtual void backward_propagation() = 0;
    virtual auto in_shape()  -> std::vector<shape3d_t> = 0;
    virtual auto out_shape() -> std::vector<shape3d_t> = 0;

protected:
    std::vector<data_type> in_types;
    std::vector<data_type> out_types;
};

auto& operator<<(layer& lhs, layer& rhs)
{
    // TODO
    return lhs;
}

inline void connect(
    std::shared_ptr<layer>& prev, std::shared_ptr<layer>& next,
    size_t head_index = 0, size_t tail_index = 0)
{
    next->prev = std::make_shared<edge>();
    prev->next = next->prev;
    next->prev->prev = prev;
    next->prev->next = next;
}

} // namespace yonn

