#pragma once
#include <memory>
#include "tensor.hh"
#include "type.hh"

namespace yonn
{

struct layer;
struct edge;

struct node
{
    virtual void compute() = 0;

    virtual ~node() = default;

    friend void connect(
        std::shared_ptr<layer>& prev, std::shared_ptr<layer>& next,
        size_t head_index, size_t tail_index
    );

protected:
    std::shared_ptr<edge> prev;
    std::shared_ptr<edge> next;
};

struct edge
{
    tensor data;
    tensor grad;
    std::weak_ptr<node> prev;
    std::weak_ptr<node> next;
};

} // namespace yonn

