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
        size_t out_index, size_t in_index
    );

protected:
    std::vector<std::shared_ptr<edge>> input;
    std::vector<std::shared_ptr<edge>> output;
};

struct edge
{
    //  FIXME tensor has outer vector wrapper
    edge(shape3d_t shape) : data(shape.size()), grad(shape.size())
    {
    }

    tensor data;
    tensor grad;
    std::weak_ptr<node> prev;
    std::weak_ptr<node> next;
};

} // namespace yonn

