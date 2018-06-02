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
    node(size_t in_size, size_t out_size)
        : input(in_size), output(out_size) {}

    virtual ~node() = default;

    friend void connect(
        std::shared_ptr<layer>& prev, std::shared_ptr<layer>& next,
        size_t out_index, size_t in_index
    );

// TODO uncomment
// protected:
    std::vector<std::shared_ptr<edge>> input;
    std::vector<std::shared_ptr<edge>> output;
};

struct edge
{
    edge() = default;

    //  FIXME tensor has outer vector wrapper
    edge(shape3d_t shape) : data(1, vec_t(shape.size())), grad(1, vec_t(shape.size()))
    {
    }

    void allocate_nsamples(size_t batch_size, shape3d_t shape)
    {
        data.resize(batch_size, vec_t(shape.size()));
        grad.resize(batch_size, vec_t(shape.size()));
    }

    auto get_data() -> tensor* { return &data; }
    auto get_grad() -> tensor* { return &grad; }

    // merge all grads to grad[0]
    void merge_grads()
    {
        // TODO parallelize
        for (size_t i{1}; i < grad.size(); i++)
            for (size_t j{0}; j < grad[0].size(); j++)
                grad[0][j] += grad[i][j];

        for (size_t j{0}; j < grad[0].size(); j++)
            grad[0][j] /= value_type(grad.size());
    }

    tensor data;
    tensor grad;
    std::weak_ptr<node> prev;
    std::weak_ptr<node> next;
};

} // namespace yonn

