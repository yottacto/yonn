#pragma once
#include <memory>
#include <CL/cl.hpp>
#include "tensor.hh"
#include "type.hh"

#include "core/backend.hh"

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

    template <class Context>
    edge(shape3d_t shape, Context const& context) :
        data(1, vec_t(shape.size())),
        grad(1, vec_t(shape.size()))
    {
        auto bsize = sizeof(value_type) * shape.size();
        data_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, bsize);
        grad_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, bsize);
    }

    void allocate_nsamples(size_t batch_size, shape3d_t shape)
    {
        data.resize(batch_size, vec_t(shape.size()));
        grad.resize(batch_size, vec_t(shape.size()));
    }

    template <class Context>
    void allocate_nsamples(size_t batch_size, shape3d_t shape, Context const& context)
    {
        auto bsize = sizeof(value_type) * batch_size * shape.size();
        data.resize(batch_size, vec_t(shape.size()));
        grad.resize(batch_size, vec_t(shape.size()));
        data_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, bsize);
        grad_buffer = cl::Buffer(context, CL_MEM_READ_WRITE, bsize);
    }

    auto get_data() -> tensor* { return &data; }
    auto get_grad() -> tensor* { return &grad; }
    auto get_data_buffer() -> cl::Buffer* { return &data_buffer; }
    auto get_grad_buffer() -> cl::Buffer* { return &grad_buffer; }

    void set_data(vec_t const& v, core::engine::opencl& e)
    {
        auto const bsize = sizeof(value_type) * v.size();
        e.queue.enqueueWriteBuffer(data_buffer, CL_TRUE, 0, bsize, v.data());
    }

    void set_grad(vec_t const& v, core::engine::opencl& e)
    {
        auto const bsize = sizeof(value_type) * v.size();
        e.queue.enqueueWriteBuffer(grad_buffer, CL_TRUE, 0, bsize, v.data());
    }

    auto get_data(core::engine::opencl& e) -> vec_t
    {
        vec_t v(data.size() * data[0].size());
        auto bsize = sizeof(value_type) * v.size();
        e.queue.enqueueReadBuffer(data_buffer, CL_TRUE, 0, bsize, v.data());
        return v;
    }

    auto get_grad(core::engine::opencl& e) -> vec_t
    {
        vec_t v(grad.size() * grad[0].size());
        auto bsize = sizeof(value_type) * v.size();
        e.queue.enqueueReadBuffer(grad_buffer, CL_TRUE, 0, bsize, v.data());
        return v;
    }

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
    cl::Buffer data_buffer;
    cl::Buffer grad_buffer;

    // TODO deprecated, remove this
    // std::weak_ptr<node> prev;
    // std::weak_ptr<node> next;
};

} // namespace yonn

