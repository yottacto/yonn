#pragma once
#include <vector>
#include <memory>
#include "core/backend.hh"
#include "node.hh"
#include "type.hh"
#include "optimizer/optimizer.hh"

namespace yonn
{

struct layer : node
{
    layer(
        std::vector<data_type> const& in_types,
        std::vector<data_type> const& out_types,
        core::backend backend = core::backend::internal
    ) : node(in_types.size(), out_types.size()),
        in_types{in_types}, out_types{out_types},
        in_channels(in_types.size()), out_channels(out_types.size()),
        backend{backend}
    {
    }

    virtual ~layer() = default;

    virtual void forward_propagation()  = 0;
    virtual void backward_propagation() = 0;
    virtual auto input_shapes()  -> std::vector<shape3d_t> = 0;
    virtual auto output_shapes() -> std::vector<shape3d_t> = 0;
    virtual auto input_shape(size_t)  -> shape3d_t = 0;
    virtual auto output_shape(size_t) -> shape3d_t = 0;

    void allocate_output()
    {
        output[0] = std::make_shared<edge>(output_shape(0));
    }

    void allocate_input(shape3d_t const& shape)
    {
        if (!out_shapes.empty())
            return;
        in_shapes.emplace_back(shape);
        out_shapes.emplace_back(shape);
        input.resize(1);
        output.resize(1);
        input[0] = std::make_shared<edge>(shape);
    }

    auto engine() const -> core::backend { return backend; }

    void forward() { forward_propagation(); }
    void backward() { backward_propagation(); }

    void set_input_data(tensor const& input)
    {
        // TODO assume all needed memory allocated and for opencl need to
        // deal with seperately
        this->input[0]->data = input;
    }

    void set_output_grad(tensor const& grad)
    {
        // TODO assume all needed memory allocated and for opencl need to
        // deal with seperately
        output[0]->grad = grad;
    }

    void output_data(tensor& out)
    {
        out = output[0]->data;
    }

    auto get_input_data(size_t i) -> vec_t&
    {
        return input[i]->data[0];
    }

    // TODO init weight

    void update_weight(optimizer::optimizer* opt)
    {
        // TODO mark trainable for data, here the input[0] is not trainable
        for (size_t i{1}; i < in_types.size(); i++) {
            input[i]->merge_grads();
            auto& weight = get_input_data(i);
            opt->update(input[i]->grad[0], weight);
        }
    }

protected:
    std::vector<data_type> in_types;
    std::vector<data_type> out_types;
    // FIXME cosntruct these two first
    std::vector<shape3d_t> in_shapes;
    std::vector<shape3d_t> out_shapes;
    size_t in_channels;
    size_t out_channels;
    core::backend backend;
};

auto& operator<<(layer& lhs, layer& rhs)
{
    // TODO
    return lhs;
}

inline void connect(
    std::shared_ptr<layer>& prev, std::shared_ptr<layer>& next,
    size_t out_index = 0, size_t in_index = 0)
{
    // TODO assert prev->output_shape(head_index) == next->input_shape(tail_index)
    // TODO assume each layer all input channels already alloced
    // next->input[0] = std::make_shared<edge>(next->input_shape(tail_index));
    prev->allocate_output();
    next->allocate_input(prev->output_shape(0));
    prev->output[out_index] = next->input[in_index];
    next->input[in_index]->prev = prev;
    next->input[in_index]->next = next;
}

} // namespace yonn

