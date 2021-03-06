#pragma once
#include <vector>
#include <memory>
#include "core/backend.hh"
#include "node.hh"
#include "type.hh"
#include "optimizer/optimizer.hh"

#include "util/util.hh"

namespace yonn
{

struct layer : node
{
    layer(
        std::vector<data_type> const& in_types,
        std::vector<data_type> const& out_types,
        core::backend_type backend = core::layer_default_engine()
    ) : node(in_types.size(), out_types.size()),
        in_types{in_types}, out_types{out_types},
        in_channels(in_types.size()), out_channels(out_types.size()),
        backend{backend}
    {
    }

    virtual ~layer() = default;

    virtual auto name() const -> std::string = 0;
    virtual auto fan_in_size()  const -> size_t = 0;
    virtual auto fan_out_size() const -> size_t = 0;

    virtual void forward_propagation(core::engine::engine_type& eng, bool united_backend = true) = 0;
    virtual void backward_propagation(core::engine::engine_type& eng, bool united_backend = true) = 0;

    void allocate_nsamples(size_t batch_size);

    virtual void allocate_nsamples_opencl(size_t batch_size, core::engine::opencl& e)
    {
        // in case the layer doesnt support opencl backend
        ignore(batch_size);
        ignore(e);
    }

    void allocate_output();
    void allocate_input(shape3d_t const& shape);

    // FIXME make this pure virtual
    virtual void init_engine(
        core::backend_type const& backend,
        core::engine::engine_type& eng
    ) = 0;

    auto engine() const -> core::backend_type { return backend; }
    void set_engine(core::backend_type const& backend) { this->backend = backend; }

    void forward(core::engine::engine_type& eng, bool united_backend = true) { forward_propagation(eng, united_backend); }
    void backward(core::engine::engine_type& eng, bool united_backend = true) { backward_propagation(eng, united_backend); }

    void set_input_data(std::vector<tensor> const& input, core::engine::engine_type& eng);
    void set_input_data(tensor const& input, core::engine::engine_type& eng);
    void set_output_grad(tensor const& grad, core::engine::engine_type& eng);
    void output_data(tensor& out, core::engine::engine_type& eng);
    auto get_input_data(size_t i) -> tensor&;
    auto get_input_grad() const -> std::vector<tensor>;

    auto get_output_data() const -> tensor;
    auto get_output_grad() const -> tensor;

    void reset_output_grad(value_type x);

    // TODO init weight

    void update_weight(optimizer::optimizer* opt, core::engine::engine_type& eng);

    auto input_shapes()         const -> std::vector<shape3d_t> { return in_shapes;     }
    auto input_shape(size_t i)  const -> shape3d_t              { return in_shapes[i];  }
    auto output_shapes()        const -> std::vector<shape3d_t> { return out_shapes;    }
    auto output_shape(size_t i) const -> shape3d_t              { return out_shapes[i]; }

// TODO uncomment
// protected:
    std::vector<data_type> in_types;
    std::vector<data_type> out_types;
    // FIXME cosntruct these two first
    std::vector<shape3d_t> in_shapes;
    std::vector<shape3d_t> out_shapes;
    size_t in_channels;
    size_t out_channels;
    core::backend_type backend;
    size_t batch_size;
};

// auto& operator<<(layer& lhs, layer& rhs)
// {
//     // TODO
//     return lhs;
// }

inline void connect(
    std::shared_ptr<layer>& prev, std::shared_ptr<layer>& next,
    size_t out_index = 0, size_t in_index = 0)
{
    // TODO assert prev->output_shape(head_index) == next->input_shape(tail_index)
    // TODO assume each layer all input channels already alloced
    // next->input[0] = std::make_shared<edge>(next->input_shape(tail_index));
    // prev->allocate_output();
    next->allocate_input(prev->output_shape(0));
    next->allocate_output();
    next->input[in_index] = prev->output[out_index];
    // prev->output[out_index] = next->input[in_index];
    // next->input[in_index]->prev = prev;
    // next->input[in_index]->next = next;
}

// implentation of layer
void layer::allocate_nsamples(size_t batch_size)
{
    this->batch_size = batch_size;
    // backend must be internal
    if (backend == core::backend_type::internal) {
        input[0]->allocate_nsamples(batch_size, input_shape(0));
        output[0]->allocate_nsamples(batch_size, output_shape(0));
    }
}

void layer::allocate_output()
{
    // FIXME
    if (!output[0])
        output[0] = std::make_shared<edge>();
}

void layer::allocate_input(shape3d_t const& shape)
{
    if (!in_shapes.empty())
        return;
    // for activation layer
    in_shapes.emplace_back(shape);
    out_shapes.emplace_back(shape);
    input[0] = std::make_shared<edge>();
}

void layer::set_input_data(std::vector<tensor> const& input, core::engine::engine_type& eng)
{
    if (backend == core::backend_type::internal) {
        for (auto i = 0u; i < input.size(); i++)
            this->input[i]->data = input[i];
    } else if (backend == core::backend_type::opencl) {
        auto& e = std::get<core::engine::opencl>(eng);
        for (auto i = 0u; i < input.size(); i++) {
            this->input[i]->data = input[i];
            this->input[i]->set_data(tensor_to_vector(input[i]), e);
        }
    }
}

void layer::set_input_data(tensor const& input, core::engine::engine_type& eng)
{
    if (backend == core::backend_type::internal) {
        this->input[0]->data = input;
    } else if (backend == core::backend_type::opencl){
        this->input[0]->data = input;
        auto& e = std::get<core::engine::opencl>(eng);
        this->input[0]->set_data(tensor_to_vector(input), e);
    }
}

void layer::set_output_grad(tensor const& grad, core::engine::engine_type& eng)
{
    if (backend == core::backend_type::internal) {
        output[0]->grad = grad;
    } else if (backend == core::backend_type::opencl) {
        output[0]->grad = grad;
        auto& e = std::get<core::engine::opencl>(eng);
        output[0]->set_grad(tensor_to_vector(grad), e);
    }
}

void layer::output_data(tensor& out, core::engine::engine_type& eng)
{
    if (backend == core::backend_type::internal) {
        out = output[0]->data;
    } else if (backend == core::backend_type::opencl) {
        auto& e = std::get<core::engine::opencl>(eng);
        vector_to_tensor(output[0]->get_data(e), output[0]->data);
        out = output[0]->data;
    }
}

auto layer::get_input_data(size_t i) -> tensor&
{
    return input[i]->data;
}

auto layer::get_input_grad() const -> std::vector<tensor>
{
    std::vector<tensor> grads(input.size());
    for (auto i = 0u; i < grads.size(); i++)
        grads[i] = input[i]->grad;
    return grads;
}

auto layer::get_output_data() const -> tensor
{
    return output[0]->data;
}

auto layer::get_output_grad() const -> tensor
{
    return output[0]->grad;
}

void layer::reset_output_grad(value_type x)
{
    for (auto& v: output[0]->grad)
        for (auto& i : v)
            i = x;
}

void layer::update_weight(optimizer::optimizer* opt, core::engine::engine_type& eng)
{
    if (backend == core::backend_type::internal) {
        // TODO mark trainable for data, here the input[0] is not trainable
        for (size_t i{1}; i < in_types.size(); i++) {
            // TODO deprecated? for now, dw and db only has one outside dimension,
            // not like input data
            // input[i]->merge_grads();
            auto& weight = get_input_data(i)[0];
            opt->update(input[i]->grad[0], weight);
        }
    } else if (backend == core::backend_type::opencl) {
        auto& e = std::get<core::engine::opencl>(eng);
        for (size_t i{1}; i < in_types.size(); i++) {
            auto& weight = *input[i]->get_data_buffer();
            auto& delta  = *input[i]->get_grad_buffer();
            // TODO use input_shapes?
            opt->update(delta, weight, e, input[i]->data[0].size());
        }
    }
}

} // namespace yonn

