#pragma once

// TODO remove this
#include <iostream>

#include <vector>
#include <variant>
#include <CL/cl.hpp>
#include "nodes.hh"
#include "tensor.hh"
#include "optimizer/optimizer.hh"
#include "core/backend.hh"

#include "util/timer.hh"

namespace yonn
{
namespace topo
{

struct sequential : nodes<sequential>
{
    sequential() = default;
    sequential(core::backend_type backend)
        : backend{backend}
    {
        if (backend == core::backend_type::internal)
            ;
        else if (backend == core::backend_type::opencl) {
            eng.emplace<core::engine::opencl>();
        } else {
            // TODO unsupported backend or network backend cannot be
            // network_default
        }
    }


    template <class Layer>
    void add(Layer&& l)
    {
        emplace_back(std::forward<Layer>(l));

        auto& last = *all_nodes.back();

        if (last.engine() == core::backend_type::network_default)
            last.init_engine(backend, eng);
        else
            last.init_engine(last.engine(), eng);

        if (last.engine() != backend)
            united_backend = false;

        if (all_nodes.size() != 1) {
            connect(all_nodes[all_nodes.size() - 2], all_nodes.back());
        }
    }

    void allocate_nsamples(size_t batch_size);
    auto get_output_tensor() -> tensor;

    auto forward(tensor const& first) -> std::variant<tensor*, cl::Buffer*>;
    void backward(tensor const& first);
    void backward(cl::Buffer&);

    void update_weight(optimizer::optimizer* opt);

    auto out_size() const -> size_t
    {
        return all_nodes.back()->output[0]->data[0].size();
    }

    // debug info
    // TODO pass outstream
    void print_out_shapes() const
    {
        for (auto const& l : all_nodes) {
            auto out_shape = l->output_shape(0);
            std::cerr << "(" << out_shape.width
                << ", " << out_shape.height
                << ", " << out_shape.depth << ")\n";
            std::cerr << l->output[0]->data.size() << "\n";
            std::cerr << l->output[0]->data[0].size() << "\n";
        }
    }

// TODO uncoment this
// private:
    core::backend_type backend;
    core::engine::engine_type eng;
    bool united_backend{true};

    // TODO return value of forward(...), in case of dangling pointer
    tensor out_data;
};

void sequential::allocate_nsamples(size_t batch_size)
{
    for (auto const& l : all_nodes) {
        auto backend = l->engine();
        if (backend == core::backend_type::internal) {
            l->allocate_nsamples(batch_size);
        } else if (backend == core::backend_type::opencl) {
            auto& e = std::get<core::engine::opencl>(eng);
            l->allocate_nsamples_opencl(batch_size, e);
        }
    }
}

auto sequential::get_output_tensor() -> tensor
{
    tensor out_data;
    all_nodes.back()->output_data(out_data, eng);
    return out_data;
}

auto sequential::forward(tensor const& first) -> std::variant<tensor*, cl::Buffer*>
{
    all_nodes.front()->set_input_data(first, eng);

    for (auto const& l : all_nodes)
        l->forward(eng, united_backend);

    std::variant<tensor*, cl::Buffer*> out;
    if (backend == core::backend_type::internal) {
        // FIXME
        // TODO output channel index?
        all_nodes.back()->output_data(out_data, eng);
        // TODO normalize output
        out.emplace<tensor*>(&out_data);
    } else if (backend == core::backend_type::opencl) {
        // FIXME access private member?
        out.emplace<cl::Buffer*>(all_nodes.back()->output[0]->get_data_buffer());
    }
    return out;
}

void sequential::backward(tensor const& first)
{
    own_nodes.back()->set_output_grad(first, eng);

    for (auto l = all_nodes.crbegin(); l != all_nodes.crend(); ++l)
        (*l)->backward(eng, united_backend);
}

void sequential::backward(cl::Buffer&)
{
    // own_nodes.back()->output[0]->grad_buffer = first;

    auto count = 0;
    for (auto l = all_nodes.crbegin(); l != all_nodes.crend(); ++l) {
        util::timer t;
        t.start();
        (*l)->backward(eng, united_backend);
        t.stop();
        // std::cerr << "back: " << count++ << "  === " << t.elapsed_seconds() << "\n";
    }
}

void sequential::update_weight(optimizer::optimizer* opt)
{
    for (auto& l : all_nodes)
        l->update_weight(opt, eng);
}

} // namespace topo
} // namespace yonn

