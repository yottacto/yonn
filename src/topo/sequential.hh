#pragma once

// TODO remove this
#include <iostream>

#include <vector>
#include "nodes.hh"
#include "tensor.hh"
#include "optimizer/optimizer.hh"
#include "core/backend.hh"

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

        if (l.engine() == core::backend_type::network_default)
            l.init_engine(backend, eng);
        else
            l.init_engine(l.engine(), eng);

        if (l.engine() != backend)
            united_backend = false;

        if (all_nodes.size() != 1) {
            connect(all_nodes[all_nodes.size() - 2], all_nodes.back());
        }
    }

    void allocate_nsamples(size_t batch_size);
    auto forward(tensor const& first) -> tensor;
    void backward(tensor const& first);
    void update_weight(optimizer::optimizer* opt);

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

auto sequential::forward(tensor const& first) -> tensor
{
    all_nodes.front()->set_input_data(first, eng);

    for (auto const& l : all_nodes)
        l->forward(eng, united_backend);

    tensor outt;
    // FIXME
    // TODO output channel index?
    all_nodes.back()->output_data(outt);
    // TODO normalize output
    return outt;
}

void sequential::backward(tensor const& first)
{
    own_nodes.back()->set_output_grad(first, eng);

    for (auto l = all_nodes.crbegin(); l != all_nodes.crend(); ++l)
        (*l)->backward(eng);
}

void sequential::update_weight(optimizer::optimizer* opt)
{
    for (auto& l : all_nodes)
        l->update_weight(opt);
}

} // namespace topo
} // namespace yonn

