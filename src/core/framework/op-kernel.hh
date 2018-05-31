#pragma once
#include <vector>
#include <memory>
#include "tensor.hh"
#include "core/backend.hh"
#include "core/parameter/parameter.hh"

namespace yonn
{
namespace core
{
namespace framework
{

struct op_kernel_context
{

    // TODO in const?
    void set_in_out(
        std::vector<tensor*>& _in_data,
        std::vector<tensor*>& _out_data
    )
    {
        in_data = &_in_data;
        out_data = &_out_data;
    }

    void set_in_out(
        std::vector<tensor*>& _in_data,
        std::vector<tensor*>& _in_grad,
        std::vector<tensor*>& _out_data,
        std::vector<tensor*>& _out_grad
    )
    {
        in_data  = &_in_data;
        in_grad  = &_in_grad;
        out_data = &_out_data;
        out_grad = &_out_grad;
    }

    auto engine() const -> core::backend { return backend; }
    auto set_engine(core::backend engine) { backend = engine; }
    auto input(size_t index)       -> tensor&       { return *(*in_data)[index]; }
    auto input(size_t index) const -> tensor const& { return *(*in_data)[index]; }
    auto output(size_t index)       -> tensor&       { return *(*out_data)[index]; }
    auto output(size_t index) const -> tensor const& { return *(*out_data)[index]; }

    auto input_grad(size_t index)       -> tensor&       { return *(*in_grad)[index]; }
    auto input_grad(size_t index) const -> tensor const& { return *(*in_grad)[index]; }
    auto output_grad(size_t index)       -> tensor&       { return *(*out_grad)[index]; }
    auto output_grad(size_t index) const -> tensor const& { return *(*out_grad)[index]; }

// private:
    core::backend backend;
    std::vector<tensor*>* in_data;
    std::vector<tensor*>* out_data;
    std::vector<tensor*>* in_grad;
    std::vector<tensor*>* out_grad;
};

struct op_kernel
{
    op_kernel() = default;

    virtual ~op_kernel() = default;

    virtual void compute(op_kernel_context& context) = 0;

protected:
};

} // namespace framework
} // namespace core
} // namespace yonn

