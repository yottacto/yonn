#pragma once
#include <vector>
#include <memory>
#include <variant>
#include <CL/cl.hpp>
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
    using data_type = std::variant<tensor*, cl::Buffer*>;

    // TODO in const?
    void set_in_out(
        std::vector<data_type>& _in_data,
        std::vector<data_type>& _out_data
    )
    {
        in_data = &_in_data;
        out_data = &_out_data;
    }

    void set_in_out(
        std::vector<data_type>& _in_data,
        std::vector<data_type>& _in_grad,
        std::vector<data_type>& _out_data,
        std::vector<data_type>& _out_grad
    )
    {
        in_data  = &_in_data;
        in_grad  = &_in_grad;
        out_data = &_out_data;
        out_grad = &_out_grad;
    }

    auto engine() const -> core::backend_type { return backend; }
    auto set_engine(core::backend_type engine) { backend = engine; }

    auto input(size_t index)       -> data_type&
    {
        return (*in_data)[index];
    }

    auto input(size_t index) const -> data_type const&
    {
        return (*in_data)[index];
    }

    auto output(size_t index)       -> data_type&
    {
        return (*out_data)[index];
    }

    auto output(size_t index) const -> data_type const&
    {
        return (*out_data)[index];
    }

    auto input_grad(size_t index)       -> data_type&
    {
        return (*in_grad)[index];
    }

    auto input_grad(size_t index) const -> data_type const&
    {
        return (*in_grad)[index];
    }

    auto output_grad(size_t index)       -> data_type&
    {
        return (*out_grad)[index];
    }

    auto output_grad(size_t index) const -> data_type const&
    {
        return (*out_grad)[index];
    }

// TODO uncomment
// private:
    core::backend_type backend;
    std::vector<data_type>* in_data;
    std::vector<data_type>* out_data;
    std::vector<data_type>* in_grad;
    std::vector<data_type>* out_grad;
    std::unordered_map<std::string, data_type> extra;
};

struct op_kernel
{
    op_kernel() = default;

    virtual ~op_kernel() = default;

    virtual void compute(op_kernel_context& context, core::engine::engine_type& eng) = 0;

// TODO uncomment
// protected:
};

} // namespace framework
} // namespace core
} // namespace yonn

