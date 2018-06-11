#pragma once
#include <iostream>

#include <cmath>
#include <memory>
#include <CL/cl.hpp>
#include "core/backend.hh"
#include "util/util.hh"
#include "type.hh"

#include "core/kernel/opencl/optimizer.hh"

namespace yonn
{
namespace optimizer
{

struct optimizer
{
    optimizer()
    {
    }

    void init_opencl(core::engine::opencl& eng)
    {
        if (!opencl_initialized) {
            sources.emplace_back(
                opencl_kernel::optimizer_kernel_code.c_str(),
                opencl_kernel::optimizer_kernel_code.size()
                );
            program = cl::Program{eng.context, sources};
            if (program.build({eng.default_device}) != CL_SUCCESS) {
                // FIXME
                std::cerr << "Error building: "
                    << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(eng.default_device) << "\n";
                throw;
            }
            opencl_initialized = true;
        }
    }

    // TODO unique_ptr, non-copyable
    // optimizer(optimizer const&) = default;
    // optimizer(optimizer&&)      = default;
    // optimizer& operator=(optimizer const&) = default;
    // optimizer& operator=(optimizer&&)      = default;

    virtual ~optimizer() = default;
    virtual void update(vec_t const& dw, vec_t& w) = 0;

    virtual void update(
        cl::Buffer& dw,
        cl::Buffer& w,
        core::engine::opencl& e,
        size_t size
    )
    {
        ignore(dw);
        ignore(w);
        ignore(e);
        ignore(size);
        // TODO if not implemented, can't call this
        throw;
    }

    virtual void reset() {}

    bool opencl_initialized{false};
    cl::Program::Sources sources;
    cl::Program program;
};

struct naive : optimizer
{
    using kernel_type = cl::make_kernel<
        value_type,
        cl::Buffer&, cl::Buffer&
    >;

    naive(value_type alpha = 0.01) : alpha{alpha}
    {
    }

    void update(vec_t const& dw, vec_t& w) override
    {
        for (size_t i{0}; i < w.size(); i++)
            w[i] -= alpha * dw[i];
    }

    void update(cl::Buffer& dw, cl::Buffer& w, core::engine::opencl& e, size_t size) override
    {
        if (!opencl_kernel_initialized) {
            optimizer::init_opencl(e);
            kernel = std::make_unique<kernel_type>(program, "naive");
            opencl_kernel_initialized = true;
        }
        (*kernel)(
            cl::EnqueueArgs{e.queue, cl::NDRange(size)},
            alpha,
            dw,
            w
        ).wait();
    }

    value_type alpha;

    bool opencl_kernel_initialized{false};
    std::unique_ptr<kernel_type> kernel;
};

template <int N>
struct stateful_optimizer : optimizer
{
    void reset() override
    {
        for (auto& e : extra)
            e.clear();
    }

protected:
    template <int Index>
    vec_t& get(vec_t const& key)
    {
        static_assert(Index < N, "index out of range");
        if (extra[Index][&key].empty())
            extra[Index][&key].resize(key.size(), value_type{});
        return extra[Index][&key];
    }
    std::unordered_map<vec_t const*, vec_t> extra[N];
};

struct adagrad : stateful_optimizer<1>
{
    adagrad() : alpha{value_type{0.01}}, eps{value_type{1e-8}} {}

    void update(vec_t const& dw, vec_t& w)
    {
        vec_t& g = get<0>(w);
        for (size_t i{0}; i < w.size(); i++) {
            g[i] += dw[i] * dw[i];
            w[i] -= alpha * dw[i] / (std::sqrt(g[i]) + eps);
        }
    }

    value_type alpha;  // learning rate
private:
    value_type eps;
};

struct RMSprop : stateful_optimizer<1>
{
    RMSprop() :
        alpha{value_type{0.0001}},
        mu{value_type{0.99}},
        eps{value_type{1e-8}}
    {}

    void update(vec_t const& dw, vec_t &w)
    {
        vec_t& g = get<0>(w);

        for (size_t i{0}; i < w.size(); i++) {
            g[i] = mu * g[i] + (1 - mu) * dw[i] * dw[i];
            w[i] -= alpha * dw[i] / std::sqrt(g[i] + eps);
        }
    }

    value_type alpha;  // learning rate
    value_type mu;     // decay term
private:
    value_type eps;  // constant value to avoid zero-division
};

struct adam : stateful_optimizer<2>
{
    adam() :
        alpha{value_type{0.001}},
        b1{value_type{0.9}},
        b2{value_type{0.999}},
        b1_t{value_type{0.9}},
        b2_t{value_type{0.999}},
        eps{value_type{1e-8}}
    {}

    void update(vec_t const&dw, vec_t &w)
    {
        vec_t &mt = get<0>(w);
        vec_t &vt = get<1>(w);

        for (size_t i{0}; i < w.size(); i++) {
            mt[i] = b1 * mt[i] + (value_type(1) - b1) * dw[i];
            vt[i] = b2 * vt[i] + (value_type(1) - b2) * dw[i] * dw[i];

            // L2 norm based update rule
            w[i] -= alpha * (mt[i] / (value_type(1) - b1_t)) /
                std::sqrt((vt[i] / (value_type(1) - b2_t)) + eps);
        }

        b1_t *= b1;
        b2_t *= b2;
    }

    value_type alpha;  // learning rate
    value_type b1;     // decay term
    value_type b2;     // decay term
    value_type b1_t;   // decay term power t
    value_type b2_t;   // decay term power t

private:
    value_type eps;  // constant value to avoid zero-division
};

struct adamax : stateful_optimizer<2>
{
    adamax() :
        alpha{value_type{0.002}},
        b1{value_type{0.9}},
        b2{value_type{0.999}},
        b1_t{b1},
        eps{value_type{1e-8}}
    {}

    void update(vec_t const&dw, vec_t &w)
    {
        vec_t &mt = get<0>(w);
        vec_t &ut = get<1>(w);

        for (size_t i{0}; i < w.size(); i++) {
            mt[i] = b1 * mt[i] + (value_type(1) - b1) * dw[i];
            ut[i] = std::max(b2 * ut[i], std::abs(dw[i]));

            // Lp norm based update rule
            w[i] -= (alpha / (1.0 - b1_t)) * (mt[i] / (ut[i] + eps));
        }

        b1_t *= b1;
    }

    value_type alpha;  // learning rate
    value_type b1;     // decay term
    value_type b2;     // decay term
    value_type b1_t;   // decay term power t

private:
    value_type eps;  // constant value to avoid zero-division
};

// SGD without momentum
struct gradient_descent : optimizer
{
    gradient_descent() : alpha{value_type(0.01)}, lambda{value_type(0)} {}

    void update(vec_t const&dw, vec_t &w)
    {
        for (size_t i{0}; i < w.size(); i++)
            w[i] = w[i] - alpha * (dw[i] + lambda * w[i]);
    }

    value_type alpha;   // learning rate
    value_type lambda;  // weight decay
};

// SGD with momentum
struct momentum : stateful_optimizer<1>
{
    momentum() :
        alpha{value_type{0.01}},
        lambda{value_type{0}},
        mu{value_type{0.9}}
    {}

    void update(vec_t const&dw, vec_t &w)
    {
        vec_t &dwprev = get<0>(w);

        for (size_t i{0}; i < w.size(); i++) {
            value_type v = mu * dwprev[i] - alpha * (dw[i] + w[i] * lambda);
            w[i] += v;
            dwprev[i] = v;
        }
    }

    value_type alpha;   // learning rate
    value_type lambda;  // weight decay
    value_type mu;      // momentum
};

struct nesterov_momentum : stateful_optimizer<1>
{
    nesterov_momentum() :
        alpha{value_type{0.01}}, lambda{value_type{0}}, mu{value_type{0.9}}
    {}

    void update(vec_t const& dw, vec_t& w)
    {
        vec_t &dwprev = get<0>(w);
        for (size_t i{0}; i < w.size(); i++) {
            value_type v = mu * dwprev[i] - alpha * (dw[i] + w[i] * lambda);
            w[i] += (-mu) * dwprev[i] + (1 + mu) * v;
            dwprev[i] = v;
        }
    }

    value_type alpha;   // learning rate
    value_type lambda;  // weight decay
    value_type mu;      // momentum
};

} // namespace optimizer
} // namespace yonn

