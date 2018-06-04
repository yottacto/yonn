#pragma once
#include <cmath>
#include "type.hh"

namespace yonn
{
namespace optimizer
{

struct optimizer
{
    optimizer()                 = default;
    optimizer(optimizer const&) = default;
    optimizer(optimizer&&)      = default;
    optimizer& operator=(optimizer const&) = default;
    optimizer& operator=(optimizer&&)      = default;

    virtual ~optimizer() = default;
    virtual void update(vec_t const& dw, vec_t& w) = 0;
    virtual void reset() {}
};

struct naive : optimizer
{
    naive(value_type alpha = 0.01) : alpha{alpha} {}

    void update(vec_t const& dw, vec_t& w) override
    {
        for (size_t i{0}; i < w.size(); i++)
            w[i] -= alpha * dw[i];
    }

    value_type alpha;
};

template <int N>
struct stateful_optimizer : optimizer
{
    void reset() override
    {
        for (auto &e : extra) e.clear();
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

    void update(const vec_t &dw, vec_t &w)
    {
        vec_t &g = get<0>(w);
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

    void update(const vec_t &dw, vec_t &w)
    {
        vec_t &g = get<0>(w);

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

    void update(const vec_t &dw, vec_t &w)
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

    void update(const vec_t &dw, vec_t &w)
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

    void update(const vec_t &dw, vec_t &w)
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

    void update(const vec_t &dw, vec_t &w)
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

} // namespace optimizer
} // namespace yonn

