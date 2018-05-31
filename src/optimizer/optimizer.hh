#pragma once
#include "type.hh"

namespace yonn
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
};

struct naive_optimizer : optimizer
{
    naive_optimizer(value_type alpha) : alpha{alpha} {}

    void update(vec_t const& dw, vec_t& w) override
    {
        for (size_t i{0}; i < w.size(); i++)
            w[i] -= alpha * dw[i];
    }

    value_type alpha;
};

} // namespace yonn

