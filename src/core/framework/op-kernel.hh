#pragma once

namespace yonn
{
namespace core
{
namespace framework
{

struct op_kernel
{
    virtual ~op_kernel() = default;

    virtual void compute() = 0;

protected:
};

} // namespace framework
} // namespace core
} // namespace yonn

