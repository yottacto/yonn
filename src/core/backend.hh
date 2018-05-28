#pragma once

namespace yonn
{
namespace core
{

enum class backend
{
    internal,
    opencl,
};

inline auto default_engine()
{
    return backend::internal;
}

} // namespace core
} // namespace yonn

