#pragma once
#include <variant>
#include "core/engine/opencl.hh"

namespace yonn
{
namespace core
{

enum class backend_type
{
    internal,
    opencl,
    network_default,
};

inline auto default_engine()
{
    return backend_type::internal;
}

inline auto layer_default_engine()
{
    return backend_type::network_default;
}

namespace engine
{

using engine_type = std::variant<opencl>;

} // namespace engine
} // namespace core
} // namespace yonn

