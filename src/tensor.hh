#pragma once
#include <vector>
#include "type.hh"

namespace yonn
{

// inner vector for h*w*depth, outer vector for sample (mini batch)
using tensor = std::vector<vec_t>;
// using tensor = std::vector<float>;

} // namespace yonn

