#pragma once
#include <vector>

namespace yonn
{

// inner vector for h*w*depth, outer vector for sample (mini batch)
using tensor = std::vector<std::vector<float>>;
// using tensor = std::vector<float>;

} // namespace yonn

