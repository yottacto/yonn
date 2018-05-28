#pragma once
#include "layer/layer.hh"
#include "nodes.hh"

namespace yonn
{

template <class Net>
struct network
{
    using network_type = Net;

    void add(layer& l);

private:
    network_type net;
};

template <class Layer>
auto& operator<<(network<sequential>& net, Layer&& l)
{
    net.add(std::forward<Layer>(l));
    return net;
}


// implementation of network<Net>
template <class Net>
void network<Net>::add(layer& l)
{
    net.add(l);
}

} // namespace yonn

