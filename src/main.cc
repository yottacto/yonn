#include <iostream>
#include "network.hh"
#include "loss-function/absolute.hh"
#include "layer/fully-connected-layer.hh"
#include "topo/sequential.hh"

int main()
{
    using fc = yonn::fully_connected_layer;
    yonn::network<yonn::topo::sequential> net;
    net << fc(100, 10, true);
    std::cout << "hello world\n";
}

