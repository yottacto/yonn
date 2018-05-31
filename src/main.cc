#include <iostream>
#include "network.hh"
#include "loss-function/absolute.hh"

int main()
{
    yonn::network<yonn::loss_function::absolute> net;
    std::cout << "hello world\n";
}

