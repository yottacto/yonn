#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include "yonn.hh"

void random_generate(yonn::tensor& t)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.1, 0.1);
    for (auto& v : t)
        for (auto& i : v)
            i = dis(gen);
}

void print(yonn::tensor const& t)
{
    for (auto v : t) {
        for (auto i : v)
            std::cout << std::fixed << std::setprecision(12) << i << " ";
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main()
{
    std::vector<yonn::tensor> in{
        {
            {0, 0, 0, 0, 0, 0, 0, 0},
            {0, 0, 0, 0, 0, 0, 0, 0},
        },
    };
    yonn::tensor dout{
        {0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0},
    };

    for (auto& i : in)
        random_generate(i);
    random_generate(dout);


    using tanh = yonn::activation::tanh;
    auto opencl    = yonn::core::backend_type::opencl;
    auto internal  = yonn::core::backend_type::internal;

    yonn::network<yonn::topo::sequential> net;
    auto& eng = net.net.eng;
    auto& e   = std::get<yonn::core::engine::opencl>(eng);

    auto back = internal;

    auto l = tanh(back);
    l.init_engine(back, eng);
    l.allocate_input({2, 2, 2});
    l.allocate_output();
    if (back == internal)
        l.allocate_nsamples(2);
    else if (back == opencl)
        l.allocate_nsamples_opencl(2, e);


    // print(yonn::gradient_check(l, in, dout, eng)[0]);
    std::cout << yonn::gradient_check(l, in, dout, eng) << "\n";

    std::cout << "hello tanh\n";
}

