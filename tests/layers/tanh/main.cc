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
    using tanh = yonn::activation::tanh;

    auto l = tanh();
    l.allocate_input({2, 2, 2});
    l.allocate_output();
    l.allocate_nsamples(2);

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

    // print(yonn::gradient_check(l, in)[0]);
    std::cout << yonn::gradient_check(l, in, dout) << "\n";

    std::cout << "hello world\n";
}
