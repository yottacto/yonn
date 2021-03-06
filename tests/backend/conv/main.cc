#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include "yonn.hh"

void random_generate(yonn::tensor& t)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    // std::mt19937 gen(444);
    std::uniform_real_distribution<> dis(-1, 1);
    for (auto& v : t)
        for (auto& i : v)
            i = dis(gen);
}

void print(yonn::tensor const& t)
{
    for (auto v : t) {
        for (auto i : v)
            // std::cout << std::fixed << std::setprecision(4) << i << " ";
            std::cout << i << " ";
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main()
{
    // auto l = conv(4, 4, 2, 1, 1);
    // std::vector<yonn::tensor> in{
    //     {{
    //          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //      }},

    //     {{
    //       0, 0, 0, 0
    //      }},

    //     {{0}},
    // };
    // yonn::tensor dout{
    //     {0, 0, 0, 0, 0, 0, 0, 0, 0},
    // };


    // auto l = conv(4, 4, 2, 2, 1, internal);
    // l.allocate_nsamples(1);
    // std::vector<yonn::tensor> in{
    //     {{
    //          1,  2,  3,  4,
    //          5,  6,  7,  8,
    //          9,  10, 11, 12,
    //          13, 14, 15, 16,

    //          1,  2,  3,  4,
    //          5,  6,  7,  8,
    //          9,  10, 11, 12,
    //          13, 14, 15, 16,
    //      }},

    //     {{1, 2,
    //       3, 4,

    //       1, 2,
    //       3, 4}},

    //     {{0}},
    // };
    // yonn::tensor dout{
    //     {1, 1, 1, 1, 1, 1, 1, 1, 1},
    // };

    std::vector<yonn::tensor> in{
        {
            {
                1,  2,  3,  4,
                5,  6,  7,  8,
                9,  10, 11, 12,
                13, 14, 15, 16,

                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
            },

            {
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
            },
        },

        {{
             1, 2,
             3, 4,

             1, 2, 3, 4,
             1, 2, 3, 4,
             1, 2, 3, 4,
             1, 2, 3, 4,
             1, 2, 3, 4,
         }},

        {{0, 0, 0}}
    };
    yonn::tensor dout{
        {
            1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1,
        },

        {
            1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1,
        },
    };

    for (auto& i : in) random_generate(i);
    random_generate(dout);


    using conv    = yonn::convolutional_layer;
    auto opencl   = yonn::core::backend_type::opencl;
    auto internal = yonn::core::backend_type::internal;

    yonn::network<yonn::topo::sequential> net;
    auto& eng = net.net.eng;
    auto& e   = std::get<yonn::core::engine::opencl>(eng);

    auto back = internal;
    auto l = conv(4, 4, 2, 2, 3, back);
    l.init_engine(back, eng);
    if (back == internal)
        l.allocate_nsamples(2);
    else if (back == opencl)
        l.allocate_nsamples_opencl(2, e);

    // l.set_input_data(in, eng);
    // l.forward(eng, false);
    // print(l.get_output_data());
    // l.set_output_grad(dout, eng);
    // l.backward(eng, false);
    // print(l.get_input_grad()[1]);

    // print(yonn::gradient_check(l, in, dout, eng)[0]);
    std::cout << yonn::gradient_check(l, in, dout, eng) << "\n";

    std::cout << "hello conv\n";
}

