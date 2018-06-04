#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include "yonn.hh"

void random_generate(yonn::tensor& t)
{
    std::random_device rd;
    std::mt19937 gen(rd());
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
    using conv = yonn::convolutional_layer;

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


    // auto l = conv(4, 4, 2, 2, 1);
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
    //     {0, 0, 0, 0, 0, 0, 0, 0, 0},
    // };


    auto l = conv(4, 4, 2, 2, 3);
    l.allocate_nsamples(2);

    std::vector<yonn::tensor> in{
        {
            {
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
            },

            {
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
            },
        },

        {{
             1, 2, 3, 4,
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

    // for (auto& i : in) random_generate(i);
    random_generate(dout);

    // l.set_input_data(in);
    // l.forward();
    // print(l.get_output_data());
    // l.set_output_grad(dout);
    // l.backward();
    // print(l.get_input_grad()[0]);

    // print(yonn::gradient_check(l, in, dout)[0]);
    std::cout << yonn::gradient_check(l, in, dout) << "\n";

    std::cout << "hello world\n";
}

