#include <iostream>
#include <iomanip>
#include <algorithm>
#include <string>
#include <cmath>
#include "yonn.hh"

std::string const COLOR_RST{"\e[0m"};
std::string const COLOR_ACT{"\e[1;32m"};
std::string const COLOR_ARG{"\e[1;35m"};

#define O true
#define X false
static const bool tb[] = {
    O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
    O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
    O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
    X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
    X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
    X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
};
#undef O
#undef X

int main()
{
    using fc               = yonn::fully_connected_layer;
    using conv             = yonn::convolutional_layer;
    using avg_pool         = yonn::average_pooling_layer;
    using tanh             = yonn::activation::tanh;
    using leaky_relu       = yonn::activation::leaky_relu;
    using connection_table = yonn::core::connection_table;

    yonn::ignore(tanh{});

    yonn::tensor train_images, test_images;
    std::vector<yonn::label_t> train_labels, test_labels;

    std::string const data_path{"data"};
    yonn::io::mnist::parse_images(
        data_path + "/train-images-idx3-ubyte",
        train_images,
        -1.0, +1.0, 2, 2
    );
    yonn::io::mnist::parse_labels(
        data_path + "/train-labels-idx1-ubyte",
        train_labels
    );
    yonn::io::mnist::parse_images(
        data_path + "/t10k-images-idx3-ubyte",
        test_images,
        -1.0, +1.0, 2, 2
    );
    yonn::io::mnist::parse_labels(
        data_path + "/t10k-labels-idx1-ubyte",
        test_labels
    );

    auto internal = yonn::core::backend_type::internal;
    auto opencl   = yonn::core::backend_type::opencl;

    yonn::ignore(internal);
    yonn::ignore(opencl);

    auto back = internal;
    yonn::network<yonn::topo::sequential> net{back};

    net << conv(32, 32, 5, 1, 6, opencl)
        << leaky_relu()
        << avg_pool(28, 28, 6, 2)
        << leaky_relu()
        << conv(14, 14, 5, 6, 16, connection_table(tb, 6, 16))
        // << conv(14, 14, 5, 6, 16)
        << leaky_relu()
        << avg_pool(10, 10, 16, 2)
        << leaky_relu()
        << conv(5, 5, 5, 16, 120)
        << leaky_relu()
        << fc(120, 10)
        << leaky_relu();

    std::cerr << "net constructed.\n";

    auto cut = 4000;
    train_images.resize(cut);
    train_labels.resize(cut);

    test_images.resize(100);
    test_labels.resize(100);


    auto mini_batch_size = 1;

    yonn::util::timer t;
    yonn::util::progress_display pd(train_images.size());

    auto first_batch = true;
    auto each_batch = [&](auto last = false) {
        if (first_batch) {
            t.reset();
            t.start();
            first_batch = false;
        }
        pd.tick(mini_batch_size);
        pd.display(std::cerr);
        if (last) {
            t.stop();
            std::cerr << "\ntraining completed.\n";
            std::cerr << COLOR_ACT << "time elapsed: "
                << COLOR_ARG << t.elapsed_seconds() << "s.\n\n"
                << COLOR_RST;
        }
    };

    auto epoch = 0;
    auto first_epoch = true;
    auto each_epoch = [&](auto last = false) {
        // result for test images
        if (!first_epoch) {
            yonn::util::progress_display test_pd(test_images.size());
            std::cerr << COLOR_ACT << "testing"
                << COLOR_ARG << " (" << test_images.size() << ") images:\n"
                << COLOR_RST;

            auto each_test = [&](auto last = false) {
                test_pd.tick();
                test_pd.display(std::cerr);
                if (last) {
                    std::cerr << "\ntest completed.\n";
                }
            };

            net.test(test_images, test_labels, each_test).print_detail(std::cerr);
            pd.reset();
            first_batch = true;
        }
        first_epoch = false;
        if (last)
            std::cerr << "\nall epoches completed.\n";
        else {
            std::cerr << COLOR_ACT << "epoch: "
                << COLOR_ARG << epoch++ << "\n"
                << COLOR_RST;
            std::cerr << COLOR_ACT << "training"
                << COLOR_ARG << " (" << train_images.size() << ") images:\n"
                << COLOR_RST;
        }
    };

    // yonn::optimizer::adam optimizer;
    // yonn::optimizer::adagrad optimizer;
    // yonn::optimizer::nesterov_momentum optimizer;
    yonn::optimizer::naive optimizer;
    optimizer.alpha = 0.1;

    optimizer.alpha *= std::min(
        yonn::value_type(4),
        static_cast<yonn::value_type>(std::sqrt(mini_batch_size))
    );

    // debug info
    auto print = [](auto const& v) {
        for (auto i : v)
            std::cerr << std::fixed << std::setprecision(10) << i << " ";
        std::cerr << "\n";
    };

    net.train<yonn::loss_function::mse>(
        optimizer,
        train_images,
        train_labels,
        mini_batch_size,
        1,
        each_batch,
        each_epoch
    );

    // print(net.net.all_nodes[10]->output[0]->data[0]);


    // auto id = 10;
    // std::cout << "-> " << test_labels[0] << " "
    //     << net.forward_prop_max_index(test_images[0]) << "\n";
    // print(net.forward_propagation(test_images[0])[0]);

    // result for train images
    // {
    //     yonn::util::progress_display test_pd(train_images.size());
    //     std::cerr << "testing (" << train_images.size() << ") images:\n";
    //     auto each_test = [&](auto last = false) {
    //         test_pd.tick();
    //         test_pd.display(std::cerr);
    //         if (last) {
    //             std::cerr << "\ntest completed.\n";
    //         }
    //     };

    //     auto r = net.test(train_images, train_labels, each_test);
    //     r.print_detail(std::cerr);
    // }

    std::cerr << "hello world\n";
}

