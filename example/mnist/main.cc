#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <string>
#include <cmath>
#include "yonn.hh"

// std::string const COLOR_RST{"\e[0m"};
// std::string const COLOR_ACT{"\e[1;32m"};
// std::string const COLOR_ARG{"\e[1;35m"};

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

// #define INFO(act, arg) \
//     std::cerr << COLOR_ACT << act \
//         << COLOR_ARG << arg << "\n" \
//         << COLOR_RST

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

    // config
    auto back = internal;
    auto mini_batch_size = 32;
    auto alpha = 0.1;
    auto train_cut = 10000;
    auto test_cut = 10000;
    auto epoch_size = 0;

    {
        std::fstream fin{"config"};
        std::string str;
        while (std::getline(fin, str)) {
            std::stringstream buf{str};
            std::string name;
            buf >> name;
            if (name == "backend") {
                int b;
                buf >> b;
                back = static_cast<yonn::core::backend_type>(b);
            } else if (name == "mini_batch_size") {
                buf >> mini_batch_size;
            } else if (name == "alpha") {
                buf >> alpha;
            } else if (name == "train_cut") {
                buf >> train_cut;
                if (!train_cut)
                    train_cut = train_images.size();
            } else if (name == "test_cut") {
                buf >> test_cut;
                if (!test_cut)
                    test_cut = test_images.size();
            } else if (name == "epoch") {
                buf >> epoch_size;
                epoch_size--;
            }
        }

        std::cerr << "config:\n";
        auto const width = 24;
        INFO(
            std::setw(width) << "backend: ",
            (back == yonn::core::backend_type::internal
                ? "internal"
                : "opencl")
        );
        INFO(std::setw(width) << "mini_batch_size: ", mini_batch_size);
        INFO(std::setw(width) << "alpha: ", alpha);
        INFO(std::setw(width) << "epoch: ", epoch_size+1);

        std::cerr << "\n";
    }

    yonn::network<yonn::topo::sequential> net{back};

    net << conv(32, 32, 5, 1, 6)
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

    // yonn::optimizer::adam optimizer;
    // yonn::optimizer::adagrad optimizer;
    // yonn::optimizer::nesterov_momentum optimizer;
    yonn::optimizer::naive optimizer;
    optimizer.alpha = alpha;

    // optimizer.alpha *= std::min(
    //     yonn::value_type(4),
    //     static_cast<yonn::value_type>(std::sqrt(mini_batch_size))
    // );


    train_images.resize(train_cut);
    train_labels.resize(train_cut);

    test_images.resize(test_cut);
    test_labels.resize(test_cut);


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
            INFO("time elapsed: ", t.elapsed_seconds() << "s.\n");
        }
    };

    auto epoch = 0;
    auto first_epoch = true;
    auto each_epoch = [&](auto last = false) {
        // result for test images
        if (!first_epoch) {
            yonn::util::progress_display test_pd(test_images.size());
            INFO("testing", " (" << test_images.size() << ") images:\n");

            auto each_test = [&](auto last = false) {
                test_pd.tick();
                test_pd.display(std::cerr);
                if (last) {
                    std::cerr << "\ntest completed.\n";
                }
            };

            yonn::util::timer tt;
            tt.start();
            net.test(test_images, test_labels, each_test).print_detail(std::cerr);
            tt.stop();

            INFO("time elapsed: ", tt.elapsed_seconds() << "s.\n");

            pd.reset();
            first_batch = true;
        }
        first_epoch = false;
        if (last)
            std::cerr << "\nall epoches completed.\n";
        else {
            INFO("epoch: ", epoch++);
            INFO("training", " (" << train_images.size() << ") images:\n");
        }
    };

    // debug info
    // auto print = [](auto const& v) {
    //     for (auto i : v)
    //         std::cerr << std::fixed << std::setprecision(10) << i << " ";
    //     std::cerr << "\n";
    // };

    net.train<yonn::loss_function::mse>(
        optimizer,
        train_images,
        train_labels,
        mini_batch_size,
        epoch_size,
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

