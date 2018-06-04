#include <iostream>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include "yonn.hh"

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
    using fc = yonn::fully_connected_layer;
    using conv = yonn::convolutional_layer;
    using avg_pool = yonn::average_pooling_layer;
    using tanh = yonn::activation::tanh;
    using connection_table = yonn::core::connection_table;

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

    yonn::network<yonn::topo::sequential> net;

    net << conv(32, 32, 5, 1, 6)
        << tanh()
        << avg_pool(28, 28, 6, 2)
        << tanh()
        << conv(14, 14, 5, 6, 16, connection_table(tb, 6, 16))
        << tanh()
        << avg_pool(10, 10, 16, 2)
        << tanh()
        << conv(5, 5, 5, 16, 120)
        << tanh()
        << fc(120, 10)
        << tanh();

    std::cerr << "net constructed.\n";

    // auto cut = 10000;
    // train_images.resize(cut);
    // train_labels.resize(cut);

    auto mini_batch_size = 16;

    yonn::util::timer t;
    yonn::util::progress_display pd(train_images.size());

    auto first = true;
    auto each_batch = [&](auto last = false) {
        if (first) {
            t.start();
            first = false;
        }
        pd.tick(mini_batch_size);
        pd.display(std::cerr);
        if (last) {
            t.stop();
            std::cerr << "\ntraining completed.\n";
            std::cerr << "time elapsed: " << t.elapsed_seconds() << "s.\n";
        }
    };

    auto epoch = 0;
    auto each_epoch = [&](auto last = false) {
        if (last)
            std::cerr << "\nall epoches completed.\n";
        else {
            std::cerr << "epoch: " << epoch++ << "\n";
            std::cerr << "training (" << train_images.size() << ") images:\n";
        }
    };

    // yonn::optimizer::adamax optimizer;
    yonn::optimizer::adagrad optimizer;

    optimizer.alpha *= std::min(
        yonn::value_type(4),
        static_cast<yonn::value_type>(std::sqrt(mini_batch_size))
    );

    net.train<yonn::loss_function::mse>(
        optimizer,
        train_images,
        train_labels,
        mini_batch_size,
        1,
        each_batch,
        each_epoch
    );


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

    // result for test images
    {
        yonn::util::progress_display test_pd(test_images.size());
        std::cerr << "testing (" << test_images.size() << ") images:\n";
        auto each_test = [&](auto last = false) {
            test_pd.tick();
            test_pd.display(std::cerr);
            if (last) {
                std::cerr << "\ntest completed.\n";
            }
        };

        auto r = net.test(test_images, test_labels, each_test);
        r.print_detail(std::cerr);
    }

    std::cerr << "hello world\n";
}

