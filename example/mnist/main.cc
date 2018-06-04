#include <iostream>
#include <iomanip>
#include "yonn.hh"

void print(yonn::tensor const& t)
{
    for (auto v : t) {
        auto count = 0;
        for (auto i : v) {
            if (i == -1)
                std::cout << ".";
            else
                std::cout << "*";
            // std::cout << std::fixed << std::setprecision(2) << std::setw(4) << i << " ";
            // std::cout << i << " ";
            count++;
            if (count % 32 == 0)
                std::cout << "\n";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

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

    // auto i = 18;
    // print({train_images[i]});
    // std::cout << train_labels[i] << "\n";
    // return 0;

    yonn::network<yonn::topo::sequential> net;

    // net << fc(32 * 32, 10)
    //     << tanh();

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

    // net.allocate_nsamples(1);
    // std::cerr << "n samples allocated.\n";

    // net.print_out_shapes();
    // return 0;

    // std::cerr << net.net.all_nodes.back()->output[0]->data[0].size() << "\n";

    yonn::optimizer::naive optimizer{0.01};

    // auto cut = 10000;
    // train_images.resize(cut);
    // train_labels.resize(cut);

    auto mini_batch_size = 32;

    auto iter = 0;
    auto each_batch = [&]() {
        if (iter % 1000 < mini_batch_size)
            std::cerr << "iter: " << iter << "\n";
        iter += mini_batch_size;
    };

    auto epoch = 0;
    auto each_epoch = [&]() {
        std::cerr << "epoch: " << epoch << "\n";
        epoch++;
    };


    net.train<yonn::loss_function::softmax>(
        optimizer,
        train_images,
        train_labels,
        mini_batch_size,
        1,
        each_batch,
        each_epoch
    );

    auto vec = net.forward_propagation(test_images[0]);
    for (auto i : vec[0])
        std::cout << i << " ";
    std::cout << "\n";

    // result for train images
    {
        auto r = net.test(train_images, train_labels);
        r.print_detail(std::cout);
        std::cerr << "accuracy: " << r.accuracy() << "\n";
    }

    // result for test images
    {
        auto r = net.test(test_images, test_labels);
        // auto r = net.test(train_images, train_labels);
        r.print_detail(std::cout);
        std::cerr << "accuracy: " << r.accuracy() << "\n";
    }

    std::cout << "hello world\n";
}

