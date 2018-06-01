#include <iostream>
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
    net << conv(32, 32, 5, 1, 6, yonn::padding::valid, true, 1, 1)
        << fc(4704, 10, true);

    yonn::optimizer::naive optimizer;
    net.train<yonn::loss_function::absolute>(
        optimizer,
        train_images,
        train_labels,
        1,
        1
    );

    auto r = net.test(test_images, test_labels);
    std::cerr << "accuracy: " << r.accuracy() << "\n";

    std::cout << "hello world\n";
}

