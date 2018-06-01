#include <iostream>
#include "yonn.hh"

int main()
{
    using fc = yonn::fully_connected_layer;

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
    net << fc(100, 10, true);

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

