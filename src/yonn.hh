#pragma once
#include "network.hh"
#include "tensor.hh"
#include "util/util.hh"
#include "util/gradient_check.hh"
#include "type.hh"

// TODO header file include all layers
#include "layer/fully-connected-layer.hh"
#include "layer/convolutional-layer.hh"
#include "layer/average-pooling-layer.hh"

#include "loss-function/loss-function.hh"

#include "optimizer/optimizer.hh"

#include "io/mnist-parser.hh"

#include "activation/tanh.hh"

