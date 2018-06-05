#pragma once
#include "type.hh"
#include "network.hh"
#include "tensor.hh"

#include "util/util.hh"
#include "util/gradient-check.hh"
#include "util/progress-display.hh"
#include "util/timer.hh"

// TODO header file include all layers
#include "layer/fully-connected-layer.hh"
#include "layer/convolutional-layer.hh"
#include "layer/average-pooling-layer.hh"

#include "loss-function/loss-function.hh"

#include "optimizer/optimizer.hh"

#include "io/mnist-parser.hh"

#include "activation/tanh.hh"
#include "activation/leaky_relu.hh"

