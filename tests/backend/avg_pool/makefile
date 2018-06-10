# named colors
COLOR_RST = \e[0m
COLOR_ACT = \e[1;32m
COLOR_ARG = \e[1;35m

# build tools and flags
CC = clang++
LD = clang++

# debug flags -D DEBUGGING_ENABLED -g
CCFLAGS = -fno-operator-names -march=native -std=c++17 -Wall -Wextra -I../../../src/ -g -O0
LDFLAGS = -g -lOpenCL
OBJECTS = $(BUILD)/../conv/main.o
BUILD = build
BIN = $(BUILD)/build

# phonies
.PHONY: all clean test reconf rebuild
all: $(BIN)
clean:
	@echo -e "$(COLOR_ACT)removing $(COLOR_ARG)$(BUILD)$(COLOR_RST)..."
	rm -rf $(BUILD)/
test: all
	@echo -e "$(COLOR_ACT)running $(COLOR_ARG)$(BIN)$(COLOR_RST)..."
	$(BIN)
reconf:
	@echo -e "$(COLOR_ACT)reconfiguring$(COLOR_RST)..."
	./configure
rebuild: clean
	@$(MAKE) --no-print-directory all

# build rules
$(BUILD)/:
	@echo -e "$(COLOR_ACT)making directory $(COLOR_ARG)$(BUILD)/$(COLOR_RST)..."
	mkdir -p $(BUILD)/
$(BUILD)/..: | $(BUILD)/
	@echo -e "$(COLOR_ACT)making directory $(COLOR_ARG)$(BUILD)/..$(COLOR_RST)..."
	mkdir -p $(BUILD)/..
$(BUILD)/../conv: | $(BUILD)/..
	@echo -e "$(COLOR_ACT)making directory $(COLOR_ARG)$(BUILD)/../conv$(COLOR_RST)..."
	mkdir -p $(BUILD)/../conv
$(BUILD)/../conv/main.o: ../conv/main.cc ../../../src/activation/leaky_relu.hh ../../../src/activation/tanh.hh ../../../src/core/backend.hh ../../../src/core/engine/opencl.hh ../../../src/core/framework/op-kernel.hh ../../../src/core/kernel/average-pooling-op-internal.hh ../../../src/core/kernel/average-pooling-op.hh ../../../src/core/kernel/convolutional-op-internal.hh ../../../src/core/kernel/convolutional-op.hh ../../../src/core/kernel/fully-connected-op-internal.hh ../../../src/core/kernel/fully-connected-op.hh ../../../src/core/kernel/opencl/average-pooling.hh ../../../src/core/kernel/opencl/convolutional.hh ../../../src/core/kernel/opencl/fully-connected.hh ../../../src/core/kernel/opencl/leaky-relu.hh ../../../src/core/kernel/opencl/tanh.hh ../../../src/core/parameter/avg-pool-parameter.hh ../../../src/core/parameter/conv-parameter.hh ../../../src/core/parameter/fully-parameter.hh ../../../src/core/parameter/parameter.hh ../../../src/io/mnist-parser.hh ../../../src/layer/average-pooling-layer.hh ../../../src/layer/convolutional-layer.hh ../../../src/layer/fully-connected-layer.hh ../../../src/layer/layer.hh ../../../src/loss-function/absolute.hh ../../../src/loss-function/loss-function.hh ../../../src/loss-function/mse.hh ../../../src/loss-function/softmax.hh ../../../src/network.hh ../../../src/node.hh ../../../src/nodes.hh ../../../src/optimizer/optimizer.hh ../../../src/tensor.hh ../../../src/topo/sequential.hh ../../../src/type.hh ../../../src/util/gradient-check.hh ../../../src/util/progress-display.hh ../../../src/util/timer.hh ../../../src/util/util.hh ../../../src/yonn.hh | $(BUILD)/../conv
	@echo -e "$(COLOR_ACT)compiling $(COLOR_ARG)../conv/main.cc$(COLOR_RST)..."
	$(CC) -c -o '$@' '$<' $(CCFLAGS)
$(BIN): $(OBJECTS) | $(BUILD)/
	@echo -e "$(COLOR_ACT)loading $(COLOR_ARG)build$(COLOR_RST)..."
	$(LD) -o '$@' $(OBJECTS) $(LDFLAGS)

