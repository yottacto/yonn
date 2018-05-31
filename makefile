# named colors
COLOR_RST = \e[0m
COLOR_ACT = \e[1;32m
COLOR_ARG = \e[1;35m

# build tools and flags
CC = clang++
LD = clang++

# debug flags -D DEBUGGING_ENABLED -g
CCFLAGS = -fno-operator-names -march=native -std=c++14 -Wall -Wextra -Isrc/ -O3
LDFLAGS = -fopenmp -fsanitize=undefined
OBJECTS = $(BUILD)/src/main.o
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
$(BUILD)/src: | $(BUILD)/
	@echo -e "$(COLOR_ACT)making directory $(COLOR_ARG)$(BUILD)/src$(COLOR_RST)..."
	mkdir -p $(BUILD)/src
$(BUILD)/src/main.o: src/main.cc src/core/backend.hh src/layer/layer.hh src/loss-function/absolute.hh src/loss-function/loss-function.hh src/network.hh src/node.hh src/nodes.hh src/tensor.hh src/topo/sequential.hh src/type.hh | $(BUILD)/src
	@echo -e "$(COLOR_ACT)compiling $(COLOR_ARG)src/main.cc$(COLOR_RST)..."
	$(CC) -c -o '$@' '$<' $(CCFLAGS)
$(BIN): $(OBJECTS) | $(BUILD)/
	@echo -e "$(COLOR_ACT)loading $(COLOR_ARG)build$(COLOR_RST)..."
	$(LD) -o '$@' $(OBJECTS) $(LDFLAGS)

