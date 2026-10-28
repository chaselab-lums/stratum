# Resolve paths from this Makefile's location so `make -f ...` works from any cwd.
THIS_MAKEFILE := $(abspath $(lastword $(MAKEFILE_LIST)))
PROJECT_ROOT := $(patsubst %/,%,$(dir $(THIS_MAKEFILE)))
EXAMPLE_DIR := $(PROJECT_ROOT)
WORKSPACE_ROOT := $(abspath $(PROJECT_ROOT)/..)
SNN_MLIR_ROOT ?=
PYTHON ?= $(SNN_MLIR_ROOT)/.venv/bin/python
MLIR_DIR ?=
LLVM_BIN ?=
SNN_OPT ?= $(SNN_MLIR_ROOT)/build/bin/snn-opt
MLIR_OPT_BIN ?=
MLIR_TRANSLATE_BIN ?=
LLC     ?= $(LLVM_BIN)/llc
LLD     ?= $(LLVM_BIN)/ld.lld
OBJDUMP ?= riscv64-unknown-elf-objdump
OBJCOPY ?= riscv64-unknown-elf-objcopy
GEM5_INCLUDE_DIR ?= $(WORKSPACE_ROOT)/gem5/include
GEM5_M5OPS_SRC ?= $(WORKSPACE_ROOT)/gem5/util/m5/src/abi/riscv/m5op.S
ENABLE_GEM5_M5OPS ?= 1

LINKER_SCRIPT ?= $(EXAMPLE_DIR)/include/link.ld

# --- Target arch/ABI: RV32 with hardware single-precision float ---
# Installed multilib is rv32imafc/ilp32f. rv32imfc has no matching libgcc.
RISCV_TARGET ?= riscv32-unknown-elf
RISCV_ARCH   ?= rv32imafc_zicsr
RISCV_ABI    ?= ilp32f
RISCV_MATTR  ?= +m,+a,+f,+c,+zicsr
RISCV_GCC  ?= riscv64-unknown-elf-gcc
# common flags for all C/S source files
HTIF_DIR     := $(EXAMPLE_DIR)/include/htif
RISCV_CFLAGS := -march=$(RISCV_ARCH) -mabi=$(RISCV_ABI) -O3 \
                -ffunction-sections -fdata-sections \
				-I$(EXAMPLE_DIR)/include -I$(HTIF_DIR) -I$(GEM5_INCLUDE_DIR) \
				-DGEM5_SEMIHOSTING \
                -nostdlib -nostartfiles
LIBGCC       ?= $(shell $(RISCV_GCC) -march=$(RISCV_ARCH) -mabi=$(RISCV_ABI) -print-file-name=libgcc.a)

# run.py argument mapping
N_STEPS ?=
# gem5 runs: one drawing (input_0.h) unless TEST_VECTORS is raised.
TEST_VECTORS ?= 1
WEIGHTS ?=
NO_TRAIN ?=0
BUILD_NO_TRAIN ?=1
TRAIN_EPOCHS ?=
TRAIN_BATCH_SIZE ?=
TRAIN_NUM_SAMPLES ?=
RUN_PY_FLAGS ?=
PY_TEST_FLAGS ?=

RUN_COMMON_FLAGS :=
ifneq ($(strip $(N_STEPS)),)
RUN_COMMON_FLAGS += --n-steps $(N_STEPS)
endif
ifneq ($(strip $(TEST_VECTORS)),)
RUN_COMMON_FLAGS += --test-vectors $(TEST_VECTORS)
endif
ifneq ($(strip $(WEIGHTS)),)
RUN_COMMON_FLAGS += --weights $(WEIGHTS)
endif
ifeq ($(NO_TRAIN),1)
RUN_COMMON_FLAGS += --no-train
endif

# make build skips training. An explicit NO_TRAIN=0, or any TRAIN_EPOCHS, trains.
ifeq ($(origin NO_TRAIN),command line)
BUILD_EFFECTIVE_NO_TRAIN := $(NO_TRAIN)
else ifneq ($(strip $(TRAIN_EPOCHS)),)
BUILD_EFFECTIVE_NO_TRAIN := 0
else
BUILD_EFFECTIVE_NO_TRAIN := $(BUILD_NO_TRAIN)
endif
ifneq ($(strip $(TRAIN_EPOCHS)),)
RUN_COMMON_FLAGS += --train-epochs $(TRAIN_EPOCHS)
endif
ifneq ($(strip $(TRAIN_BATCH_SIZE)),)
RUN_COMMON_FLAGS += --train-batch-size $(TRAIN_BATCH_SIZE)
endif
ifneq ($(strip $(TRAIN_NUM_SAMPLES)),)
RUN_COMMON_FLAGS += --train-num-samples $(TRAIN_NUM_SAMPLES)
endif

EXAMPLE_FLAGS :=
ifeq ($(QUANTIZE),1)
EXAMPLE_FLAGS += --quantize
endif

BUILD_DIR := $(EXAMPLE_DIR)/build
NETWORK_MLIR := $(BUILD_DIR)/network.mlir
NETWORK_LL ?= $(BUILD_DIR)/network.ll
SIM := $(BUILD_DIR)/sim.elf
PHASE_PROFILE ?= 0
PHASE_MARKER_SRC := $(EXAMPLE_DIR)/phase_marker.c
PHASE_MARKER_OBJ := $(BUILD_DIR)/phase_marker.o

ifeq ($(ENABLE_GEM5_M5OPS),1)
RISCV_CFLAGS += -DENABLE_GEM5_M5OPS
M5OPS_OBJ := $(BUILD_DIR)/m5op.o
endif

.DELETE_ON_ERROR:

.PHONY: help print-paths generate example python-test draw-input build size clean

help:
	@echo "Targets:"
	@echo "  generate        - run run.py export/codegen and lower to RV32 LLVM IR"
	@echo "  python-test     - run run.py --python-test flow"
	@echo "  draw-input      - launch drawing app to create build/input.h"
	@echo "  build           - build sim.elf + sim.elf.hex for RV32 bare-metal runs"
	@echo "  profile         - static disassembly profile for snn_forward_step"
	@echo "  size            - print section sizes"
	@echo "  clean           - remove generated files"
	@echo ""
	@echo "run.py args via make vars:"
	@echo "  QUANTIZE=0|1 NO_TRAIN=0|1 N_STEPS=<int> TEST_VECTORS=<int> WEIGHTS=<path>"
	@echo "  BUILD_NO_TRAIN=0|1 (used by build target when invoking generate; default 1)"
	@echo "  TRAIN_EPOCHS=<int> TRAIN_BATCH_SIZE=<int> TRAIN_NUM_SAMPLES=<int>"
	@echo "  RUN_PY_FLAGS='<extra run.py args>' PY_TEST_FLAGS='<extra python-test args>'"
	@echo ""
	@echo "Path vars (override on command line if needed):"
	@echo "PROJECT_ROOT=$(PROJECT_ROOT)"
	@echo "WORKSPACE_ROOT=$(WORKSPACE_ROOT)"
	@echo "SNN_MLIR_ROOT=$(SNN_MLIR_ROOT)"
	@echo "SNN_OPT=$(SNN_OPT)"
	@echo "MLIR_DIR=$(MLIR_DIR)"
	@echo "LLVM_BIN=$(LLVM_BIN)"
	@echo "MLIR_OPT_BIN=$(MLIR_OPT_BIN)"
	@echo "MLIR_TRANSLATE_BIN=$(MLIR_TRANSLATE_BIN)"
	@echo "GEM5_INCLUDE_DIR=$(GEM5_INCLUDE_DIR)"
	@echo "ENABLE_GEM5_M5OPS=$(ENABLE_GEM5_M5OPS)"

$(SNN_OPT):
	MLIR_DIR=$(MLIR_DIR) bash $(SNN_MLIR_ROOT)/scripts/build_snn_dialect.sh

generate: $(SNN_OPT)
	$(PYTHON) $(EXAMPLE_DIR)/run.py $(EXAMPLE_FLAGS) $(RUN_COMMON_FLAGS) $(RUN_PY_FLAGS)
	SNN_MLIR_ROOT=$(SNN_MLIR_ROOT) \
	SNN_OPT=$(SNN_OPT) \
	MLIR_DIR=$(MLIR_DIR) \
	LLVM_BIN=$(LLVM_BIN) \
	MLIR_OPT_BIN=$(MLIR_OPT_BIN) \
	MLIR_TRANSLATE_BIN=$(MLIR_TRANSLATE_BIN) \
	bash $(EXAMPLE_DIR)/lower_cpu_linux.sh $(NETWORK_MLIR)

python-test:
	$(PYTHON) $(EXAMPLE_DIR)/run.py --python-test $(RUN_COMMON_FLAGS) $(PY_TEST_FLAGS)

draw-input:
	$(PYTHON) $(EXAMPLE_DIR)/draw_input_app.py

build:
	$(MAKE) -f $(THIS_MAKEFILE) generate NO_TRAIN=$(BUILD_NO_TRAIN)
	$(MAKE) -f $(THIS_MAKEFILE) link-elf

.PHONY: build-nogen link-elf
build-nogen: link-elf

link-elf:
	$(LLC) -mtriple=$(RISCV_TARGET) -mattr=$(RISCV_MATTR) -target-abi=$(RISCV_ABI) -filetype=obj \
	  $(NETWORK_LL) -o $(BUILD_DIR)/network.o
	$(if $(M5OPS_OBJ),$(RISCV_GCC) $(RISCV_CFLAGS) -c \
	  $(GEM5_M5OPS_SRC)               -o $(M5OPS_OBJ))
	$(RISCV_GCC) $(RISCV_CFLAGS) -c \
	  $(HTIF_DIR)/crt.S                 -o $(BUILD_DIR)/crt.o
	$(RISCV_GCC) $(RISCV_CFLAGS) -c \
	  $(HTIF_DIR)/syscalls.c            -o $(BUILD_DIR)/syscalls.o
	$(RISCV_GCC) $(RISCV_CFLAGS) $(if $(filter 1,$(PHASE_PROFILE)),-DSNN_PHASE_PROFILE,) -c \
	  $(BUILD_DIR)/main.c               -o $(BUILD_DIR)/main.o
ifeq ($(PHASE_PROFILE),1)
	$(RISCV_GCC) $(RISCV_CFLAGS) -c $(PHASE_MARKER_SRC) -o $(PHASE_MARKER_OBJ)
endif
	$(RISCV_GCC) $(RISCV_CFLAGS) -T$(LINKER_SCRIPT) -Wl,--gc-sections \
	  $(BUILD_DIR)/crt.o \
	  $(BUILD_DIR)/syscalls.o \
	  $(M5OPS_OBJ) \
	  $(BUILD_DIR)/network.o \
	  $(BUILD_DIR)/main.o \
	  $(if $(filter 1,$(PHASE_PROFILE)),$(PHASE_MARKER_OBJ),) \
	  $(LIBGCC) -o $(SIM)
	$(OBJCOPY) -O verilog $(SIM) $(SIM).hex
	@echo "Built: $(SIM) $(SIM).hex"
	$(OBJDUMP) -d $(SIM) > $(SIM).dis
	@echo "Disassembly saved to $(SIM).dis"

PROFILE_SCRIPT ?= $(WORKSPACE_ROOT)/scripts/profile_dis.py
PROFILE_ARGS   ?=
PROFILE_PLOTS  ?= $(BUILD_DIR)/profile_plots
PROFILE_STATS  ?= $(BUILD_DIR)/profile_stats.json

.PHONY: profile
profile: build
	python3 $(PROFILE_SCRIPT) $(SIM).dis \
	  --plots-dir $(PROFILE_PLOTS) --save-stats $(PROFILE_STATS) \
	  --func snn_forward_step $(PROFILE_ARGS)

clean:
	rm -rf $(BUILD_DIR)