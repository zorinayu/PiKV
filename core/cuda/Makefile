# PiKV CUDA Kernels Makefile
# Compiles routing, compression, and scheduling kernels

# CUDA Compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -std=c++14 -O3 -use_fast_math -Xcompiler -fPIC
NVCC_FLAGS += -gencode arch=compute_70,code=sm_70  # V100
NVCC_FLAGS += -gencode arch=compute_75,code=sm_75  # T4, RTX 20xx
NVCC_FLAGS += -gencode arch=compute_80,code=sm_80  # A100, RTX 30xx
NVCC_FLAGS += -gencode arch=compute_86,code=sm_86  # RTX 30xx
NVCC_FLAGS += -gencode arch=compute_89,code=sm_89  # RTX 40xx
NVCC_FLAGS += -gencode arch=compute_90,code=sm_90  # H100

# Include directories
INCLUDES = -I/usr/local/cuda/include
INCLUDES += -I$(shell python -c "import torch; print(torch.utils.cpp_extension.include_paths()[0])")

# Library directories and libraries
LIBS = -L/usr/local/cuda/lib64 -lcudart -lcublas -lcusolver -lcurand

# Source files
ROUTING_SRC = pikv_routing_kernel.cu
COMPRESSION_SRC = pikv_compression_kernel.cu
SCHEDULING_SRC = pikv_scheduling_kernel.cu

# Object files
ROUTING_OBJ = pikv_routing_kernel.o
COMPRESSION_OBJ = pikv_compression_kernel.o
SCHEDULING_OBJ = pikv_scheduling_kernel.o

# Shared library
SHARED_LIB = libpikv_kernels.so

# Default target
all: $(SHARED_LIB)

# Compile individual object files
$(ROUTING_OBJ): $(ROUTING_SRC)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

$(COMPRESSION_OBJ): $(COMPRESSION_SRC)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

$(SCHEDULING_OBJ): $(SCHEDULING_SRC)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Link shared library
$(SHARED_LIB): $(ROUTING_OBJ) $(COMPRESSION_OBJ) $(SCHEDULING_OBJ)
	$(NVCC) $(NVCC_FLAGS) -shared $^ -o $@ $(LIBS)

# Clean build artifacts
clean:
	rm -f *.o *.so

# Install library to system path (optional)
install: $(SHARED_LIB)
	sudo cp $(SHARED_LIB) /usr/local/lib/
	sudo ldconfig

# Test compilation
test:
	@echo "Testing CUDA compilation..."
	$(NVCC) --version
	@echo "Available GPU architectures:"
	nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits

# Debug build
debug: NVCC_FLAGS += -g -G -DDEBUG
debug: $(SHARED_LIB)

# Profile build
profile: NVCC_FLAGS += -lineinfo -DPROFILE
profile: $(SHARED_LIB)

# Help
help:
	@echo "PiKV CUDA Kernels Build System"
	@echo "Available targets:"
	@echo "  all      - Build all kernels (default)"
	@echo "  clean    - Remove build artifacts"
	@echo "  install  - Install library to system"
	@echo "  test     - Test CUDA compilation"
	@echo "  debug    - Build with debug symbols"
	@echo "  profile  - Build with profiling info"
	@echo "  help     - Show this help"

.PHONY: all clean install test debug profile help 