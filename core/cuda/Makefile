# PiKV CUDA Kernels Makefile
# Supports multiple CUDA versions and optimization levels

# Compiler and flags
NVCC = nvcc
CC = gcc
CXX = g++

# CUDA version detection
CUDA_VERSION := $(shell nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
CUDA_MAJOR := $(shell echo $(CUDA_VERSION) | cut -d. -f1)
CUDA_MINOR := $(shell echo $(CUDA_VERSION) | cut -d. -f2)

# Architecture flags
ARCH_FLAGS = -arch=sm_60 -arch=sm_70 -arch=sm_75 -arch=sm_80 -arch=sm_86 -arch=sm_89

# Optimization levels
DEBUG_FLAGS = -g -G -O0 -DDEBUG
RELEASE_FLAGS = -O3 -DNDEBUG
PROFILE_FLAGS = -O3 -g -G -lineinfo -DNDEBUG

# Default flags
NVCC_FLAGS = $(ARCH_FLAGS) -std=c++17 -Xcompiler -fPIC
CXX_FLAGS = -std=c++17 -fPIC -Wall -Wextra

# Libraries
CUDA_LIBS = -lcudart -lcublas -lcusparse
THRUST_LIBS = -I/usr/local/cuda/include

# Source files
CUDA_SOURCES = pikv_kernels.cu
CXX_SOURCES = test_kernels.cpp

# Object files
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)
CXX_OBJECTS = $(CXX_SOURCES:.cpp=.o)

# Target library
TARGET_LIB = libpikv_kernels.so
TARGET_STATIC = libpikv_kernels.a

# Test executable
TEST_EXEC = test_pikv_kernels

# Default target
.PHONY: all
all: $(TARGET_LIB) $(TEST_EXEC)

# Debug build
.PHONY: debug
debug: NVCC_FLAGS += $(DEBUG_FLAGS)
debug: CXX_FLAGS += -g -O0 -DDEBUG
debug: all

# Release build
.PHONY: release
release: NVCC_FLAGS += $(RELEASE_FLAGS)
release: CXX_FLAGS += -O3 -DNDEBUG
release: all

# Profile build
.PHONY: profile
profile: NVCC_FLAGS += $(PROFILE_FLAGS)
profile: CXX_FLAGS += -O3 -g -DNDEBUG
profile: all

# Build shared library
$(TARGET_LIB): $(CUDA_OBJECTS)
	@echo "Linking shared library: $@"
	$(NVCC) $(NVCC_FLAGS) -shared -o $@ $^ $(CUDA_LIBS)

# Build static library
$(TARGET_STATIC): $(CUDA_OBJECTS)
	@echo "Creating static library: $@"
	ar rcs $@ $^

# Build test executable
$(TEST_EXEC): $(CXX_OBJECTS) $(TARGET_LIB)
	@echo "Linking test executable: $@"
	$(CXX) $(CXX_FLAGS) -o $@ $(CXX_OBJECTS) -L. -lpikv_kernels $(CUDA_LIBS)

# Compile CUDA source files
%.o: %.cu
	@echo "Compiling CUDA: $<"
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile C++ source files
%.o: %.cpp
	@echo "Compiling C++: $<"
	$(CXX) $(CXX_FLAGS) -c $< -o $@

# Install targets
.PHONY: install
install: $(TARGET_LIB)
	@echo "Installing libraries..."
	@mkdir -p $(PREFIX)/lib
	@mkdir -p $(PREFIX)/include
	cp $(TARGET_LIB) $(PREFIX)/lib/
	cp *.h $(PREFIX)/include/ 2>/dev/null || true

# Clean targets
.PHONY: clean
clean:
	@echo "Cleaning build files..."
	rm -f *.o *.so *.a $(TEST_EXEC)

.PHONY: distclean
distclean: clean
	@echo "Cleaning all generated files..."
	rm -f *.so *.a $(TEST_EXEC)

# Test targets
.PHONY: test
test: $(TEST_EXEC)
	@echo "Running tests..."
	./$(TEST_EXEC)

.PHONY: test-cuda
test-cuda: $(TEST_EXEC)
	@echo "Running CUDA tests..."
	CUDA_VISIBLE_DEVICES=0 ./$(TEST_EXEC)

# Performance profiling
.PHONY: profile-kernels
profile-kernels: $(TEST_EXEC)
	@echo "Profiling CUDA kernels..."
	nvprof --metrics all ./$(TEST_EXEC)

.PHONY: profile-memory
profile-memory: $(TEST_EXEC)
	@echo "Profiling memory usage..."
	nvprof --print-gpu-trace --print-api-trace ./$(TEST_EXEC)

# Documentation
.PHONY: docs
docs:
	@echo "Generating documentation..."
	doxygen Doxyfile 2>/dev/null || echo "Doxygen not found, skipping documentation"

# Help target
.PHONY: help
help:
	@echo "PiKV CUDA Kernels Makefile"
	@echo "=========================="
	@echo ""
	@echo "Available targets:"
	@echo "  all          - Build shared library and test executable (default)"
	@echo "  debug        - Build with debug symbols and no optimization"
	@echo "  release      - Build with full optimization"
	@echo "  profile      - Build with profiling information"
	@echo "  install      - Install libraries and headers"
	@echo "  test         - Run tests"
	@echo "  test-cuda    - Run tests with CUDA device 0"
	@echo "  profile-kernels - Profile kernel performance"
	@echo "  profile-memory   - Profile memory usage"
	@echo "  clean        - Remove object files and libraries"
	@echo "  distclean    - Remove all generated files"
	@echo "  docs         - Generate documentation"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Environment variables:"
	@echo "  PREFIX       - Installation prefix (default: /usr/local)"
	@echo "  CUDA_HOME    - CUDA installation path"
	@echo ""
	@echo "Examples:"
	@echo "  make release        # Build optimized version"
	@echo "  make test-cuda      # Run tests on GPU"
	@echo "  make install PREFIX=~/local  # Install to user directory"

# Check CUDA installation
.PHONY: check-cuda
check-cuda:
	@echo "CUDA Version: $(CUDA_VERSION)"
	@echo "CUDA Major: $(CUDA_MAJOR)"
	@echo "CUDA Minor: $(CUDA_MINOR)"
	@nvcc --version | head -1
	@nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1

# Dependencies
-include $(CUDA_OBJECTS:.o=.d)
-include $(CXX_OBJECTS:.o=.d)

# Generate dependency files
%.d: %.cu
	@set -e; rm -f $@; \
	$(NVCC) $(NVCC_FLAGS) -M $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

%.d: %.cpp
	@set -e; rm -f $@; \
	$(CXX) $(CXX_FLAGS) -M $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$ 