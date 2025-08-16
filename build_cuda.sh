#!/bin/bash

# PiKV CUDA Build Script
# Automatically builds CUDA kernels with appropriate optimizations

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check CUDA installation
check_cuda() {
    print_status "Checking CUDA installation..."
    
    if ! command_exists nvcc; then
        print_error "nvcc not found. Please install CUDA toolkit."
        exit 1
    fi
    
    if ! command_exists nvidia-smi; then
        print_warning "nvidia-smi not found. CUDA drivers may not be installed."
    fi
    
    # Get CUDA version
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    print_success "CUDA version: $CUDA_VERSION"
    
    # Check GPU availability
    if command_exists nvidia-smi; then
        GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | wc -l)
        print_success "Found $GPU_COUNT GPU(s)"
        
        # Show GPU info
        nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits | while IFS=, read -r name memory compute_cap; do
            print_status "GPU: $name, Memory: $memory, Compute Cap: $compute_cap"
        done
    fi
}

# Function to build CUDA kernels
build_cuda() {
    print_status "Building CUDA kernels..."
    
    cd core/cuda
    
    # Clean previous builds
    print_status "Cleaning previous builds..."
    make clean
    
    # Build with release optimizations
    print_status "Building with release optimizations..."
    make release
    
    if [ $? -eq 0 ]; then
        print_success "CUDA kernels built successfully!"
        
        # Check if library was created
        if [ -f "libpikv_kernels.so" ]; then
            print_success "Library created: libpikv_kernels.so"
            ls -lh libpikv_kernels.so
        else
            print_error "Library not found after build!"
            exit 1
        fi
    else
        print_error "Build failed!"
        exit 1
    fi
    
    cd ../..
}

# Function to run tests
run_tests() {
    print_status "Running CUDA tests..."
    
    cd core/cuda
    
    if [ -f "test_pikv_kernels" ]; then
        print_status "Running test executable..."
        ./test_pikv_kernels
        
        if [ $? -eq 0 ]; then
            print_success "All tests passed!"
        else
            print_error "Some tests failed!"
            exit 1
        fi
    else
        print_warning "Test executable not found. Skipping tests."
    fi
    
    cd ../..
}

# Function to install library
install_library() {
    print_status "Installing CUDA library..."
    
    cd core/cuda
    
    if [ -f "libpikv_kernels.so" ]; then
        # Default installation path
        INSTALL_PATH="/usr/local"
        
        if [ "$1" != "" ]; then
            INSTALL_PATH="$1"
        fi
        
        print_status "Installing to $INSTALL_PATH"
        
        # Create directories
        sudo mkdir -p "$INSTALL_PATH/lib"
        sudo mkdir -p "$INSTALL_PATH/include"
        
        # Copy library
        sudo cp libpikv_kernels.so "$INSTALL_PATH/lib/"
        
        # Copy headers if they exist
        if [ -f "*.h" ]; then
            sudo cp *.h "$INSTALL_PATH/include/" 2>/dev/null || true
        fi
        
        # Update library cache
        if command_exists ldconfig; then
            sudo ldconfig
        fi
        
        print_success "Library installed successfully!"
    else
        print_error "Library not found. Please build first."
        exit 1
    fi
    
    cd ../..
}

# Function to show help
show_help() {
    echo "PiKV CUDA Build Script"
    echo "======================"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  build       Build CUDA kernels (default)"
    echo "  clean       Clean build artifacts"
    echo "  test        Build and run tests"
    echo "  install     Install library to system"
    echo "  install-user Install to user directory (~/local)"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0              # Build kernels"
    echo "  $0 test         # Build and test"
    echo "  $0 install      # Install to /usr/local"
    echo "  $0 install-user # Install to ~/local"
    echo ""
    echo "Environment variables:"
    echo "  CUDA_HOME    CUDA installation path"
    echo "  PREFIX       Installation prefix (for install)"
}

# Main script logic
main() {
    print_status "PiKV CUDA Build Script Starting..."
    
    # Check if we're in the right directory
    if [ ! -d "core/cuda" ]; then
        print_error "Please run this script from the PiKV root directory"
        exit 1
    fi
    
    # Check CUDA installation
    check_cuda
    
    # Parse command line arguments
    case "${1:-build}" in
        "build")
            build_cuda
            ;;
        "clean")
            print_status "Cleaning build artifacts..."
            cd core/cuda && make clean && cd ../..
            print_success "Clean completed!"
            ;;
        "test")
            build_cuda
            run_tests
            ;;
        "install")
            build_cuda
            install_library
            ;;
        "install-user")
            build_cuda
            install_library "$HOME/local"
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
    
    print_success "Build script completed successfully!"
}

# Run main function with all arguments
main "$@"
