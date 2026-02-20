# HIP Threads Module Installation Guide

## Quick Start

### 1. Install ROCm 7.0.2

This is the version required by HIP Threads. Other versions are not currently supported.

#### On Ubuntu 24.04 (Recommended)

```bash
# Step 1: Meet kernel prerequisites
# ROCm requires kernel 6.0+. Check your kernel:
uname -r

# Update if necessary
sudo apt update && sudo apt upgrade -y

# Step 2: Install AMD GPU kernel driver
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -

# Add repository (IMPORTANT: Use 7.0.2 repo URLs)
sudo tee /etc/apt/sources.list.d/amdgpu.list > /dev/null <<EOF
deb [arch=amd64] https://repo.radeon.com/amdgpu/ubuntu focal main
EOF

# Install AMDGPU driver
sudo apt update
sudo apt install -y amdgpu-dkms
sudo reboot  # Required!

# Step 3: Install ROCm 7.0.2 (CRITICAL VERSION)
# Remove any existing ROCm first
sudo apt remove -y rocm-core rocm-hip-runtime rocm-hip-devel hipcc

# Use specific 7.0.2 repository
sudo tee /etc/apt/sources.list.d/rocm.list > /dev/null <<EOF
deb [arch=amd64] https://repo.radeon.com/rocm/apt/7.0.2 focal main
EOF

# Install
sudo apt update
sudo apt install -y rocm-hip-runtime-7.0.2 rocm-hip-devel-7.0.2 hipcc-7.0.2

# Post-installation setup
sudo usermod -aG render $USER
sudo usermod -aG video $USER
sudo reboot  # Required!

# Verify installation
rocm-smi
hipcc --version
```

#### On Ubuntu 22.04

```bash
sudo tee /etc/apt/sources.list.d/rocm.list > /dev/null <<EOF
deb [arch=amd64] https://repo.radeon.com/rocm/apt/7.0.2 jammy main
EOF

sudo apt update
sudo apt install -y rocm-hip-runtime-7.0.2 rocm-hip-devel-7.0.2 hipcc-7.0.2
sudo usermod -aG render $USER
sudo usermod -aG video $USER
sudo reboot
```

### 2. Install HIP Threads Library

```bash
# Clone HIP Threads repository
git clone https://github.com/ROCm/hipThreads.git
cd hipThreads

# Verify ROCm version
rocm-smi --query

# Build and install
cmake -B build \
  -DCMAKE_INSTALL_PREFIX=/opt/rocm \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build -j$(nproc)
sudo cmake --install build

# Verify installation
ls /opt/rocm/include/hip/hip_threads.hpp
ls /opt/rocm/lib/libhipthreads*
```

### 3. Build OpenCV with HIP Support

```bash
cd opencv
mkdir build && cd build

# Configure with HIP support
cmake .. \
  -DWITH_HIP=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DHIP_PATH=/opt/rocm \
  -DCMAKE_INSTALL_PREFIX=/usr/local

# Optional: specify target GPU architecture
# Use 'rocminfo' to find yours, then add:
#   -DHIP_ARCHITECTURES=gfx1036,gfx1201

# Build OpenCV
make -j$(nproc)

# Install
sudo make install
```

## Verification

### Test GPU Detection

```cpp
#include <opencv2/hip/hip_config.hpp>
#include <iostream>

int main() {
    if (cv::hip::isGPUAvailable()) {
        std::cout << "GPU is available!\n";
        auto& config = cv::hip::getGPUConfig();
        std::cout << "Min image size: " 
                  << (config.min_image_size_bytes / (1024*1024)) << " MB\n";
    } else {
        std::cout << "GPU not available - using CPU only\n";
    }
    return 0;
}
```

Compile and test:
```bash
g++ test.cpp -o test -I/usr/local/include/opencv2 \
  -L/usr/local/lib -lopencv_core -lopencv_hip
./test
```

### Run Unit Tests

```bash
cd opencv/build
ctest -R hip -V
```

### Run Benchmarks

```bash
# If samples were built
./bin/hip_benchmark
./bin/hip_gaussian_blur_demo image.jpg
```

## Troubleshooting

### Problem: "rocm-smi: command not found"

```bash
# ROCm tools not in PATH
export PATH=$PATH:/opt/rocm/bin
source /opt/rocm/setup-env.sh
```

### Problem: "hipcc: command not found"

```bash
# Set PATH to hipcc
export PATH=/opt/rocm/bin:$PATH
which hipcc
```

### Problem: "Permission denied" when loading GPU

```bash
# User not in render group
sudo usermod -aG render $USER
sudo usermod -aG video $USER
# Log out and back in, or restart
newgrp render
```

### Problem: HIP Threads not found during CMake

```bash
# CMake can't find hipthreads
cmake .. \
  -DCMAKE_PREFIX_PATH="/opt/rocm;/opt/rocm/hipthreads"

# Or specify explicitly
cmake .. \
  -Dhipthreads_DIR=/opt/rocm/lib/cmake/hipthreads
```

### Problem: "Wrong ROCm version" error

```bash
# Check installed version
rocm-smi --query | grep "ROCm version"

# Currently required version
echo "ROCm 7.0.2 required by HIP Threads"

# If you have wrong version:
sudo apt remove -y rocm-*
# Then reinstall correct version from step 1
```

### Problem: GPU operations are slow

```bash
# Might be using CPU fallback
#include <opencv2/hip/hip_config.hpp>

auto& config = cv::hip::getGPUConfig();
config.verbose = true;  // Enable logging
// Re-run to see dispatch decisions
```

### Problem: Segmentation fault in GPU operations

```bash
# Likely GPU memory issue
auto& config = cv::hip::getGPUConfig();
// Increase minimum image size threshold
config.min_image_size_bytes = 100 * 1024 * 1024;  // 100MB

// Or disable GPU temporarily
config.enabled = false;
```

## Advanced Configuration

### CMake Build Options

```bash
cmake .. \
  -DWITH_HIP=ON                          # Enable HIP module \
  -DHIP_PATH=/opt/rocm                   # ROCm installation path \
  -DHIP_COMPILER=hipcc                   # HIP compiler executable \
  -DHIP_ARCHITECTURES=gfx1036,gfx1201    # Target GPU architectures \
  -DCMAKE_BUILD_TYPE=Release             # Optimization level \
  -DBUILD_TESTS=ON                       # Build unit tests \
  -DBUILD_PERF_TESTS=ON                  # Build performance tests
```

### Finding Your GPU Architecture

```bash
rocminfo | grep "Marketing Name"
# Then map to gfx* value:

# Common mappings:
# Radeon RX 7600 XT -> gfx1201
# Radeon RX 6800 XT -> gfx908
# Radeon RX 5700 XT -> gfx906
# Radeon RX Vega 64 -> gfx900

# Or search ROCm documentation for your model
```

### Runtime Configuration

Set environment variables before running:

```bash
# Enable verbose GPU logging
export OPENCV_HIP_VERBOSE=1

# Force GPU (no fallback)
export OPENCV_HIP_FALLBACK=0

# Force CPU (disable GPU)
export OPENCV_HIP_DISABLE=1

# Change GPU device
export HIP_DEVICE=0  # Use GPU 0

# Memory configuration
export HIP_MALLOC_COHERENT_HOST=1  # Coherent GPU-Host memory
```

## Multi-GPU Setup

### Detect Available GPUs

```cpp
#include <opencv2/hip/hip_dispatcher.hpp>

int num_gpus = cv::hip::GPUDevice::getDeviceCount();
std::cout << "Available GPUs: " << num_gpus << "\n";

// Select specific GPU
cv::hip::GPUDevice::selectDevice(0);

// Query GPU memory
size_t free = cv::hip::GPUDevice::getFreeMemory();
size_t total = cv::hip::GPUDevice::getTotalMemory();
std::cout << "Free: " << (free / (1024*1024)) << " MB\n";
std::cout << "Total: " << (total / (1024*1024)) << " MB\n";
```

## Performance Tuning

### Environment Variables

```bash
# Kernel launch parameters
export HIP_LAUNCH_BLOCKING=1  # Synchronous launches (slower but safer)

# Memory allocation strategy
export HIP_USE_UNIFIED_MEMORY=1  # Use unified memory model

# Compiler optimization
export HIPCC_VERBOSE=1  # See compilation details
```

### Build Optimizations

```bash
cmake .. \
  -DCMAKE_CXX_FLAGS="-O3 -march=native" \
  -DCMAKE_BUILD_TYPE=Release
```

## Docker Setup (Alternative)

If you prefer containerized development:

```dockerfile
FROM rocm/rocm-terminal:7.0.2

RUN apt update && apt install -y \
    cmake \
    git \
    build-essential \
    rocm-hip-devel \
    hipcc

WORKDIR /workspace

# Clone and build HIP Threads
RUN git clone https://github.com/ROCm/hipThreads.git && \
    cd hipThreads && \
    cmake -B build && \
    cmake --build build && \
    cmake --install build

# Clone and build OpenCV
RUN git clone https://github.com/opencv/opencv.git && \
    cd opencv && \
    mkdir build && cd build && \
    cmake .. -DWITH_HIP=ON && \
    make -j$(nproc) && \
    make install
```

Build and run:
```bash
docker build -t opencv-hip .
docker run --rm --device=/dev/kfd --device=/dev/dri \
    -it opencv-hip /bin/bash
```

## Next Steps

1. **Try Examples**: Run the samples in `modules/hip/samples/`
2. **Read Documentation**: See `modules/hip/README.md`
3. **Run Benchmarks**: Use `hip_benchmark` to measure speedup
4. **Integrate into Your Code**: Use `cv::hip::` functions in your applications
5. **Contributing**: See `INTEGRATION_GUIDE.md` for adding new operations

## Support & Resources

- **HIP Threads Documentation**: https://github.com/ROCm/hipThreads
- **HIP Documentation**: https://rocm.docs.amd.com/projects/HIP/en/latest/
- **ROCm Support**: https://rocm.docs.amd.com/
- **OpenCV Issues**: https://github.com/opencv/opencv/issues
- **ROCm Community**: https://community.amd.com/t5/ROCm-Blogs/ct-p/rocm-blogs

## License

OpenCV HIP module is distributed under Apache License v2.0 with LLVM Exceptions.
