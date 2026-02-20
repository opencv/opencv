# HIP Module Architecture

## Module Structure

```
modules/hip/
├── CMakeLists.txt              # Module build configuration
├── include/
│   └── opencv2/hip/
│       ├── hip_config.hpp      # Configuration & GPU detection
│       ├── hip_dispatcher.hpp  # GPU memory & device management
│       └── hip_kernels.hpp     # GPU kernel declarations
├── src/
│   ├── hip_config.cpp          # Configuration implementation
│   ├── hip_dispatcher.cpp      # Device management implementation
│   ├── hip_kernels.cpp         # Kernel wrappers
│   ├── hip_gaussian_blur.cpp   # Gaussian Blur GPU/CPU dispatch
│   ├── hip_resize.cpp          # Resize GPU/CPU dispatch
│   └── hip_color_convert.cpp   # Color conversion GPU/CPU dispatch
├── test/
│   ├── test_precomp.hpp        # Test precompiled headers
│   └── test_hip_kernels.cpp    # Unit tests
├── samples/
│   ├── hip_gaussian_blur_demo.cpp    # Demo application
│   └── hip_benchmark.cpp              # Performance benchmarks
├── doc/
│   └── hip.rst                 # Documentation source
├── README.md                    # Module overview
├── INSTALLATION.md             # Installation guide
├── INTEGRATION_GUIDE.md        # Adding new operations
└── HIP_MODULE_ARCHITECTURE.md  # This file
```

## Component Overview

### 1. Configuration Layer (`hip_config.hpp`)

**Purpose**: Global GPU configuration and device detection

**Key Classes/Functions**:
- `GPUConfig`: Settings for GPU usage thresholds
- `getGPUConfig()`: Access global configuration
- `isGPUAvailable()`: Check if GPU is ready
- `shouldUseGPU()`: Determine if operation should use GPU
- `setGPUEnabled()`: Toggle GPU globally

**Data Flow**:
```
┌─────────────────┐
│ User Code       │
└────────┬────────┘
         │
    Calls getGPUConfig()
         │
    ┌────▼────────────────────┐
    │ GPU Config Manager      │
    │ - min_image_size_bytes  │
    │ - min_flops_per_element │
    │ - enabled               │
    │ - fallback_to_cpu       │
    └─────────────────────────┘
```

### 2. Device Management Layer (`hip_dispatcher.hpp`)

**Purpose**: GPU memory management and device operations

**Key Classes**:
- `HIPException`: Exception wrapper for HIP errors
- `GPUMemory`: GPU memory allocation/deallocation
- `GPUDevice`: Device enumeration and control
- `Dispatcher<Func>`: Template for CPU/GPU function dispatch

**Memory Management**:
```
Host Memory (RAM)
    │
    ├─ Host Ptr (pinned)
    │      │
    │      ├─ hipMemcpyHtoDAsync ───┐
    │      │                         │
    └──────┼─────────────────────────┼──────► GPU Memory
           │                         │
           │        GPU Computation  │
           │                         │
           └─ hipMemcpyDtoHAsync ◄───┘
                      │
                      ▼
           Output Host Ptr
```

**Device Control Functions**:
- `getDeviceCount()`: Number of available GPUs
- `selectDevice()`: Choose GPU for operations
- `synchronize()`: Wait for GPU operations
- `getFreeMemory()`: Query available GPU memory

### 3. Kernel Layer (`hip_kernels.hpp`)

**Purpose**: GPU kernel implementations and wrappers

**Supported Operations**:
```
┌──────────────────────────────────────────┐
│        GPU-Accelerated Operations        │
├──────────────────────────────────────────┤
│ • gaussianBlur_gpu      (image filtering)│
│ • resize_gpu            (resampling)     │
│ • cvtColor_gpu          (color space)    │
│ • bilateralFilter_gpu   (edge preserving)│
│ • morphOp_gpu           (morphology)     │
│ • Canny_gpu             (edge detection) │
│ • calcHist_gpu          (histogram)      │
│ • adjustBrightnessContrast_gpu           │
└──────────────────────────────────────────┘
```

### 4. Dispatcher Implementation

Each GPU function follows this pattern:

```cpp
namespace cv::hip {
    // 1. CPU fallback implementation
    void operation_cpu(const Mat& src, Mat& dst, ...) {
        cv::operation(src, dst, ...);  // Use OpenCV CPU version
    }
    
    // 2. GPU implementation  
    void operation_gpu_impl(const Mat& src, Mat& dst, ...) {
        // Allocate GPU memory
        // Upload input data
        // Launch HIP kernel
        // Download results
    }
    
    // 3. Public API with dispatch
    void operation_gpu(InputArray src, OutputArray dst, ...) {
        // Check if GPU is viable
        if (shouldUseGPU(image_size, flops)) {
            // Try GPU with fallback
            try {
                operation_gpu_impl(...);
            } catch (...) {
                if (fallback_enabled) {
                    operation_cpu(...);
                }
            }
        } else {
            // Use CPU directly
            operation_cpu(...);
        }
    }
}
```

## Execution Flow Diagram

### Typical GPU Operation Sequence

```
User Code
    │
    ├─ Input: Mat src, parameters
    │
    ▼
cv::hip::gaussianBlur_gpu()
    │
    ├─ Calculate image size in bytes
    │
    ├─ Estimate FLOPs per element
    │
    ├─ Call shouldUseGPU(size, flops)
    │      │
    │      ├─ Check min_image_size_bytes? ──Yes──┐
    │      │                                      │
    │      ├─ Check min_flops_per_element? ──Yes──┤
    │      │                                      │
    │      ├─ Check free GPU memory? ─────Yes────┤
    │      │                                      │
    │      └─ Return: true/false                 │
    │                                             │
    ├─ if (shouldUseGPU && hasGPU)              │
    │  │                          No──┐          │
    │  ├─ Try operation_gpu_impl()    │         │
    │  │  │                           │         │
    │  │  ├─ GPUMemory::allocate()    │         │
    │  │  │  └─ hipMalloc(d_src, size)│         │
    │  │  │  └─ hipMalloc(d_dst, size)│         │
    │  │  │                           │         │
    │  │  ├─ GPUMemory::upload()      │         │
    │  │  │  └─ hipMemcpyHtoDAsync()  │         │
    │  │  │                           │         │
    │  │  ├─ Launch HIP Kernel        │         │
    │  │  │  └─ hipLaunchKernelGGL()  │         │
    │  │  │                           │         │
    │  │  ├─ GPUMemory::download()    │         │
    │  │  │  └─ hipMemcpyDtoHAsync()  │         │
    │  │  │                           │         │
    │  │  └─ Return success           │         │
    │  │                              │         │
    │  └─ catch (exception)           │         │
    │     │                            │         │
    │     └─ if (fallback_enabled) ───┴─Yes──┐ │
    │                                        │ │
    ├─ else / fallback path:                 │ │
    │  └─ Call operation_cpu()        ◄──────┘ │
    │     └─ Use OpenCV CPU version          │
    │                                        │
    └─ Assign result to output Mat     ◄─────┘
```

## Memory Management Strategy

### Async Memory Transfers

```
Timeline:
─────────────────────────────────────────────────────

Host CPU         │ Upload      │ Kernel    │ Download   │
           Upload│ Complete    │ Execution │ Complete   │
                 ▼             ▼           ▼            ▼
Stream    ───────────────────────────────────────────────→

GPU               │<─ Transfers ─→│ Compute  │<─ Transfer→
                  │                │◄────────►│
                  │                │          │
                 Copy            Kernel     Copy
                H2D              Exec       D2H
```

**Advantages**:
- GPU can overlap computation with data transfer
- No blocking calls on host
- Maximum throughput utilization

### Memory Pooling (Future)

```cpp
// Future implementation concept
class GPUMemoryPool {
    std::unordered_map<size_t, std::vector<GPUMemory>> pool;
    
    GPUMemory allocate(size_t size) {
        // Reuse from pool if available
        // Otherwise allocate new
    }
    
    void deallocate(GPUMemory& mem) {
        // Return to pool instead of freeing
    }
};
```

## Performance Characteristics

### Overhead Breakdown

For a typical GPU operation:

```
Total Time = Upload + Kernel + Download

Upload (H2D):     ~1-2 ms per 10MB
Kernel Exec:      Varies (50% CPU for large ops)
Download (D2H):   ~1-2 ms per 10MB
──────────────────────────────────
Min Overhead:     ~2-4 ms

Therefore, GPU worth it when:
Kernel Exec Time > 2-4 ms
```

### Example: Gaussian Blur

```
5×5 Kernel, 100MB image:

            Upload  │ Kernel │ Download │ Total
            2.2 ms  │ 12 ms  │ 2.1 ms   │ 16.3 ms
CPU:        ────────────────────────────│ 85 ms
────────────────────────────────────────────────
Speedup: 85 / 16.3 ≈ 5.2x ✓
```

## Dispatch Decision Tree

```
┌─ Operation called
│
├─ Is GPU available? ──────No──┐
│  Yes                         │
│   │                         │
├─ Image > min_size? ──No──┬──┤
│  Yes                    │  │
│   │                     │  │
├─ Flops > min_density?──No┤  │
│  Yes                     │  │
│   │                      │  │
├─ Free memory OK? ──No────┤  │
│  Yes                      │  │
│   │                       │  │
├─ Try GPU Operation        │  │
│  │                        │  │
│  ├─ Success? ──Yes──┐    │  │
│  │                  │    │  │
│  ├─ Failure ──┐    │    │  │
│     │         │    │    │  │
│     └─ Fallback   │    │  │
│        Enabled? ──┴────┴──┴─→ Use CPU
│        No? ───────────────→ Throw Exception
│        Yes? ──────────────→ Use CPU
│                            │
└────────────────────────────→ Return Result
```

## Error Handling Strategy

```cpp
try {
    // GPU operation attempt
    gpu_operation();
    return;
} catch (const HIPException& e) {
    // GPU-specific error
    switch (e.hip_error) {
        case hipErrorInvalidDevice:
            // Invalid device selection
            break;
        case hipErrorMemoryAllocation:
            // Not enough GPU memory
            break;
        case hipErrorHostMemoryAllocation:
            // Host memory issue
            break;
        default:
            // Generic GPU error
            break;
    }
    
    if (getGPUConfig().fallback_to_cpu) {
        // Transparently use CPU
        cpu_operation();
    } else {
        // Propagate error
        throw;
    }
} catch (const std::exception& e) {
    // General error
    if (getGPUConfig().fallback_to_cpu) {
        cpu_operation();
    } else {
        throw;
    }
}
```

## Testing Strategy

### Unit Tests
- Verify GPU results match CPU within tolerance
- Test different image sizes
- Test different configurations
- Test error conditions

### Performance Tests
- Benchmark against CPU implementation
- Measure GPU utilization
- Profile memory usage
- Find break-even points

### Integration Tests
- Test with typical OpenCV workflows
- Verify compatibility with other modules
- Test GPU/CPU fallback scenarios

## Future Enhancements

```
Current (Phase 1)          Future (Phase 2-3)
─────────────────          ──────────────────

Single Operations    -->   Kernel Fusion
GPU/CPU Dispatch    -->   Multi-GPU Distribution
Static Thresholds   -->   Adaptive Thresholds
Manual Tuning       -->   Auto-Tuning
One-shot Ops       -->   Batch Processing
```

## Dependencies

```
HIP Module
    │
    ├─ OpenCV Core
    │  ├─ Mat, UMat data structures
    │  └─ Memory management
    │
    ├─ OpenCV imgproc
    │  ├─ Image processing operations
    │  └─ CPU reference implementations
    │
    ├─ HIP Threads (ROCm)
    │  ├─ hip::thread primitives
    │  └─ GPU memory management
    │
    └─ HIP Runtime
       ├─ hipMalloc, hipMemcpy
       └─ kernel launches
```

## Compilation Process

```
Source (.cpp, .hpp)
    │
    ├─ Preprocessing
    │  └─ Expand macros, includes
    │
    ├─ Parsing
    │  └─ Syntax analysis
    │
    ├─ Semantic Analysis
    │  └─ Type checking
    │
    ├─ HIP Compilation (hipcc)
    │  └─ Generate device code for target GPU
    │
    ├─ Codegen
    │  └─ Generate machine code
    │
    └─ Linking
       └─ Link with HIP runtime libraries
           │
           ▼
       libopencv_hip.so
```

## Performance Monitoring

```cpp
// Concept for future performance monitoring
class GPUMetrics {
    struct Measurement {
        std::string operation;
        double gpu_time_ms;
        double cpu_time_ms;
        double gpu_memory_mb;
        double speedup;
    };
    
    std::vector<Measurement> measurements;
    
    void log(const Measurement& m) {
        measurements.push_back(m);
    }
    
    void reportSummary() {
        // Print performance analysis
    }
};
```

This architecture provides a clean separation of concerns, modular design, and easy extensibility for adding new GPU-accelerated operations.
