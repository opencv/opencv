Vectorizing your code using Universal Intrinsics {#tutorial_univ_intrin}
==================================================================

@tableofcontents

@prev_tutorial{tutorial_how_to_use_OpenCV_parallel_for_new}

|    |    |
| -: | :- |
| Compatibility | OpenCV >= 4.11 |

Goal
----

The goal of this tutorial is to provide a guide to using the @ref core_hal_intrin feature to vectorize your C++ code for a faster runtime.
We'll briefly look into _SIMD intrinsics_ and how to work with wide _registers_, followed by a tutorial on the basic operations using wide registers.

Theory
------

In this section, we will briefly look into a few concepts to better help understand the functionality.

### Intrinsics
Intrinsics are functions which are separately handled by the compiler. These functions are often optimized to perform in the most efficient ways possible and hence run faster than normal implementations. However, since these functions depend on the compiler, it makes it difficult to write portable applications.

### SIMD
SIMD stands for **Single Instruction, Multiple Data**. SIMD Intrinsics allow the processor to vectorize calculations. The data is stored in what are known as *registers*. A *register* may be *128-bits*, *256-bits* or *512-bits* wide. Each *register* stores **multiple values** of the **same data type**. The size of the register and the size of each value determines the number of values stored in total.

Depending on what *Instruction Sets* your CPU supports, you may be able to use the different registers. To learn more, look [here](https://en.wikipedia.org/wiki/Instruction_set_architecture)

### VLA
VLA stands for **Vector Length Agnostic** .
A mechanism where the register width is determined by the hardware at runtime rather than being fixed at compile time.
This allows a single binary to scale its performance across different CPUs within the same architecture (e.g., RVV or SVE).

Universal Intrinsics
--------------------

OpenCV's universal intrinsics provides an abstraction to SIMD and VLA vectorization methods and allows the user to use intrinsics without the need to write system specific code.
Supported SIMD/VLA technologies are detailed in @ref core_hal_intrin .

**We will now introduce the available structures and functions:**
* Register structures
* Load and store
* Mathematical Operations
* Reduce and Mask
* 16-bit float arithmetic (FP16 and BF16)
* Math function intrinsics
* Multi-channel operations

### Register Structures

The Universal Intrinsics set implements every register as a structure based on the particular SIMD register.
All types contain the `nlanes` enumeration which gives the exact number of values that the type can hold. This eliminates the need to hardcode the number of values during implementations.

@note Each register structure is under the `cv` namespace.

There are **two types** of registers:

* **Variable sized registers**: These structures do not have a fixed size and their exact bit length is deduced during compilation, based on the available SIMD capabilities. Consequently, the value of the `nlanes` enum is determined in compile time.
    <br>

    Each structure follows the following convention:

        v_[type of value][size of each value in bits]

    For instance, **v_uint8 holds 8-bit unsigned integers** and **v_float32 holds 32-bit floating point values**. We then declare a register like we would declare any object in C++

    Based on the available SIMD instruction set, a particular register will hold different number of values.
    For example: If your computer supports a maximum of 256bit registers,
    * *v_uint8* will hold 32 8-bit unsigned integers
    * *v_float64* will hold 4 64-bit floats (doubles)

            v_uint8 a;                            // a is a register supporting uint8(char) data
            int n = a.nlanes;                     // n holds 32

    Available data type and sizes:
    |Type|Size in bits|
    |-:|:-|
    |uint| 8, 16, 32, 64|
    |int | 8, 16, 32, 64|
    |float | 16, 32, 64|
    |bfloat | 16|

    @note The 16-bit floating-point types (`v_float16`, `v_bfloat16`) require their SIMD guards to be enabled - see the [FP16 and BF16 Arithmetic](#fp16-and-bf16-arithmetic) section below.

* **Constant sized registers**: These structures have a fixed bit size and hold a constant number of values. We need to know what SIMD instruction set is supported by the system and select compatible registers. Use these only if exact bit length is necessary.
    <br>

    Each structure follows the convention:

        v_[type of value][size of each value in bits]x[number of values]

    Suppose we want to store
    * 32-bit(*size in bits*) signed integers in a **128 bit register**. Since the register size is already known, we can find out the *number of data points in register* (*128/32 = 4*):

            v_int32x8 reg1                       // holds 8 32-bit signed integers.

    * 64-bit floats in 512 bit register:

            v_float64x8 reg2                     // reg2.nlanes = 8

### Load and Store operations

Now that we know how registers work, let us look at the functions used for filling these registers with values.

* **Load**: Load functions allow you to *load* values into a register.
    * *Constructors* - When declaring a register structure, we can either provide a memory address from where the register will pick up contiguous values, or provide the values explicitly as multiple arguments (Explicit multiple arguments is available only for Constant Sized Registers):

            float ptr[32] = {1, 2, 3 ..., 32};   // ptr is a pointer to a contiguous memory block of 32 floats

            // Variable Sized Registers //
            int x = v_float32().nlanes;          // set x as the number of values the register can hold

            v_float32 reg1(ptr);                 // reg1 stores first x values according to the maximum register size available.
            v_float32 reg2(ptr + x);             // reg stores the next x values

            // Constant Sized Registers //
            v_float32x4 reg1(ptr);               // reg1 stores the first 4 floats (1, 2, 3, 4)
            v_float32x4 reg2(ptr + 4);           // reg2 stores the next 4 floats (5, 6, 7, 8)

            // Or we can explicitly write down the values.
            v_float32x4(1, 2, 3, 4);

    * *Load Function* - We can use the load method and provide the memory address of the data:

            float ptr[32] = {1, 2, 3, ..., 32};
            v_float32 reg_var;
            reg_var = vx_load(ptr);              // loads values from ptr[0] upto ptr[reg_var.nlanes - 1]

            v_float32x4 reg_128;
            reg_128 = v_load(ptr);               // loads values from ptr[0] upto ptr[3]

            v_float32x8 reg_256;
            reg_256 = v256_load(ptr);            // loads values from ptr[0] upto ptr[7]

            v_float32x16 reg_512;
            reg_512 = v512_load(ptr);            // loads values from ptr[0] upto ptr[15]

        @note The load function assumes data is unaligned. If your data is aligned, you may use the `vx_load_aligned()` function.
    <br>

* **Store**: Store functions allow you to *store* the values from a register into a particular memory location.
    * To store values from a register into a memory location, you may use the *v_store()* function:

            float ptr[4];
            v_store(ptr, reg); // store the first 128 bits(interpreted as 4x32-bit floats) of reg into ptr.
<br>
@note Ensure **ptr** has the same type as register. You can also cast the register into the proper type before carrying out operations. Simply typecasting the pointer to a particular type will lead wrong interpretation of data.

### Binary and Unary Operators

The universal intrinsics set provides element wise binary and unary operations.

@note Since OpenCV 4.11, C++ operator overloading (e.g., +, ) in Universal Intrinsics has been deprecated in favor of explicit wrapper functions (e.g., v_add, v_mul) to ensure compatibility with VLA architectures.
See also: https://github.com/opencv/opencv/issues/27267

* **Arithmetics**: We can add, subtract, multiply and divide two registers element-wise. The registers must be of the same width and hold the same type. To multiply two registers, for example:

        v_float32 a, b;                          // {a1, ..., an}, {b1, ..., bn}
        v_float32 c = v_add(a, b);               // {a1 + b1, ..., an + bn}
        v_flaot32 d = v_mul(a, b);               // {a1 * b1, ..., an * bn}

<br>

* **Bitwise Logic and Shifts**: We can left shift or right shift the bits of each element of the register. We can also apply bitwise and, or, xor and not operators between two registers element-wise:

        v_int32 as;                              // {a1, ..., an}
        v_int32 al = v_shl(as, 2);               // {a1 << 2, ..., an << 2}
        v_int32 bl = v_shr(as, 2);               // {a1 >> 2, ..., an >> 2}

        v_int32 a, b;
        v_int32 a_and_b = v_and(a, b);           // {a1 & b1, ..., an & bn}

<br>

* **Comparison Operators**: We can compare values between two registers using the v_lt(<), v_gt(>), v_le(<=) , v_ge(>=), v_eq(==) and v_ne(!=). Since each register contains multiple values, we don't get a single bool for these operations. Instead, for true values, all bits are converted to one (0xff for 8 bits, 0xffff for 16 bits, etc), while false values return bits converted to zero.

        // let us consider the following code is run in a 128-bit register
        v_uint8 a;                               // a = {0, 1, 2, ..., 13, 14, 15}
        v_uint8 b;                               // b = {15, 14, 13, ..., 2, 1, 0}

        v_uint8 c = v_lt(a, b);                  // c = {255, 255, 255, ..., 0, 0, 0}

        /*
            let us look at the first 4 values in binary

            a = |00000000|00000001|00000010|00000011|
            b = |00001111|00001110|00001101|00001100|
            c = |11111111|11111111|11111111|11111111|

            If we store the values of c and print them as integers, we will get 255 for true values and 0 for false values.
        */
        ---
        // In a computer supporting 256-bit registers
        v_int32 a;                               // a = {1, 2, 3, 4, 5, 6, 7, 8}
        v_int32 b;                               // b = {8, 7, 6, 5, 4, 3, 2, 1}

        v_int32 c = v_lt(a, b);                  // c = {-1, -1, -1, -1, 0, 0, 0, 0}

        /*
            The true values are 0xffffffff, which in signed 32-bit integer representation is equal to -1.
        */
<br>

* **Min/Max operations**: We can use the *v_min()* and *v_max()* functions to return registers containing element-wise min, or max, of the two registers:

        v_int32 a;                               // {a1, ..., an}
        v_int32 b;                               // {b1, ..., bn}

        v_int32 mn = v_min(a, b);                // {min(a1, b1), ..., min(an, bn)}
        v_int32 mx = v_max(a, b);                // {max(a1, b1), ..., max(an, bn)}
<br>

@note Comparison and Min/Max operators are not available for 64 bit integers. Bitwise shift and logic operators are available only for integer values. Bitwise shift is available only for 16, 32 and 64 bit registers.

### Reduce and Mask

* **Reduce Operations**: The *v_reduce_min()*, *v_reduce_max()* and *v_reduce_sum()* return a single value denoting the min, max or sum of the entire register:

        v_int32 a;                                //  a = {a1, ..., a4}
        int mn = v_reduce_min(a);                 // mn = min(a1, ..., an)
        int sum = v_reduce_sum(a);                // sum = a1 + ... + an
<br>

* **Mask Operations**: Mask operations allow us to replicate conditionals in wide registers. These include:
    * *v_check_all()* - Returns a bool, which is true if all the values in the register are less than zero.
    * *v_check_any()* - Returns a bool, which is true if any value in the register is less than zero.
    * *v_select()* - Returns a register, which blends two registers, based on a mask.

            v_uint8 a;                           // {a1, .., an}
            v_uint8 b;                           // {b1, ..., bn}

            v_int32x4 mask:                      // {0xff, 0, 0, 0xff, ..., 0xff, 0}

            v_uint8 Res = v_select(mask, a, b)   // {a1, b2, b3, a4, ..., an-1, bn}

            /*
                "Res" will contain the value from "a" if mask is true (all bits set to 1),
                and value from "b" if mask is false (all bits set to 0)

                We can use comparison operators to generate mask and v_select to obtain results based on conditionals.
                It is common to set all values of b to 0. Thus, v_select will give values of "a" or 0 based on the mask.
            */

FP16 and BF16 Arithmetic {#fp16-and-bf16-arithmetic}
------------------------

OpenCV 5.0 introduces universal intrinsic support for 16-bit floating-point types: FP16 (`cv::hfloat`, `CV_16F`) and BF16 (`cv::bfloat`, `CV_16BF`). These are guarded by two preprocessor flags that mirror the existing `CV_SIMD_64F` / `CV_SIMD_SCALABLE_64F` guards for double-precision support.

| Flag | Description |
|---|---|
| `CV_SIMD_16F` | Fixed-width SIMD targets where native FP16 arithmetic is available (e.g. ARMv8.2+ NEON, RISC-V RVV). |
| `CV_SIMD_SCALABLE_16F` | VLA targets (SVE, RVV) where the register width is determined at runtime. |

Code written against these guards is forward-compatible. As x86 and other architectures add native FP16/BF16 arithmetic instructions in future hardware generations, the same source will automatically benefit without modification.

### Basic Usage

To operate on 16-bit floating-point types, utilize the `vx_load_f16`, `v_add`, and `vx_store_f16` APIs.

@code{.cpp}
#if CV_SIMD_16F || CV_SIMD_SCALABLE_16F
// Load FP16 values from memory into a v_float16 register
v_float16 a = vx_load_f16(src1_ptr);
v_float16 b = vx_load_f16(src2_ptr);

// Element-wise addition and fused multiply-add
v_float16 c = v_add(a, b);
v_float16 d = v_fma(a, b, c);    // a*b + c

// Store result back to FP16 memory
vx_store_f16(dst_ptr, d);
#endif
@endcode

### Expanding and Packing

To perform intermediate computation in higher precision (FP32), the universal intrinsics provide `v_expand` and `v_pack`. Expanding converts a single 16-bit register into two 32-bit registers, and packing reverses the operation.

@code{.cpp}
#if CV_SIMD_16F || CV_SIMD_SCALABLE_16F
v_float16 val_16 = vx_load_f16(src_ptr);

// Widen to FP32 for mixed-precision computation
v_float32 lo, hi;
v_expand(val_16, lo, hi);

// Intermediate computation on FP32 registers
v_float32 res_lo = v_mul(lo, vx_setall_f32(3.14159f));
v_float32 res_hi = v_mul(hi, vx_setall_f32(3.14159f));

// Pack back to FP16 for storage
v_float16 res_16 = v_pack(res_lo, res_hi);
vx_store_f16(dst_ptr, res_16);
#endif
@endcode

### BF16 Load/Store and Mixed Precision

BF16 shares its exponent range with FP32, making it critical for deep learning inference. Conversion between BF16 and FP32 via expand/pack is highly efficient and executes primarily as a bit-shift. To handle BF16 data, use the specific `v_bfloat16` type alongside `vx_load_bf16` and `vx_store_bf16`.

@code{.cpp}
#if CV_SIMD_16BF || CV_SIMD_SCALABLE_16BF
// Load BF16 values from memory
v_bfloat16 b_val = vx_load_bf16(bf16_ptr);

// Widen to FP32 for high-precision deep learning accumulation
v_float32 b_lo, b_hi;
v_expand(b_val, b_lo, b_hi);

// Computation in FP32
v_float32 res_lo = v_add(b_lo, vx_setall_f32(1.0f));
v_float32 res_hi = v_add(b_hi, vx_setall_f32(1.0f));

// Pack back to BF16 for output
v_bfloat16 b_res = v_pack_b(res_lo, res_hi);
vx_store_bf16(dst_bf16_ptr, b_res);
#endif
@endcode

### Integer Dot Products (Quantization)

To support the optimized DNN module in OpenCV 5.0, universal intrinsics now include fast 8-bit integer dot products. These are essential for writing custom layers for quantized neural networks. Architectures like ARMv8.4+ and AVX-VNNI have dedicated hardware instructions to multiply 8-bit integers and accumulate them into 32-bit integers in a single pass.

@code{.cpp}
v_int8 vec_a = vx_load(int8_src1);
v_int8 vec_b = vx_load(int8_src2);
v_int32 acc = vx_setzero_s32();

// Multiply 8-bit integers and accumulate into 32-bit registers
acc = v_dotprod_int8(vec_a, vec_b, acc);
@endcode

### Build Requirements

FP16 SIMD arithmetic and scalable vectors are not available by default. The required targets must be enabled at build time. OpenCV's runtime dispatcher will select the appropriate kernel based on the actual CPU capabilities at launch.

@code{.sh}
# ARM - enable ARMv8.2+ FP16 extensions
cmake -DCPU_BASELINE=NEON -DCPU_DISPATCH=FP16 ..

# RISC-V - enable RVV vector extension
cmake -DRISCV_RVV_SCALABLE=ON ..

# WebAssembly (WASM) - enable 128-bit SIMD for browser execution
cmake -DCMAKE_TOOLCHAIN_FILE=../platforms/js/build_wasm.sh -DENABLE_WASM_SIMD=ON ..
@endcode

@note On platforms where `CV_SIMD_16F` is not defined, FP16 intrinsics are not compiled in. Any code that calls them must be wrapped in the corresponding `#if` guard.

### Dynamic Dispatch & Lane Counting

When compiling for Vector Length Agnostic (VLA) architectures (guarded by `CV_SIMD_SCALABLE_*`), the size of the SIMD register is unknown at compile time. It is determined dynamically by the hardware at runtime.

Because of this, you cannot hardcode loop increments (e.g., `i += 4` or `i += 8`). You must use OpenCV's lane-counting macros to advance memory pointers safely.

@code{.cpp}
// Correct approach for scalable loops
int step = VTraits<v_float32>::vlanes();   // Dynamic step size
for (int i = 0; i <= length - step; i += step) {
    v_float32 v = vx_load(src + i);
    // ...
}
@endcode

Math Function Intrinsics
------------------------

Vectorized implementations of common mathematical functions were added in OpenCV 5.0 (and backported to 4.x). They operate element-wise on `v_float32` and `v_float64` registers. Generic implementations cover all platforms, with hardware-specific fast paths mapped where supported by the architecture.

### Supported Operations

| Function | Operation | Notes |
|---|---|---|
| `v_exp(x)` | \f$e^x\f$ | |
| `v_log(x)` | \f$\ln(x)\f$ | Requires \f$x > 0\f$ |
| `v_erf(x)` | Gauss error function | Utilized in GELU activation kernels |
| `v_sincos(x, s, c)` | Sine and cosine | Simultaneous calculation, more efficient than separate calls |
| `v_pow(x, y)` | \f$x^y\f$ | |
| `v_tanh(x)` | Hyperbolic tangent | Utilized in activation kernels |
| `v_atan2(y, x)` | Arctangent of \f$y/x\f$ | Result in radians |

### Basic Usage

@code{.cpp}
v_float32 x = vx_load(src_ptr);

v_float32 e   = v_exp(x);           // e^x for each lane
v_float32 l   = v_log(x);           // ln(x) for each lane, x > 0
v_float32 err = v_erf(x);           // Gauss error function

v_float32 s, c;
v_sincos(x, s, c);                  // sine and cosine in one pass

// Sigmoid: 1 / (1 + e^{-x})
v_float32 ones  = vx_setall_f32(1.f);
v_float32 neg_x = v_mul(x, vx_setall_f32(-1.f));
v_float32 sig   = v_div(ones, v_add(ones, v_exp(neg_x)));
@endcode

@note Math intrinsics do not require guard macros. Generic scalar-loop fallback implementations ensure code compilation across all target platforms.

Multi-Channel Data Operations
-----------------------------

When vectorizing operations on multi-channel data types (e.g., interleaved 3-channel BGR images), standard load functions such as `vx_load` will capture adjacent channel values into the same register. Element-wise arithmetic operations on such a register will yield incorrect scalar results across the respective channels.

To process multi-channel arrays, use the `v_load_deinterleave` and `v_store_interleave` APIs to distribute channel values into independent registers.

### Basic Usage

@code{.cpp}
// 3-channel deinterleave
v_float32 b, g, r;
v_load_deinterleave(src_ptr, b, g, r);

// Apply independent scalar multipliers per channel
v_float32 y = v_mul(b, vx_setall_f32(0.114f));
y = v_fma(g, vx_setall_f32(0.587f), y);
y = v_fma(r, vx_setall_f32(0.299f), y);

// Interleave registers back to target address
v_store_interleave(dst_ptr, y, y, y);
@endcode

## Demonstration
In the following section, we will vectorize a simple convolution function for single channel and compare the results to a scalar implementation.
@note Not all algorithms are improved by manual vectorization. In fact, in certain cases, the compiler may *autovectorize* the code, thus producing faster results for scalar implementations.

You may learn more about convolution from the previous tutorial. We use the same naive implementation from the previous tutorial and compare it to the vectorized version.

The full tutorial code is [here](https://github.com/opencv/opencv/tree/5.x/samples/cpp/tutorial_code/core/univ_intrin/univ_intrin.cpp).

### Vectorizing Convolution

We will first implement a 1-D convolution and then vectorize it. The 2-D vectorized convolution will perform 1-D convolution across the rows to produce the correct results.

#### 1-D Convolution: Scalar
@snippet univ_intrin.cpp convolution-1D-scalar

1. We first set up variables and make a border on both sides of the src matrix, to take care of edge cases.
    @snippet univ_intrin.cpp convolution-1D-border

2. For the main loop, we select an index *i* and offset it on both sides along with the kernel, using the k variable. We store the value in *value* and add it to the *dst* matrix.
    @snippet univ_intrin.cpp convolution-1D-scalar-main

#### 1-D Convolution: Vector

We will now look at the vectorized version of 1-D convolution.
@snippet univ_intrin.cpp convolution-1D-vector

1. In our case, the kernel is a float. Since the kernel's datatype is the largest, we convert src to float32, forming *src_32*. We also make a border like we did for the naive case.
    @snippet univ_intrin.cpp convolution-1D-convert

2. Now, for each column in the *kernel*, we calculate the scalar product of the value with all *window* vectors of length `step`. We add these values to the already stored values in ans
    @snippet univ_intrin.cpp convolution-1D-main

    * We declare a pointer to the src_32 and kernel and run a loop for each kernel element
        @snippet univ_intrin.cpp convolution-1D-main-h1

    * We load a register with the current kernel element. A window is shifted from *0* to *len - step* and its product with the kernel_wide array is added to the values stored in *ans*. We store the values back into *ans*
        @snippet univ_intrin.cpp convolution-1D-main-h2

    * Since the length might not be divisible by steps, we take care of the remaining values directly. The number of *tail* values will always be less than *step* and will not affect the performance significantly. We store all the values to *ans* which is a float pointer. We can also directly store them in a `Mat` object
        @snippet univ_intrin.cpp convolution-1D-main-h3

    * Here is an iterative example:

            For example:
            kernel: {k1, k2, k3}
            src:           ...|a1|a2|a3|a4|...


            iter1:
            for each idx i in (0, len), 'step' idx at a time
                kernel_wide:          |k1|k1|k1|k1|
                window:               |a0|a1|a2|a3|
                ans:               ...| 0| 0| 0| 0|...
                sum =  ans + window * kernel_wide
                    =  |a0 * k1|a1 * k1|a2 * k1|a3 * k1|

            iter2:
                kernel_wide:          |k2|k2|k2|k2|
                window:               |a1|a2|a3|a4|
                ans:               ...|a0 * k1|a1 * k1|a2 * k1|a3 * k1|...
                sum =  ans + window * kernel_wide
                    =  |a0 * k1 + a1 * k2|a1 * k1 + a2 * k2|a2 * k1 + a3 * k2|a3 * k1 + a4 * k2|

            iter3:
                kernel_wide:          |k3|k3|k3|k3|
                window:               |a2|a3|a4|a5|
                ans:               ...|a0 * k1 + a1 * k2|a1 * k1 + a2 * k2|a2 * k1 + a3 * k2|a3 * k1 + a4 * k2|...
                sum =  sum + window * kernel_wide
                    =  |a0*k1 + a1*k2 + a2*k3|a1*k1 + a2*k2 + a3*k3|a2*k1 + a3*k2 + a4*k3|a3*k1 + a4*k2 + a5*k3|


@note The function parameters also include *row*, *rowk* and *len*. These values are used when using the function as an intermediate step of 2-D convolution

#### 2-D Convolution

Suppose our kernel has *ksize* rows. To compute the values for a particular row, we compute the 1-D convolution of the previous *ksize/2* and the next *ksize/2* rows, with the corresponding kernel row. The final values is simply the sum of the individual 1-D convolutions
@snippet univ_intrin.cpp convolution-2D

1. We first initialize variables and make a border above and below the *src* matrix. The left and right sides are handled by the 1-D convolution function.
    @snippet univ_intrin.cpp convolution-2D-init

2. For each row, we calculate the 1-D convolution of the rows above and below it. we then add the values to the *dst* matrix.
    @snippet univ_intrin.cpp convolution-2D-main

3. We finally convert the *dst* matrix to a *8-bit* `unsigned char` matrix
    @snippet univ_intrin.cpp convolution-2D-conv

Results
-------

In the tutorial, we used a horizontal gradient kernel. We obtain the same output image for both methods.

Improvement in runtime varies and will depend on the SIMD capabilities available in your CPU.
