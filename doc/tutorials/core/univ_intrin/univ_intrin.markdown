Vectorizing your code using Universal Intrinsics {#tutorial_univ_intrin}
==================================================================

@tableofcontents

@prev_tutorial{tutorial_how_to_use_OpenCV_parallel_for_new}

|    |    |
| -: | :- |
| Compatibility | OpenCV >= 3.0 |

Goal
----

The goal of this tutorial is to provide a guide to using the Universal Intrinsics module to vectorize your C++ code for a faster runtime.
We'll briefly look into _SIMD intrinsics_ and how to work with wide _registers_, followed by a tutorial on the basic operations using wide registers.
The tutorial will only demonstrate basic operations. To know more about Universal Intrinsics, visit the [documentation](https://docs.opencv.org/4.5.3/df/d91/group__core__hal__intrin.html).

Theory
------

In this section, we will briefly look into a few concepts to better help understand the functionality.

### Intrinsics
Intrinsics are functions which are separately handled by the compiler. These functions are often optimized to perform in the most efficient ways possible and hence run faster than normal implementations. However, since these functions depend on the compiler, it makes it difficult to write universal applications.

### SIMD
SIMD stands for **S**ingle **I**nstruction, **M**ultiple **D**ata. SIMD Intrinsics allow the processor to vectorize calculations. The data is stored in what are known as *registers*. A *register* may be *128-bits*, *256-bits* or *512-bits* wide. Each *register* stores **multiple values** of the **same data type**. The size of the register and the size of each value determines the number of values stored in total.

Depending on what *Instruction Sets* your CPU supports, you may be able to use the different registers. To learn more, look [here](https://en.wikipedia.org/wiki/Instruction_set_architecture)

Universal Intrinsics
--------------------

OpenCVs universal intrinsics provides an abstraction to SIMD vectorization methods and allows the user to use intrinsics without the need to write system specific code.

OpenCV Universal Intrinsics support the following instruction sets:
* *128 bit* registers of various types support is implemented for a wide range of architectures including
    * x86(SSE/SSE2/SSE4.2),
    * ARM(NEON),
    * PowerPC(VSX),
    * MIPS(MSA).
* *256 bit* registers are supported on x86(AVX2) and
* *512 bit* registers are supported on x86(AVX512)

@note To know more about what Instruction Sets are supported by your CPU, look [here](). You may also run [this] sample code to check availability. Remember to enable the corresponding tags when compiling.


We will now introduce the available structures and functions:
* Register structures
* Load and store
* Mathematical Operations
* Reduce and Mask

@note For detailed information about every available methods, look [here](https://docs.opencv.org/master/df/d91/group__core__hal__intrin.html#ga6093cc09443c787193a24ffe4b0b4dcd).

#### Register Structures

The Universal Intrinsics set implements every register as a structure based on the particular SIMD register.
All types contain the **nlanes** enumeration which gives the exact number of values that the type can hold. This eliminates the need to hardcode the number of values during implementations.

@note Each register structure is under the *cv* namespace.

There are **two types** of registers:

* **Variable sized registers**: These structures do not have a fixed size and their exact bit length is deduced during compilation, based on the available SIMD capabilities. Consequently, the value of the **nlanes** enum is determined in compile time.
    Each structure follows the following convention

    ```
    v_[type of value][size of each value in bits]
    ```

    For instance, **v_uint8 holds 8-bit unsigned integers** and **v_float32 holds 32-bit floating point values**. We then declare a register like we would declare any object in C++

    <br>

    Based on the available SIMD architecture, a particular register will hold different number of values.
    For example: If your computer supports a maximum of 256bit registers,
    * @ref cv::v_uint8 will hold 32 (*= 256 / 8*) 8-bit unsigned integers
    * @ref cv::v_float64 will hold 4 (*= 256 / 64*) 64-bit floats (doubles)

    <br>

    Available data type and sizes:
    |Type|Size in bits|
    |-:|:-|
    |uint| 8, 16, 32, 64|
    |int | 8, 16, 32, 64|
    |float | 32, 64|
<br>

* **Constant sized registers**: These structures have a fixed bit size and hold a constant number of values. We need to know what SIMD architecture is supported by the system and select compatible registers. Use these only if exact bit length is necessary.
    <br>

    Each structure follows the convention:

    ```
    v_[type of value][size of each value in bits]x[number of values]
    ```
    Suppose we want to store
    * 32-bit(*size in bits*) signed integers in a **128 bit register**. Since the register size is already known, we can find out the *number of data points in register* (*128/32 = 4*): @ref v_int32x8

        ```
        v_int32x8 name_of_register // holds 8 32-bit signed integers.
        ```
    * 64-bit floats in 512 bit register: @ref v_float64x8

        ```
        v_float64x8 name_of_register // holds 8 64 bit floating point integers
        ```

@note Every 256-bit and 512-bit structure is a typedef of the v_reg<_Tp, n> class with the corresponding data type (_Tp) and the number of values (n). For using 256-bit(512-bit) registers, check the CV_SIMD256 (CV_SIMD512) preprocessor definition.

#### Load and Store operations

Now that we know how registers work, let us look at the functions used for filling these registers with values.

* **Load**: Load functions allow you to *load* values into a register.
    * *Constructors* - When declaring a register structure, we can either provide a memory address from where the register will pick up contiguous values, or provide the values explicitly as multiple arguments (Explicit multiple arguments is available only for Constant Sized Registers):

        ```
        float ptr[32] = {1, 2, 3 ..., 32};
        // ptr is a pointer to a contiguous memory block of 32 floats

        // Variable Sized Registers //
        int x = v_float32().nlanes; // set x as the number of values the register can hold

        v_float32 reg1(ptr);      // reg1 stores first x values according to the maximum register size available.
        v_float32 reg2(ptr + x);  // reg stores the next x values



        // Constant Sized Registers //
        v_float32x4 reg1(ptr);        // reg1 stores the first 4 floats (1, 2, 3, 4)
        v_float32x4 reg2(ptr + 4);    // reg2 stores the next 4 floats (5, 6, 7, 8)

        // Or we can explicitly write down the values.
        v_float32x4(1, 2, 3, 4);
        ```

    * *Load Function* - We can use the load method and provide the memory address of the data:

        ```
        float ptr[32] = {1, 2, 3, ..., 32};
        v_float32 reg_var;
        reg_var = vx_load(ptr);    // loads values from ptr[0] upto ptr[nlanes - 1]

        v_float32x4 reg_128;
        reg_128 = v_load(ptr);     // loads values from ptr[0] upto ptr[3]

        v_float32x8 reg_256;
        reg_256 = v256_load(ptr);  // loads values from ptr[0] upto ptr[7]

        v_float32x16 reg_512;
        reg_512 = v512_load(ptr);  // loads values from ptr[0] upto ptr[15]
        ```


        @note The load function assumes data (a[] in this case) is unaligned. If your data is aligned, you may use the @ref vx_load_aligned function.


* **Store**: Store functions allow you to *store* the values from a register into a particular memory location.
    * To store values from a register into a memory location, you may use the @ref v_store() function:

        ```
        float ptr[4];
        v_store(ptr, reg); // store the first 128 bits(interpreted as 4x32-bit floats) of reg into ptr.
        ```

        @note reg containing values of a type may lead to wrong interpretation during storage. For example: if **ptr** is a pointer to a 32-bit float and reg is a register containing 32-bit integers,  the values in ptr may not be as intended. You may cast the registers into the proper type before carrying out operations.

#### Binary and Unary Operators

The universal intrinsics set provides element wise binary and unary operations.

* **Arithmetics**: We can add, subtract, multiply and divide two registers element-wise. The registers must be of the same width and hold the same type. To multiply two registers, for example:

    ```
    v_float32x4 a, b;   // {a1, a2, a3, a4}, {b1, b2, b3, b4}
    v_float32x4 c;
    c = a + b           // {a1 + b1, a2 + b2, a3 + b3, a4 + b4}
    c = a*b;            // {a1*b1, a2*b2, a3*b3, a4*b4}
    ```

* **Bitwise Logic and Shifts**: We can left shift or right shift the bits of each element of the register. We can also apply bitwise &, |, ^ and ~ operators between two registers element-wise:

    ```
    v_int32x4 as;                // {a1, a2, a3, a4}
    v_int32x4 al = as << 2;      // {a1 << 2, ..., a4 << 2}
    v_int32x4 bl = as >> 2;      // {a1 >> 2, ..., a4 >> 2}

    v_int32x4 a, b;
    v_int32x4 a_and_b = a & b;   // {a1 & b1, a2 & b2, a3 & b3, a4 & b4}
    ```

    @note: Bitwise shift and logic operators are only available for integer values. Bitwise shift is available only for 16, 32 and 64 bit integers.

* **Comparison Operators**: We can compare values between two registers using the <, >, <= , >=, == and != operators. Since each register contains multiple values, we don't get a single bool for these operations. Instead, for true values, all bits are converted to one (0xff for 8 bits, 0xffff for 16 bits, etc), while false values return bits converted to zero.

    ```
    v_uint8 a; // a = {0, 1, 2, ..., 15} (in 128-bit registers)
    v_uint8 b; // b = {15, 14, 13, ..., 0}

    v_uint8 c = a < b;

    /*
        let us look at the first 4 values in binary

        a = |00000000|00000001|00000010|00000011|
        b = |00001111|00001110|00001101|00001100|
        c = |11111111|11111111|11111111|11111111|

        If we store the values of c and print them as integers, we will get 255 for true values and 0 for false values.
    */
    ---
    v_int32 a; // a = {1, 2, 3, 4, 5, 6, 7, 8} (in a 256bit supported computer)
    v_int32 b; // b = {8, 7, 6, 5, 4, 3, 2, 1}

    v_int32 c = (a < b); // c = {-1, -1, -1, -1, 0, 0, 0, 0}

    /*
        The true values are 0xffffffff, which in signed 32-bit integer representation is equal to -1.
    */
    ```

    @note: Comparison operators are not available for 64bit integers

* **Min/Max operations**: We can use the v_min and v_max functions to return registers containing element-wise min, or max, of the two registers:

    ```
    v_int32x4 a;                    // {a1, a2, a3, a4}
    v_int32x4 b;                    // {b1, b2, b3, b4}

    v_int32x4 mn = v_min(a, b);     // {min(a1, b1), ..., min(a4, b4)}
    v_int32x4 mx = v_max(a, b);     // {max(a1, b1), ..., max(a4, b4)}
    ```

    @note: Min and max functions are not available for 64bit integers

#### Reduce and Mask

* **Reduce Operations**: The v_reduce_min, v_reduce_max and v_reduce_sum return a single value denoting the min, max or sum of the entire register:

    ```
    v_int32x4 a;                    //  a = {a1, a2, a3, a4}
    int mn = v_reduce_min(a);       // mn = min(a1, a2, a3, a4)
    int sum = v_reduce_sum(a);      // sum = a1 + a2 + a3 + a4
    ```

* **Mask Operations**: Mask operations allow us to replicate conditionals in wide registers. These include:
    * *v_check_all* - Returns a bool, which is true if all the values in the register are less than zero.
    * *v_check_any* - Returns a bool, which is true if any value in the register is less than zero.
    * *v_select* - Returns a register, which blends two registers, based on a mask.
        ```
        v_uint8 a;                           // {a1, .., a16}
        v_uint8 b;                           // {b1, ..., b16}

        v_int32x4 mask:                      // {0xff, 0, 0, 0xff, ..., 0}

        v_uint8 Res = v_select(mask, a, b)   // {a1, b2, b3, a4, ..., b16}

        /*
            "Res" will contain the value from "a" if mask is true (all bits set to 1),
            and value from "b" if mask is false (all bits set to 0)

            We can use comparison operators to generate mask and v_select to obtain results based on conditionals.
            It is common to set all values of b to 0. Thus, v_select will give values of "a" or 0 based on the mask.
        */
        ```

### Compilation Flags