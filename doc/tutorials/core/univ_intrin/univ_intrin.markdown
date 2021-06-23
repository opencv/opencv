Universal Intrinsics {#tutorial_univ_intrin}
==================================================================

@tableofcontents

@prev_tutorial{tutorial_how_to_use_OpenCV_parallel_for_new}

|    |    |
| -: | :- |
| Compatibility | OpenCV >= 3.0 |

Goal
----

The goal of this tutorial is to provide a guide to using the Universal Intrinsics module to vectorize code for a faster runtime. 
We'll briefly look into _SIMD intrinsics_ and how to work with wide _registers_, followed by a tutorial on how to use them to vectorize a convolution operation.

Theory
------

We'll briefly look into a few concepts to better help understand the functionality. 

### Intrinsics
Intrinsics are functions which are separately handled by the compiler. As a result, these functions are often optimized to perform in the most efficient ways possible and hence run faster than normal implementations. However, since these functions depend on the compiler, it makes it difficult to write cross-platform applications. 

### SIMD
SIMD stands for **S**ingle **I**nstruction, **M**ultiple **D**ata. Compilers provide SIMD Intrinsics which allow the processor to perform a single operation on a set of data simultaneously. The data is stored in what are known as *registers*. A register may be 128bits, 256bit or 512bits wide. Depending on what *Instruction Sets* your CPU supports, you may be able to use the different registers. To learn more, look [here](https:)

The Universal Intrinsics library provides macros which allow the user to write fallback code to make sure applications work regardless of the platform and the hardware. However, runtimes may differ.


