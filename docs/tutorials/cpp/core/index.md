# The Core Functionality (core module)

## Basic

```{list-table}
:class: opencv-module-table
:widths: 35 65
:header-rows: 1

* - Topic
  - Description

* - [Mat - The Basic Image Container](mat_the_basic_image_container.md)
  - How OpenCV stores and handles images with `cv::Mat` — automatic memory management, headers vs. data, and built-in print formatters.
* - [How to scan images, lookup tables and time measurement with OpenCV](how_to_scan_images.md)
  - Pixel-scanning techniques compared (pointer, iterator, `at`), color reduction with `LUT`, and measuring algorithm performance.
* - [Mask operations on matrices](mat_mask_operations.md)
  - Apply a kernel to every pixel — hand-rolled neighborhood loop versus `filter2D`, illustrated on contrast enhancement.
* - [Operations with images](mat_operations.md)
  - Cheat-sheet of common per-image operations: load/save, pixel access, arithmetic, and type conversions.
* - [Adding (blending) two images using OpenCV](adding_images.md)
  - Linear blend of two images with `addWeighted` to produce cross-dissolve effects.
* - [Changing the contrast and brightness of an image!](basic_linear_transform.md)
  - Pixel-wise linear and gamma transforms with `cv::saturate_cast` for brightness/contrast adjustment.
```

## Advanced

```{list-table}
:class: opencv-module-table
:widths: 35 65
:header-rows: 1

* - Topic
  - Description

* - [Discrete Fourier Transform](discrete_fourier_transform.md)
  - Image DFT with `dft()`, optimal-size padding via `getOptimalDFTSize`, and magnitude-spectrum visualization.
* - [File Input and Output using XML / YAML / JSON files](file_input_output_with_xml_yml.md)
  - Persisting OpenCV types and custom data structures with `cv::FileStorage`, `cv::FileNode`, and `cv::FileNodeIterator`.
* - [How to use the OpenCV parallel_for_ to parallelize your code](how_to_use_OpenCV_parallel_for_new.md)
  - Parallelize pixel-wise work with `parallel_for_` — backends (TBB, OpenMP, GCD, …), race-condition pitfalls, and speed-up measurement.
* - [Vectorizing your code using Universal Intrinsics](univ_intrin.md)
  - SIMD vectorization with OpenCV's universal intrinsics — wide registers, load/store/arithmetic, and VLA-friendly portable code.
```

```{toctree}
:hidden:
:maxdepth: 1

mat_the_basic_image_container
how_to_scan_images
mat_mask_operations
mat_operations
adding_images
basic_linear_transform
discrete_fourier_transform
file_input_output_with_xml_yml
how_to_use_OpenCV_parallel_for_new
univ_intrin
```
