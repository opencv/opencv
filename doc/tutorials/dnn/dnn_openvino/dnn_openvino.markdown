OpenCV usage with OpenVINO {#tutorial_dnn_openvino}
=====================

@prev_tutorial{tutorial_dnn_halide_scheduling}
@next_tutorial{tutorial_dnn_yolo}

|    |    |
| -: | :- |
| Original author | Aleksandr Voron |
| Compatibility | OpenCV == 4.x |

This tutorial provides OpenCV installation guidelines how to use OpenCV with OpenVINO.

Since 2021.1.1 release OpenVINO does not provide pre-built OpenCV.
The change does not affect you if you are using OpenVINO runtime directly or OpenVINO samples: it does not have a strong dependency to OpenCV.
However, if you are using Open Model Zoo demos or OpenVINO runtime as OpenCV DNN backend you need to get the OpenCV build.

There are 2 approaches how to get OpenCV:

- Install pre-built OpenCV from another sources: system repositories, pip, conda, homebrew. Generic pre-built OpenCV package may have several limitations:
    - OpenCV version may be out-of-date
    - OpenCV may not contain G-API module with enabled OpenVINO support (e.g. some OMZ demos use G-API functionality)
    - OpenCV may not be optimized for modern hardware (default builds need to cover wide range of hardware)
    - OpenCV may not support Intel TBB, Intel Media SDK
    - OpenCV DNN module may not use OpenVINO as an inference backend
- Build OpenCV from source code against specific version of OpenVINO. This approach solves the limitations mentioned above.

The instruction how to follow both approaches is provided in [OpenCV wiki](https://github.com/opencv/opencv/wiki/BuildOpenCV4OpenVINO).

## Supported targets

OpenVINO backend (DNN_BACKEND_INFERENCE_ENGINE) supports the following [targets](https://docs.opencv.org/4.x/d6/d0f/group__dnn.html#ga709af7692ba29788182cf573531b0ff5):

- **DNN_TARGET_CPU:** Runs on the CPU, no additional dependencies required.
- **DNN_TARGET_OPENCL, DNN_TARGET_OPENCL_FP16:** Runs on the iGPU, requires OpenCL drivers. Install [intel-opencl-icd](https://launchpad.net/ubuntu/jammy/+package/intel-opencl-icd) on Ubuntu.
- **DNN_TARGET_MYRIAD:** Runs on Intel® VPU like the [Neural Compute Stick](https://www.intel.com/content/www/us/en/products/sku/140109/intel-neural-compute-stick-2/specifications.html), to set up [see](https://www.intel.com/content/www/us/en/developer/archive/tools/neural-compute-stick.html).
- **DNN_TARGET_HDDL:** Runs on the Intel® Movidius™ Myriad™ X High Density Deep Learning VPU, for details [see](https://intelsmartedge.github.io/ido-specs/doc/building-blocks/enhanced-platform-awareness/smartedge-open_hddl/).
- **DNN_TARGET_FPGA:** Runs on Intel® Altera® series FPGAs [see](https://www.intel.com/content/www/us/en/docs/programmable/768970/2025-1/getting-started-guide.html).
- **DNN_TARGET_NPU:** Runs on the integrated Intel® AI Boost processor, requires [Linux drivers](https://github.com/intel/linux-npu-driver/releases/tag/v1.17.0) OR [Windows drivers](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html).