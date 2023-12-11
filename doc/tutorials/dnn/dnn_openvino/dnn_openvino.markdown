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
