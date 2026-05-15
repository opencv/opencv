# 修改记录

## 2026-05-15

- 修改 `main/main.cpp`，将复现程序改为强制启用 OpenCL，并打印当前 OpenCL 设备信息。
- 在 `main/main.cpp` 中添加 `cv::redirectError` 回调，用于捕获 `PCA::project()` 内部 OpenCL `gemm()` 路径中被 fallback 机制吞掉的异常。
- 复现到 `matmul.dispatch.cpp:178` 的类型检查错误：实际数据类型为 `CV_32FC1`，空 `Mat()` 的类型为 `CV_8UC1`，说明 `Mat()` 被 OpenCL 路径误认为真实的 C 矩阵。
- 已使用 `cmake -S main -B main/build && cmake --build main/build -j$(nproc) && main/build/opencv_pca_umat_repro` 验证，程序可以稳定打印该 OpenCL 错误并说明之后 fallback 到 CPU。

