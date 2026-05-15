# Project 4 计划：修复 PCA::project() UMat/OpenCL 崩溃问题

## 一、问题概述

`cv::PCA::project()` 在输出使用 `UMat` 且 OpenCL 启用时崩溃。属于 OpenCV 真实 bug，对应 Level C（Solve Real OpenCV Issue）。

---

## 二、问题复现代码

```cpp
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <iostream>

int main()
{
    if (!cv::ocl::haveOpenCL()) {
        std::cout << "No OpenCL device, cannot reproduce." << std::endl;
        return 0;
    }

    cv::ocl::setUseOpenCL(true);

    // 构造测试数据
    cv::Mat data = (cv::Mat_<float>(4, 3) <<
        1, 2, 3,
        2, 3, 4,
        3, 4, 5,
        4, 5, 6);

    // 训练 PCA
    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW);

    // Mat 输出 — 正常
    cv::Mat result_mat = pca.project(data);
    std::cout << "Mat project OK, size: " << result_mat.size() << std::endl;

    // UMat 输出 — 触发崩溃
    try {
        cv::UMat result_umat;
        pca.project(data, result_umat);
        std::cout << "UMat project OK" << std::endl;
    } catch (const cv::Exception& e) {
        std::cout << "UMat project FAILED: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
```

**预期行为**：修复前，UMat 路径抛出 `CV_CheckTypeEQ` 异常；修复后，两条路径均正常。

---

## 三、问题分析

### 3.1 崩溃调用链

```
PCA::project()                          [pca.cpp:310/312]
  └─ gemm(..., Mat(), 0, result, ...)   ← 传入 Mat() 作为空 C 矩阵
       └─ CV_OCL_RUN(ocl_gemm(...))     [matmul.dispatch.cpp:347]
            └─ ocl_gemm()               [matmul.dispatch.cpp:157]
                 ├─ haveC = matC.kind() != NONE   → true（Mat() 的 kind 是 MAT）
                 └─ CV_CheckTypeEQ(type, matC.type(), "")
                      → CV_32FC1(5) != CV_8UC1(0) → 断言失败，崩溃
```

### 3.2 根因：`Mat()` vs `noArray()` 的 kind 差异

| 参数 | `kind()` 值 | `haveC` 结果 | 后续行为 |
|------|-------------|-------------|---------|
| `Mat()` | `MAT` (0x10000) | `true` | 执行类型检查 → 失败 |
| `noArray()` | `NONE` (0x00000) | `false` | 跳过类型检查 → 正常 |

- `Mat()` 默认构造的空矩阵，`type()` 返回 `CV_8UC1`（flags 默认值的低位）
- 实际数据类型为 `CV_32FC1`（值 5），两者不等，`CV_CheckTypeEQ` 失败
- CPU 路径不受影响，因为 `matmul.dispatch.cpp:352` 中 `beta == 0.0` 时直接跳过 C 矩阵

### 3.3 触发条件

1. OpenCV 编译时启用 OpenCL（`WITH_OPENCL=ON`）
2. 运行时 OpenCL 可用且已启用
3. `PCA::project()` 的 `result` 参数由 `UMat` 支持（OutputArray 背后是 UMat）
4. `gemm()` 检测到 `_matD.isUMat()` 为 true，走 OpenCL 路径

### 3.4 涉及的源码位置

**Bug 所在文件**：`modules/core/src/pca.cpp`

```cpp
// 行 310（mean.rows == 1 分支）
gemm( tmp_data, eigenvectors, 1, Mat(), 0, result, GEMM_2_T );

// 行 312（mean.cols == 1 分支）
gemm( eigenvectors, tmp_data, 1, Mat(), 0, result, 0 );
```

**Bug 触发文件**：`modules/core/src/matmul.dispatch.cpp`

```cpp
// 行 173 — 判断是否有 C 矩阵
bool haveC = matC.kind() != cv::_InputArray::NONE;

// 行 178 — 类型检查（haveC 为 true 时执行）
CV_CheckTypeEQ(type, matC.type(), "");
```

**同类隐患**（目前不触发，但语义错误）：
- `pca.cpp:129` — `PCA::operator()(int maxComponents)` 中的 gemm 调用
- `pca.cpp:265` — `PCA::operator()(double retainedVariance)` 中的 gemm 调用

**不受影响**：`PCA::backProject()` 传入的是真实矩阵 `tmp_mean`，beta=1，无此问题。

---

## 四、解决方案

### 4.1 源码修改

**文件**：`modules/core/src/pca.cpp`，共 4 处 `Mat()` → `noArray()`

| 行号 | 方法 | 修改前 | 修改后 |
|------|------|--------|--------|
| 129 | `operator()(int)` | `gemm(..., Mat(), 0, ...)` | `gemm(..., noArray(), 0, ...)` |
| 265 | `operator()(double)` | `gemm(..., Mat(), 0, ...)` | `gemm(..., noArray(), 0, ...)` |
| 310 | `project()` | `gemm(..., Mat(), 0, result, GEMM_2_T)` | `gemm(..., noArray(), 0, result, GEMM_2_T)` |
| 312 | `project()` | `gemm(..., Mat(), 0, result, 0)` | `gemm(..., noArray(), 0, result, 0)` |

无需新增头文件，`noArray()` 已通过 `precomp.hpp` 引入。

### 4.2 添加回归测试

**文件**：`modules/core/test/test_mat.cpp`（在现有 `TEST(Core_PCA, accuracy)` 之后）

```cpp
TEST(Core_PCA, project_umat)
{
    if (!cv::ocl::useOpenCL())
        throw SkipTestException("OpenCL is not available");

    const int numSamples = 100, numFeatures = 10, maxComp = 5;
    Mat data(numSamples, numFeatures, CV_32FC1);
    RNG rng(12345);
    rng.fill(data, RNG::UNIFORM, Scalar::all(0.0), Scalar::all(1.0));

    PCA pca(data, Mat(), PCA::DATA_AS_ROW, maxComp);

    // CPU 参考结果
    cv::ocl::setUseOpenCL(false);
    Mat ref = pca.project(data);

    // OpenCL + UMat 路径
    cv::ocl::setUseOpenCL(true);
    UMat umat_result;
    EXPECT_NO_THROW(pca.project(data, umat_result));

    Mat ocl_result = umat_result.getMat(ACCESS_READ);
    double err = cvtest::norm(ref, ocl_result, NORM_INF);
    EXPECT_LE(err, 1e-4) << "PCA::project() UMat vs Mat mismatch";

    cv::ocl::setUseOpenCL(true);
}
```

### 4.3 验证步骤

```bash
# 1. 修复前运行复现程序（确认 bug 存在）
cmake -S main -B main/build -DOpenCV_DIR=/home/code/library/opencv/build
cmake --build main/build -j$(nproc)
./main/build/opencv_pca_umat_repro    # 预期：FAILED

# 2. 修改源码后重编译 OpenCV
cd /home/code/library/opencv/build
cmake --build . -j$(nproc)

# 3. 重新编译并运行复现程序（确认修复）
cmake --build main/build -j$(nproc)
./main/build/opencv_pca_umat_repro    # 预期：OK

# 4. 运行官方 PCA 测试
./bin/opencv_test_core --gtest_filter="Core_PCA*"    # 预期：全部 PASSED
```

### 4.4 Git 工作流与 PR

```bash
git checkout -b fix-pca-project-umat-noarray
# 修改源码 + 测试
git add modules/core/src/pca.cpp modules/core/test/test_mat.cpp
git commit -m "fix: replace Mat() with noArray() in PCA gemm() calls for OpenCL compatibility"
git push origin fix-pca-project-umat-noarray
gh pr create --base 4.x --title "Fix PCA::project() crash with UMat output on OpenCL path"
```
