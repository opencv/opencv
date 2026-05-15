# 第二阶段计划：在 `gemm()` 层面兼容空 `Mat()` + `beta == 0`

## 一、阶段目标

第一阶段已经完成了 `PCA::project()` 的局部修复：

```cpp
gemm(..., Mat(), 0, result, ...)
```

改为：

```cpp
gemm(..., noArray(), 0, result, ...)
```

这个修复解决了 PCA 的具体问题，但它仍然属于“调用点修复”。第二阶段的目标是进一步思考并尝试在 `cv::gemm()` 底层做一个更通用、更健壮的改进：

> 当 `gemm()` 的第四个参数是空 `Mat()`，且 `beta == 0` 时，将其视为“没有 C 矩阵”，行为等价于 `noArray()`。

这样可以避免 OpenCL 路径把空 `Mat()` 误判为真实 C 矩阵，从而减少类似问题在其他调用点再次出现。

---

## 二、问题背景

`gemm()` 的数学公式为：

```text
D = alpha * A * B + beta * C
```

其中第四个参数 `C` 是可选输入。当 `beta == 0` 时，公式变成：

```text
D = alpha * A * B
```

此时 C 对结果没有任何影响。因此在语义上：

```cpp
gemm(A, B, alpha, Mat(), 0, D, flags);
```

和：

```cpp
gemm(A, B, alpha, noArray(), 0, D, flags);
```

应该等价。

但是目前 OpenCV 的 OpenCL 路径并没有正确处理这个语义。

---

## 三、当前源码行为分析

### 3.1 `gemm()` 入口

文件：

```text
modules/core/src/matmul.dispatch.cpp
```

`gemm()` 大致结构如下：

```cpp
void gemm(InputArray matA, InputArray matB, double alpha,
          InputArray matC, double beta, OutputArray _matD, int flags)
{
#ifdef HAVE_OPENCL
    CV_OCL_RUN(_matD.isUMat() && matA.dims() <= 2 && matB.dims() <= 2 && matC.dims() <= 2,
               ocl_gemm(matA, matB, alpha, matC, beta, _matD, flags))
#endif

    Mat A = matA.getMat(), B = matB.getMat(), C = beta != 0.0 ? matC.getMat() : Mat();
    ...
}
```

这里有两个关键点：

1. 如果输出 `_matD` 是 `UMat`，OpenCV 会优先尝试 OpenCL 路径。
2. CPU 路径已经通过 `beta != 0.0 ? matC.getMat() : Mat()` 表达了：当 `beta == 0` 时，不读取也不使用 `matC`。

也就是说，CPU 路径本身已经把 `Mat(), beta=0` 当成“没有 C 矩阵”处理。

### 3.2 OpenCL 路径 `ocl_gemm()`

同一文件中的 `ocl_gemm()` 里，当前判断 C 是否存在的逻辑是：

```cpp
bool haveC = matC.kind() != cv::_InputArray::NONE;
```

问题在于：

```cpp
Mat()
```

虽然是空矩阵，但它仍然是一个 `Mat` 对象。被 `_InputArray` 包装后：

```text
kind() == MAT
empty() == true
type() == CV_8UC1
size() == [0 x 0]
```

因此 `ocl_gemm()` 会得到：

```cpp
haveC = true;
```

然后继续执行：

```cpp
if (haveC)
    CV_CheckTypeEQ(type, matC.type(), "");
```

如果 A/B 的类型是 `CV_32FC1`，而空 `Mat()` 的 `type()` 是 `CV_8UC1`，就会触发错误：

```text
expected: type == matC.type()
type is CV_32FC1
matC.type() is CV_8UC1
```

在默认构建中，这个异常会被 `CV_OCL_RUN` 捕获，然后 fallback 到 CPU，因此用户表面上可能看不到崩溃。但这说明 OpenCL 路径并没有真正成功。

---

## 四、第二阶段设计原则

这个修改应该遵循以下原则：

### 4.1 只处理 `beta == 0` 的情况

只有当 `beta == 0` 时，C 矩阵才可以被忽略。

允许：

```cpp
gemm(A, B, alpha, Mat(), 0, D, flags);
```

仍然应该报错：

```cpp
gemm(A, B, alpha, Mat(), 1, D, flags);
```

因为 `beta != 0` 时，C 矩阵是公式的一部分，不能缺失。

### 4.2 不改变非空 C 矩阵的行为

如果调用者传入了一个非空矩阵 C，即使 `beta == 0`，也先不改变其行为，以降低兼容性风险：

```cpp
gemm(A, B, alpha, C, 0, D, flags);
```

第二阶段只针对“空 C + beta=0”的特殊兼容场景。

### 4.3 不鼓励继续使用 `Mat()` 表示可选参数

从 API 使用习惯上说，调用方仍应优先写：

```cpp
noArray()
```

而不是：

```cpp
Mat()
```

本阶段优化的意义是增强底层函数健壮性，而不是替代 `noArray()` 的推荐用法。

---

## 五、推荐修改方案

## 5.1 修改文件

主要修改文件：

```text
modules/core/src/matmul.dispatch.cpp
```

需要关注两个 OpenCL 相关函数：

```text
ocl_gemm()
ocl_gemm_amdblas()
```

它们内部都存在类似的 `haveC` 判断。

---

## 5.2 推荐增加一个局部辅助判断

为了避免重复逻辑，可以在 `matmul.dispatch.cpp` 中增加一个小的静态辅助函数。

建议位置：放在 `ocl_gemm_amdblas()` 和 `ocl_gemm()` 之前。

示例：

```cpp
static inline bool gemm_has_addend(InputArray matC, double beta)
{
    if (matC.kind() == cv::_InputArray::NONE)
        return false;

    if (beta == 0.0 && matC.empty())
        return false;

    return true;
}
```

含义：

- `noArray()`：没有 C。
- 空 `Mat()` 且 `beta == 0`：视为没有 C。
- 空 `Mat()` 且 `beta != 0`：视为有 C，后续继续报错。
- 非空 C：视为有 C。

注意：

```cpp
InputArray
```

本质上是 `_InputArray` 的引用类型，因此不建议在 `gemm()` 入口处直接尝试写：

```cpp
matC = noArray();
```

更稳妥的方式是通过 `haveC` 判断来控制后续逻辑。

---

## 5.3 修改 `ocl_gemm()`

当前逻辑：

```cpp
bool haveC = matC.kind() != cv::_InputArray::NONE;
Size sizeA = matA.size(), sizeB = matB.size(), sizeC = haveC ? matC.size() : Size(0, 0);
```

建议改为：

```cpp
bool haveC = gemm_has_addend(matC, beta);
Size sizeA = matA.size(), sizeB = matB.size(), sizeC = haveC ? matC.size() : Size(0, 0);
```

这样，当传入：

```cpp
gemm(A, B, 1.0, Mat(), 0.0, UMatD, flags);
```

时：

```cpp
haveC == false
```

于是会跳过：

```cpp
CV_CheckTypeEQ(type, matC.type(), "");
CV_CheckEQ(sizeC, sizeD, "");
```

OpenCL kernel 也不会按有 C 矩阵的方式编译和执行。

---

## 5.4 修改 `ocl_gemm_amdblas()`

如果源码中启用了 AMD BLAS OpenCL 路径，它也有类似逻辑：

```cpp
bool haveC = matC.kind() != cv::_InputArray::NONE;
```

建议同步改为：

```cpp
bool haveC = gemm_has_addend(matC, beta);
```

原因：

- 两个 OpenCL 路径应保持语义一致。
- 否则普通 OpenCL 路径修好了，但 AMD BLAS 路径仍可能触发同类问题。

---

## 六、为什么不建议只修改 `gemm()` 入口条件

也可以考虑在 `gemm()` 的 `CV_OCL_RUN` 条件中排除空 C：

```cpp
CV_OCL_RUN(_matD.isUMat() && ... && !(beta == 0.0 && matC.empty()), ...)
```

但这个方案不推荐，原因是：

1. 它会阻止 OpenCL 路径运行，而不是修复 OpenCL 路径。
2. 它只影响 `gemm()` 当前入口，如果未来有其他路径调用 `ocl_gemm()`，问题仍存在。
3. 正确目标应该是让 OpenCL 路径理解“空 C + beta=0 等价于无 C”，而不是放弃 OpenCL。

因此，推荐在 `ocl_gemm()` 和 `ocl_gemm_amdblas()` 内部修复 `haveC` 语义。

---

## 七、测试计划

测试文件：

```text
modules/core/test/ocl/test_gemm.cpp
```

现有文件中已经有 OpenCL GEMM 测试，例如：

```cpp
OCL_TEST(Gemm, small)
{
    UMat A(2, 3, CV_32F), B(4, 3, CV_32F), uC(2, 4, CV_32F);
    Mat C(2, 4, CV_32F);

    randu(A, -1, 1);
    randu(B, -1, 1);

    OCL_OFF(cv::gemm(A, B, 1, noArray(), 0, C, GEMM_2_T));
    OCL_ON(cv::gemm(A, B, 1, noArray(), 0, uC, GEMM_2_T));

    EXPECT_LE(cvtest::norm(C, uC, cv::NORM_INF), 1e-5);
}
```

第二阶段可以在这个测试附近新增针对空 `Mat()` 的测试。

---

## 7.1 测试一：空 C + beta=0 应等价于 noArray()

目标：证明下面两种写法结果一致：

```cpp
gemm(uA, uB, 1.0, Mat(), 0.0, uD, 0);
gemm(A, B, 1.0, noArray(), 0.0, D, 0);
```

建议测试代码：

```cpp
OCL_TEST(Gemm, emptyMatCZeroBeta)
{
    Mat A(10, 8, CV_32FC1), B(8, 6, CV_32FC1);
    randu(A, -1, 1);
    randu(B, -1, 1);

    UMat uA, uB, uD;
    A.copyTo(uA);
    B.copyTo(uB);

    Mat ref;
    OCL_OFF(cv::gemm(A, B, 1.0, noArray(), 0.0, ref, 0));

    Mat emptyC;
    OCL_ON(cv::gemm(uA, uB, 1.0, emptyC, 0.0, uD, 0));

    EXPECT_LE(cvtest::norm(ref, uD, cv::NORM_INF), 1e-5);
}
```

但是要注意：默认 `CV_OCL_RUN` 会捕获 OpenCL 异常并 fallback 到 CPU，所以仅比较结果可能无法证明 OpenCL 分支没有报错。因此建议进一步加入错误捕获。

---

## 7.2 测试二：捕获 fallback 前的内部错误

为了让测试能在修复前失败、修复后通过，可以使用 `cv::redirectError()` 记录是否发生了内部 OpenCV 错误。

测试思路：

```cpp
static int gemm_error_count = 0;

static int gemmErrorCallback(int, const char*, const char*, const char*, int, void*)
{
    ++gemm_error_count;
    return 0;
}
```

测试中：

```cpp
gemm_error_count = 0;
cv::ErrorCallback oldCallback = cv::redirectError(gemmErrorCallback);

OCL_ON(cv::gemm(uA, uB, 1.0, Mat(), 0.0, uD, 0));

cv::redirectError(oldCallback);
EXPECT_EQ(0, gemm_error_count);
```

修复前：

- `ocl_gemm()` 内部触发 `CV_CheckTypeEQ`
- error callback 被调用
- `gemm_error_count > 0`
- 测试失败

修复后：

- `haveC == false`
- 不触发类型检查
- `gemm_error_count == 0`
- 测试通过

注意：`redirectError()` 是全局错误回调，测试结束后必须恢复旧回调，避免影响其他测试。

---

## 7.3 测试三：空 C + beta!=0 应继续报错

目标：确保修复没有错误地放宽非法输入。

```cpp
OCL_TEST(Gemm, emptyMatCNonZeroBeta)
{
    UMat uA(10, 8, CV_32F), uB(8, 6, CV_32F), uD;
    randu(uA, -1, 1);
    randu(uB, -1, 1);

    Mat emptyC;
    EXPECT_ANY_THROW(cv::gemm(uA, uB, 1.0, emptyC, 1.0, uD, 0));
}
```

语义：

- `beta != 0` 时，C 矩阵参与计算。
- 空 C 不应被当成 `noArray()`。
- 报错是正确行为。

---

## 7.4 测试四：转置 flags 场景

PCA 的原始问题就使用了：

```cpp
GEMM_2_T
```

因此需要覆盖转置场景：

```cpp
OCL_TEST(Gemm, emptyMatCZeroBetaTranspose)
{
    Mat A(10, 8, CV_32F), B(6, 8, CV_32F);
    randu(A, -1, 1);
    randu(B, -1, 1);

    UMat uA, uB, uD;
    A.copyTo(uA);
    B.copyTo(uB);

    Mat ref;
    OCL_OFF(cv::gemm(A, B, 1.0, noArray(), 0.0, ref, GEMM_2_T));

    Mat emptyC;
    OCL_ON(cv::gemm(uA, uB, 1.0, emptyC, 0.0, uD, GEMM_2_T));

    EXPECT_LE(cvtest::norm(ref, uD, cv::NORM_INF), 1e-5);
}
```

---

## 八、验证流程

### 8.1 修改前验证

在不修改 `matmul.dispatch.cpp` 的情况下运行复现程序，应看到：

```text
OpenCL path reported an error, then OpenCV fell back to CPU.
```

说明当前 OpenCL 路径会报错。

### 8.2 修改后重新编译 core 模块

```bash
cmake --build /home/code/library/opencv/build --target opencv_core -j$(nproc)
```

### 8.3 重新运行复现程序

```bash
cmake -S /home/code/library/opencv/main -B /home/code/library/opencv/main/build
cmake --build /home/code/library/opencv/main/build -j$(nproc)
/home/code/library/opencv/main/build/opencv_pca_umat_repro
```

预期：

```text
OpenCL enabled: YES
OpenCL device: ...
Mat project OK
UMat project returned
No OpenCL error was captured.
```

### 8.4 运行 GEMM 相关测试

```bash
/home/code/library/opencv/build/bin/opencv_test_core --gtest_filter="*Gemm*"
```

### 8.5 运行 PCA 相关测试，确认原问题仍正常

```bash
/home/code/library/opencv/build/bin/opencv_test_core --gtest_filter="Core_PCA*"
```

### 8.6 可选：运行完整 core 测试

```bash
/home/code/library/opencv/build/bin/opencv_test_core
```

---

## 九、预期修改清单

### 9.1 源码修改

```text
modules/core/src/matmul.dispatch.cpp
```

修改内容：

1. 新增 `gemm_has_addend()` 或类似辅助函数。
2. 修改 `ocl_gemm()` 中的 `haveC` 计算。
3. 修改 `ocl_gemm_amdblas()` 中的 `haveC` 计算。

### 9.2 测试修改

```text
modules/core/test/ocl/test_gemm.cpp
```

新增测试：

1. 空 `Mat()` + `beta == 0` 应等价于 `noArray()`。
2. 空 `Mat()` + `beta != 0` 应继续报错。
3. 转置 flags 下空 `Mat()` + `beta == 0` 应正常。
4. 如有必要，用 `redirectError()` 捕获 OpenCL 内部错误，确保不是 fallback 后才通过。

### 9.3 修改记录

按项目要求，在：

```text
MODIFY.md
```

用中文记录修改内容和作用。

---

## 十、PR 策略

建议第二阶段单独提交 PR，不与 PCA 单点修复混在一起。

推荐 PR 标题：

```text
Handle empty C matrix with zero beta in OpenCL gemm
```

PR 描述重点：

1. `gemm()` 的 CPU 路径已经在 `beta == 0` 时忽略 `matC`。
2. OpenCL 路径目前只看 `kind() != NONE`，导致空 `Mat()` 被误判为有效 C。
3. 修改后，空 `Mat()` + `beta == 0` 在 OpenCL 路径中与 `noArray()` 行为一致。
4. `beta != 0` 时仍保持原有错误检查，不放宽非法输入。
5. 新增 OpenCL GEMM 测试覆盖该场景。

---

## 十一、最终推荐实现顺序

1. 阅读并确认 `matmul.dispatch.cpp` 中 `ocl_gemm()` 和 `ocl_gemm_amdblas()` 的 `haveC` 判断。
2. 添加 `gemm_has_addend()` 辅助函数。
3. 将两个 OpenCL GEMM 实现中的 `haveC` 改为调用辅助函数。
4. 添加 `test_gemm.cpp` 中的空 C 回归测试。
5. 编译 `opencv_core` 和 `opencv_test_core`。
6. 运行 `*Gemm*` 测试。
7. 运行现有 PCA 复现程序确认不再捕获 OpenCL 错误。
8. 更新 `MODIFY.md`。
9. 检查 `git diff`，确认修改范围集中在 `matmul.dispatch.cpp`、`test_gemm.cpp` 和 `MODIFY.md`。
