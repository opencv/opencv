# Writing Accuracy Tests

# {#tutorial_accuracy_test_writing}

`This testing framework is based on Google Tests`

There are two major types of C++ tests: *accuracy/regression* tests and *performance* tests.  Each module can have two test binaries: `opencv_test_<MODULE_NAME>` and `opencv_perf_<MODULE_NAME>`, and two tests folders: `<opencv(_contrib)/modules/<MODULE_NAME>/test>` and `<opencv(_contrib)/modules/<MODULE_NAME>/perf>`. These applications can be built for every supported platform (Win/Lin/Mac/Android).

## ACCURACY TESTS

### Work directory

All modules have their own dir for accuracy tests: `opencv_contrib/modules/(moduleName)/test/...`

### Dir Structure

- `test_precomp.hpp` - file for includes
- `test_main.cpp` - main file of test sample
- `test_smth1.cpp` - files for tests
- `test_smth2.cpp`
- `...`

### Test structure

```c++
// name of this case is "name1.name2"
TEST(name1, name2)
{
    ASSERT_....;
}
```

### Tests sample example

**Files:**

- test_precomp.hpp
- test_main.cpp
- test_sum.cpp
- test_sub.cpp



```c++
// test_precomp.hpp

#ifndef __OPENCV_TEST_PRECOMP_HPP__
#define __OPENCV_TEST_PRECOMP_HPP__

#include <...>
#include "..."

namespace opencv_test {
  using namespace cv::(moduleName);
}

#endif
```



```c++
// test_main.cpp
#include "test_precomp.hpp"

CV_TEST_MAIN("<FOLDER_NAME_IN_TESTDATA>")
```



```c++
// test_sum.cpp

#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(SUM, int)
{
    int a=1;
    int b=2;
    int res=a+b;
    ASSERT_EQ(res, 3)
}

TEST(SUM, float)
{
    float a=0.1f;
    float b=0.2f;
    float res=a+b;
    ASSERT_EQ(res, 0.3f)
}

}} // namespace
```




```c++
// test_sub.cpp

#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(SUB, int)
{
    int a=2;
    int b=1;
    int res=a-b;
    ASSERT_EQ(res, 1)
}

TEST(SUB, float)
{
    float a=0.2f;
    float b=0.1f;
    float res=a-b;
    ASSERT_EQ(res, 0.1f)
}

}} // namespace
```


### Run

Before the first run of new tests, you need to rebuild the project with CMake.

If you want to run **ALL TESTS**, you need to run the test sample or use your system:

- (*Windows*) move to the folder `<BUILD_DIR>/bin/Debug(Release)/` and run the executive file `opencv_test_<MODULE_NAME>.exe`
- (*Linux*)

If you want to run **CERTAIN TESTS** (SUM.int test, for example).

- If you work in IDE you need to find the project preference named `Command Arguments` or similar and write there the key like `--gtest_filter=<KEY>` (KEY = SUM.int in our example).  [gtest_filter.doc](--gtest_filter)
- If you want to use your system:
  - (*Windows*) move to the folder `<BUILD_DIR>/bin/Debug(Release)/` and run the executive file in the console `opencv_test_<MODULE_NAME>.exe --gtest_filter=<KEY>` (KEY = SUM.int in our example).
  - (*Linux*)



# TEST DATA

Repository with the test data (opencv_extra): https://github.com/opencv/opencv_extra

Before working with data you need to set environment variable `OPENCV_TEST_DATA_PATH`:

- (Windows)
  1. Win+R
  2. Sysdm.cpl
  3. Go to Advanced -> Environment Variables -> System variables (New..)
  4. Set name as `OPENCV_TEST_DATA_PATH` and value as `<PATH_TO_opencv_extra>/testdata`
- (Visual Studio)
  1.  Go to Project -> Properties -> Debugging -> Environment
  2. Set the Environment as `OPENCV_TEST_DATA_PATH=<PATH_TO_opencv_extra>/testdata`
- (Linux)



# ASSERTION

**Assertion documentation**:

- Link: https://github.com/google/googletest/blob/master/googletest/docs/primer.md



# --gtest_filter

Usage: `--gtest_filter=<KEY>`

This KEY is the logical expression, which help to find needful tests names.

**\*** - logical operator, which includes all symbols before or after this expression.

​	**Examples:** (*smth* - any variation of symbols.)

​	`--gtest_filter=smth*<key_word>`

​	`--gtest_filter=smth*<key_word>*smth`

​	`--gtest_filter=<key_word>*smth`

**:** - logical operator, which concatenate logical expressions.

​	**Examples:** (log_exp - logical expresson.)

​	`--gtest_filter=log_exp1:log_exp`

**Example:**

Tests names:

- TEST_sum.int
- TEST_sum.float
- TEST_sub.int
- TEST_sub.float

Logical keys:

1. \*sum.\*
2. TEST_sum.i\*
3. \*sum.float
4. TEST_sum.i\*:TEST_sub.\*

Results:

1. TEST_sum.int, TEST_sum.float
2. TEST_sum.int
3. TEST_sum.float
4. TEST_sum.int, TEST_sub.int, TEST_sub.float



# --gtest_param_filter

Usage: `--gtest_params_filter=(<PARAM1>, <PARAM1>, ...)`

This sentence helps to pass extra parameters to functions.

For a getting the parameters you need to use `get<i>(GetParam())` or `GET_PARAM(i)`, where **i**

is the index of the parameter.



# BuildBot

OpenCV's continuous integration system is available here:

- [http://pullrequest.opencv.org](http://pullrequest.opencv.org/)
- [http://pullrequest.opencv.org/contrib](https://pullrequest.opencv.org/#/summary/contrib)

This system check code style, build all OpenCV and run tests. You can read more about this bot [here](https://pullrequest.opencv.org/buildbot/).

The BuildBot page looks like a grid where rows are the `pull requests` and columns are `id, author, Title & description, Docs, Builds `.

There are some conditions of samples:

- green (success)

  It means that your code was successful build, all tests passed and you code style is right

- orange (warnings)

  Your code has warnings. You need to click on the number of build and find the orange line in this page. There are two types of buildbot logs: `stdio` and `other like warnings, tests and ets`. `stdio` type show you all build logs. `other` show you only useful logs.

  For example: `warning` logs show only warning logs.

- red (failure)

  - Docs

    Code style check has failed. Click on the number of build and click on `stdio` . There are the error code style logs.

  - Builds

    Building has failed. Click on the number of build and click on `stdio` . There are the all building logs. Errors are marked the red color.

- yellow (building)

  Your project is building now.

- blue (scheduled)

  Building your project is scheduled.