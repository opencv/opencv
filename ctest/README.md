# OpenCV Dashboard Testing

--------------------------------------------------------------------------

## Table of Contents

- [1. Quick Start](#1-quick-start)
- [2. Configuration](#2-configuration)
    - [2.1. Target system](#21-target-system)
    - [2.2. Testing model](#22-testing-model)
    - [2.3. Configure the testing script](#23-configure-the-testing-script)
- [3. Available options](#3-available-options)
    - [3.1. Mandatory testing options](#31-mandatory-testing-options)
    - [3.2. Repository settings](#32-repository-settings)
    - [3.3. OpenCV specific settings](#33-opencv-specific-settings)
    - [3.4. Optional testing settings](#34-optional-testing-settings)
- [4. Usage from build tree](#4-usage-from-build-tree)

--------------------------------------------------------------------------

## 1. Quick Start

1. Download OpenCV Testing script from [SCRIPT_LINK].

2. Put the OpenCV Testing script to a dashboard directory
(for example, `~/Dashboards/opencv` or `c:/Dashboards/opencv`).

3. Run CTest tool from a command line:

        $ ctest -VV -S ~/Dashboards/opencv/opencv_test.cmake

4. Add the above command to a scheduler (for example, `cron`) or
   to a CI system (like buildbot or jenkins).

For CTest tool command line options please refer to [CTEST].

For detailed explanation about testing process please refer to [CTEST_EXT_DOCS].

--------------------------------------------------------------------------

## 2. Configuration

The OpenCV testing script uses **Target system** and **Testing model** notations to
perform different tests.

### 2.1. Target system

The **Target system** describes the target OS, version, architecture, etc.
This parameter allows the testing script to choose appropriate configuration for CMake and build tools.

The generic format for the **Target system** option is `<KIND>[-<NAME>][-<ARCH>]`, where

* `<KIND>` is one of **Linux**, **Windows**, **MacOS**, **Android**.
* `<NAME>` is an optional OS name and version, for example **Ubuntu-14.04**, **Vista**.
* `<ARCH>` is an optional architecture description, for example **x86_64**, **ARM**, **ARM-Tegra5**.

### 2.2. Testing model

The **Testing model** notation describes the intention of the testing and
allows the testing script to choose appropriate set of tests.

Currently the following models are supported:

* *Experimental* - performs custom testing.
* *Nightly* - performs full and clean nightly testing.
* *Continuous* - performs quick testing, only if there were updates in the remote repository.
* *Release* - builds release packages.
* *Performance* - collects benchmarking results.
* *MemCheck* - performs dynamic analysis.
* *Documentation* - builds documentation.

### 2.3. Configure the testing script

The testing script can be configured to perform special testing.
It has several variables, like `CTEST_MODEL`, which can be overwritten.

There are three ways to overwrite default values for all options:

1. Create another script (for example, `my_opencv_test.cmake`) with code of the following form:

        set(CTEST_TARGET_SYSTEM "Linux-Ubuntu-14.04-x64")
        set(CTEST_MODEL         "Performance")
        include("${CTEST_SCRIPT_DIRECTORY}/opencv_test.cmake")

   and use it for the CTest command:

        $ ctest -VV -S ~/Dashboards/opencv/my_opencv_test.cmake

2. Pass the options with CTest command line:

        $ ctest -VV -S ~/Dashboards/opencv/opencv_test.cmake \
            -DCTEST_TARGET_SYSTEM="Linux-Ubuntu-14.04-x64" \
            -DCTEST_MODEL="Nightly"

3. Define the options as environment variables before launching CTest command:

        $ export CTEST_TARGET_SYSTEM="Linux-Ubuntu-14.04-x64"
        $ export CTEST_MODEL="Nightly"
        $ ctest -VV -S ~/Dashboards/opencv/opencv_test.cmake

--------------------------------------------------------------------------

## 3. Available options

### 3.1. Mandatory testing options

The following options are mandatory and must be defined.

##### CTEST_TARGET_SYSTEM

This option describes the target platform for the testing.
By default it is equal to `${CMAKE_SYSTEM}-${CMAKE_SYSTEM_PROCESSOR}`.

See [CMAKE_SYSTEM] and [CMAKE_SYSTEM_PROCESSOR].

##### CTEST_MODEL

The testing model (default - *Experimental*).

##### CTEST_SITE

Site name for submission. By default is equal to the host name.

##### CTEST_BUILD_NAME

Build name for submission. By default is equal to `${CTEST_TARGET_SYSTEM}-${CTEST_MODEL}`.

##### CTEST_DASHBOARD_ROOT

Root folder for the testing.

The testing script will use this folder to create temporary files,
so it should have write access and should be unique for different scripts.

By default is equal to `${CTEST_SCRIPT_DIRECTORY}/${CTEST_TARGET_SYSTEM}/${CTEST_MODEL}`.

##### CTEST_SOURCE_DIRECTORY

Directory with OpenCV sources. By default is equal to `${CTEST_DASHBOARD_ROOT}/source`.

If the folder doesn't exist the testing script will clone it from the remote OpenCV repository
(see the next section).

##### CTEST_BINARY_DIRECTORY

Build folder. By default is equal to `${CTEST_DASHBOARD_ROOT}/build`.



### 3.2. Repository settings

The following options are mandatory if the testing script must support clone and update steps.

##### CTEST_WITH_UPDATE

Update source folder to latest state in remote repository. The option is enabled by default.

**Note:** This operation will reset current source folder state and will discard all not committed changes.

##### CTEST_GIT_COMMAND

Path to the `git` command line tool.

##### CTEST_PROJECT_GIT_URL

OpenCV project repository URL.

##### CTEST_PROJECT_GIT_BRANCH

Branch for testing.



### 3.3. OpenCV specific settings

The following options are optional and might be undefined.
In that case the testing script will use default values, depending on **Target system** and **Testing model**.

##### OPENCV_TEST_DATA_PATH

Path to OpenCV test data. If not set the testing script will clone `opencv_extra` repository
and will use test data from there.

##### OPENCV_EXTRA_SOURCE_DIRECTORY

Path to `opencv_extra` local repository. By default is equal to `${CTEST_DASHBOARD_ROOT}/extra`.
If `OPENCV_TEST_DATA_PATH` is not set, the testing script will clone the `opencv_extra` repository to the
`OPENCV_EXTRA_SOURCE_DIRECTORY` folder.

##### OPENCV_EXTRA_GIT_URL

`opencv_extra` repository URL. By default is equal to `git@github.com:Itseez/opencv_extra.git`.

##### OPENCV_EXTRA_GIT_BRANCH

`opencv_extra` branch for testing. By default is equal to `master`.

##### OPENCV_EXTRA_MODULES

List of OpenCV extra modules. For each extra module the following variables must be provided:

* `OPENCV_<module>_SOURCE_DIRECTORY` - path to source directory.
* `OPENCV_<module>_MODULES_DIRECTORY` - path to modules directory (eg. `modules` sub-folder of the source directory).
* `OPENCV_<module>_GIT_URL` - git repo URL.
* `OPENCV_<module>_GIT_BRANCH` - git branch.

By default `contrib` modules is included into the list with the following default values:

* `OPENCV_contrib_SOURCE_DIRECTORY` : `${CTEST_DASHBOARD_ROOT}/contrib`
* `OPENCV_contrib_MODULES_DIRECTORY` : `${CTEST_DASHBOARD_ROOT}/contrib/modules`
* `OPENCV_contrib_GIT_URL` : `git@github.com:Itseez/opencv_contrib.git`
* `OPENCV_contrib_GIT_BRANCH` : `master`

##### OPENCV_EXTRA_MODULES_PATH

Optional list with extra modules, which are located on local file system.
For example:

```CMake
set(OPENCV_EXTRA_MODULES_PATH "/home/user/my_module")
```

##### OPENCV_BUILD_SHARED_LIBS

Build shared libraries instead of static.

##### OPENCV_BUILD_EXAMPLES

Enable/disable samples compilation.

##### OPENCV_FEATURES_ONLY

List of features, which should be enabled. All other features will be disabled.
For example:

```CMake
set(OPENCV_FEATURES_ONLY TBB FFMPEG GTK)
```

##### OPENCV_FEATURES_ENABLE

List of features, which should be enabled. The list will be combined with default settings.
For example:

```CMake
set(OPENCV_FEATURES_ENABLE OPENGL)
```

##### OPENCV_FEATURES_DISABLE

List of features, which should be disabled. The list will be combined with default settings.
For example:

```CMake
set(OPENCV_FEATURES_DISABLE CUDA)
```



### 3.4. Optional testing settings

##### CTEST_UPDATE_CMAKE_CACHE

True, if the testing script should overwrite CMake cache on each launch.

##### CTEST_EMPTY_BINARY_DIRECTORY

True, if the testing script should clean build directory on each launch.

##### CTEST_WITH_TESTS

Enable/disable test launching.

##### CTEST_TEST_TIMEOUT

Timeout in seconds for single test execution.

##### CTEST_WITH_MEMCHECK

Enable/disable memory check analysis.

##### CTEST_WITH_COVERAGE

Enable/disable CTest-based code coverage analysis.

##### CTEST_WITH_SUBMIT

Enable/disable submission to remote server.

##### CTEST_CMAKE_GENERATOR

CMake generator.

##### CTEST_CONFIGURATION_TYPE

CMake configuration type (eg. Release, Debug).

##### CTEST_BUILD_FLAGS

Extra options for build command. For example:

    set(CTEST_BUILD_FLAGS "-j7")

##### CTEST_MEMORYCHECK_COMMAND

Path to memory check tool. Used only if `CTEST_WITH_MEMCHECK` is enabled.

## CTEST_MEMORYCHECK_SUPPRESSIONS_FILE

Path to suppressions file for the memory check tool.
By default the testing script will use internal file for the `valgrind` tool.

--------------------------------------------------------------------------

## 4. Usage from build tree

The same testing script can be used from OpenCV build tree.

To enable it, turn on `ENABLE_CTEST` option in CMake configuration.
It will add new targets, which will call CTest tool with appropriate configuration:

* `$ make Experimental`
* `$ make ExperimentalMemCheck`
* `$ make ExperimentalCoverage`
* `$ make Performance`

--------------------------------------------------------------------------

## 5. Usage with CI systems

The testing script can be used with CI systems, like buildbot, Jenkins, etc.
The CI system might call the same CTest command to perform project configuration, build and testing.

The testing script supports step-by-step mode, to split all steps on CI system. For example:

    $ ctest -VV -S ~/Dashboards/opencv/opencv_test.cmake,Start
    $ ctest -VV -S ~/Dashboards/opencv/opencv_test.cmake,Configure
    $ ctest -VV -S ~/Dashboards/opencv/opencv_test.cmake,Build
    $ ctest -VV -S ~/Dashboards/opencv/opencv_test.cmake,Test
    $ ctest -VV -S ~/Dashboards/opencv/opencv_test.cmake,Coverage
    $ ctest -VV -S ~/Dashboards/opencv/opencv_test.cmake,MemCheck
    $ ctest -VV -S ~/Dashboards/opencv/opencv_test.cmake,Submit
    $ ctest -VV -S ~/Dashboards/opencv/opencv_test.cmake,Extra

--------------------------------------------------------------------------

[SCRIPT_LINK]: <https://raw.githubusercontent.com/jet47/opencv/ctest-dashboard-testing/ctest/opencv_test.cmake>
[CTEST]: <http://www.cmake.org/cmake/help/v3.0/manual/ctest.1.html>
[CTEST_EXT_DOCS]: <http://ctest-ext.readthedocs.org/en/latest/>
