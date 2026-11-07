// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "test_precomp.hpp"
#include <thread>

namespace opencv_test { namespace {

/** @brief The first non-CPU device that really holds data, or skips the test when there is none. */
static ComputeDevice anyDevice()
{
    const std::vector<ComputeDevice> devices = ComputeDevice::enumerate();
    for (size_t i = 1; i < devices.size(); i++)
        if (devices[i].backend() != "opencl" || ocl::useOpenCL())
            return devices[i];
    throw SkipTestException("no device besides the CPU");
}

/** @brief Stores the calling thread's current device in @p out. */
static void recordCurrentDevice(ComputeDevice* out)
{
    *out = ComputeDevice::current();
}

TEST(Core_ComputeDevice, default_is_cpu)
{
    const ComputeDevice d;
    EXPECT_TRUE(d.isCPU());
    EXPECT_EQ(ComputeDevice::cpu(), d);
    EXPECT_EQ("cpu", d.name());
    EXPECT_EQ("cpu", d.backend());
    EXPECT_EQ(Mat::getDefaultAllocator(), d.allocator());
    EXPECT_TRUE(ComputeDevice::current().isCPU());
}

TEST(Core_ComputeDevice, enumerate_and_find)
{
    const std::vector<ComputeDevice> devices = ComputeDevice::enumerate();
    ASSERT_FALSE(devices.empty());
    EXPECT_TRUE(devices[0].isCPU());
    for (size_t i = 0; i < devices.size(); i++)
    {
        SCOPED_TRACE(devices[i].name());
        EXPECT_EQ(devices[i], ComputeDevice::find(devices[i].name()));
        EXPECT_TRUE(devices[i].allocator() != NULL);
        if (i > 0)
        {
            EXPECT_EQ(devices[i], ComputeDevice::find(devices[i].backend()));
        }
    }
}

TEST(Core_ComputeDevice, find_falls_back_to_cpu)
{
    EXPECT_TRUE(ComputeDevice::find("cpu").isCPU());
    EXPECT_TRUE(ComputeDevice::find("").isCPU());
    EXPECT_TRUE(ComputeDevice::find("no_such_backend").isCPU());
    EXPECT_TRUE(ComputeDevice::find("opencl:99").isCPU());
    EXPECT_TRUE(ComputeDevice::find("opencl:").isCPU());
    EXPECT_TRUE(ComputeDevice::find("cuda:-1").isCPU());
    EXPECT_TRUE(ComputeDevice::find("cuda:x").isCPU());
}

TEST(Core_ComputeDevice, of_host_umat)
{
    EXPECT_TRUE(ComputeDevice::of(UMat()).isCPU());
    UMat host;
    host.allocator = Mat::getDefaultAllocator();
    host.create(4, 4, CV_8UC1);
    EXPECT_TRUE(ComputeDevice::of(host).isCPU());
}

TEST(Core_ComputeDevice, of_device_umat)
{
    const ComputeDevice dev = anyDevice();
    UMat u;
    u.allocator = dev.allocator();
    u.create(4, 4, CV_8UC1);
    EXPECT_EQ(dev, ComputeDevice::of(u));
    EXPECT_EQ(dev, ComputeDevice::of(u(Rect(1, 1, 2, 2))));
}

TEST(Core_ComputeDevice, scopes_nest)
{
    const ComputeDevice dev = anyDevice();
    {
        DeviceScope outer(dev);
        EXPECT_EQ(dev, ComputeDevice::current());
        {
            DeviceScope inner(ComputeDevice::cpu());
            EXPECT_TRUE(ComputeDevice::current().isCPU());
        }
        EXPECT_EQ(dev, ComputeDevice::current());
        {
            DeviceScope byName(dev.name());
            EXPECT_EQ(dev, ComputeDevice::current());
        }
    }
    EXPECT_TRUE(ComputeDevice::current().isCPU());
}

TEST(Core_ComputeDevice, scope_is_per_thread)
{
    const ComputeDevice dev = anyDevice();
    DeviceScope scope(dev);
    ComputeDevice seen = dev;
    std::thread worker(recordCurrentDevice, &seen);
    worker.join();
    EXPECT_TRUE(seen.isCPU());
    EXPECT_EQ(dev, ComputeDevice::current());
}

TEST(Core_ComputeDevice, scope_places_new_umat)
{
    const ComputeDevice dev = anyDevice();
    DeviceScope scope(dev);
    const UMat placed(8, 8, CV_32FC1);
    EXPECT_EQ(dev, ComputeDevice::of(placed));

    UMat own;
    own.allocator = Mat::getDefaultAllocator();
    own.create(8, 8, CV_32FC1);
    EXPECT_TRUE(ComputeDevice::of(own).isCPU());
}

TEST(Core_ComputeDevice, cpu_scope_leaves_umat_placement)
{
    const ComputeDevice before = ComputeDevice::of(UMat(8, 8, CV_8UC1));
    DeviceScope scope(ComputeDevice::cpu());
    EXPECT_EQ(before, ComputeDevice::of(UMat(8, 8, CV_8UC1)));
}

}} // namespace
