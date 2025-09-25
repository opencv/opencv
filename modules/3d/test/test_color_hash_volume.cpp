#include "test_precomp.hpp"
#include <opencv2/3d/volume.hpp>
#include "../src/rgbd/color_hash_volume.hpp"

namespace opencv_test { namespace {

static size_t measureVolumeMemory(cv::Ptr<cv::ColorHashTSDFVolume> volume, const cv::Mat& depth, const cv::Mat& color,
                          const cv::Matx33f& intr, const cv::Matx44f& pose)
{
    for(int i = 0; i < 5; i++) {
        cv::Matx44f framePose = pose;
        framePose(0, 3) += 0.02f * i; // Move along X axis
        volume->integrate(depth, color, framePose, intr);
    }
}


TEST(ColorHashTSDFVolume, BasicOperation)
{
    cv::VolumeSettings settings;
    settings.setVoxelSize(0.01f);
    settings.setVolumeResolution(cv::Vec3i(64, 64, 64));
    settings.setDepthFactor(1000.0f);
    settings.setTsdfTruncateDistance(0.03f);
    
    cv::Ptr<cv::ColorHashTSDFVolume> volume = cv::ColorHashTSDFVolume::create(settings);
    
    cv::Mat depth(480, 640, CV_32F);
    cv::Mat color(480, 640, CV_8UC3);
    depth = 1.0f;
    color = cv::Vec3b(128, 128, 128);
    
    cv::Matx33f intr(525.0f, 0, 319.5f,
                   0, 525.0f, 239.5f,
                   0, 0, 1);
                 
    cv::Matx44f pose = cv::Matx44f::eye();
    
    volume->integrate(depth, color, pose, intr);
    
    cv::Mat points, normals, colors;
    volume->raycast(pose, intr, points, normals, colors, depth.size());
    
    EXPECT_FALSE(points.empty());
    EXPECT_FALSE(normals.empty());  
    EXPECT_FALSE(colors.empty());
    
    EXPECT_EQ(points.type(), CV_32FC3);
    EXPECT_EQ(normals.type(), CV_32FC3);
    EXPECT_EQ(colors.type(), CV_8UC3);
    EXPECT_EQ(points.size(), depth.size());
}


TEST(ColorHashTSDFVolume, EmptyVolume)
{
    cv::VolumeSettings settings;
    settings.setVoxelSize(0.01f);
    settings.setVolumeResolution(cv::Vec3i(64, 64, 64));
    settings.setDepthFactor(1000.0f);
    settings.setTsdfTruncateDistance(0.03f);
    
    cv::Ptr<cv::ColorHashTSDFVolume> volume = cv::ColorHashTSDFVolume::create(settings);
    
    cv::Matx33f intr = cv::Matx33f::eye();
    cv::Matx44f pose = cv::Matx44f::eye();
    cv::Size size(640, 480);
    
    cv::Mat points, normals, colors;
    volume->raycast(pose, intr, points, normals, colors, size);
    
    // Empty volume should return valid but empty matrices
    EXPECT_FALSE(points.empty());
    EXPECT_FALSE(normals.empty());
    EXPECT_FALSE(colors.empty());
}

TEST(ColorHashTSDFVolume, MemoryUsage) 
{
    cv::VolumeSettings settings;
    settings.setVoxelSize(0.01f);
    settings.setVolumeResolution(cv::Vec3i(64, 64, 64));
    settings.setDepthFactor(1000.0f);
    settings.setTsdfTruncateDistance(0.03f);

    cv::Ptr<cv::ColorHashTSDFVolume> volume = cv::ColorHashTSDFVolume::create(settings);
    
    cv::Mat depth(480, 640, CV_32F, 1.0f);
    cv::Mat color(480, 640, CV_8UC3, cv::Vec3b(128, 128, 128));
    
    cv::Matx33f intr(525.0f, 0, 319.5f,
                   0, 525.0f, 239.5f,
                   0, 0, 1);
    cv::Matx44f pose = cv::Matx44f::eye();

    // Measure initial memory
    size_t memBefore = measureVolumeMemory(volume, depth, color, intr, pose);
    
    // Double resolution and measure again
    settings.setVolumeResolution(cv::Vec3i(128, 128, 128));
    volume = cv::ColorHashTSDFVolume::create(settings);
    size_t memAfter = measureVolumeMemory(volume, depth, color, intr, pose);

    // Memory should increase roughly by factor of 2-8x due to hash table growth
    float memRatio = (float)memAfter / memBefore;
    EXPECT_GT(memRatio, 2.0f);
    EXPECT_LT(memRatio, 10.0f);

    std::cout << "Memory usage: " << memBefore / 1024 / 1024 << "MB -> "
              << memAfter / 1024 / 1024 << "MB (ratio: " << memRatio << "x)\n";
}

}} // namespace

