# ICC Color Management in OpenCV {#tutorial_icc_color_management}

@next_tutorial{tutorial_imgproc_threshold}

## Goal

In this tutorial you will learn how to:

- Use ICC v5/iccMAX profiles for professional color management
- Apply color transformations with perceptual adaptation
- Work with HDR and wide-gamut color spaces
- Implement color appearance models (CAM02/CAM16)

## Introduction

OpenCV now supports ICC (International Color Consortium) color profiles for professional color management workflows. This enables accurate color reproduction across different devices and viewing conditions, essential for professional imaging, scientific applications, and machine learning datasets.

## ICC Profile Classes and Functions

### IccProfile Class

The `cv::IccProfile` class encapsulates ICC profile data and provides methods for:
- Loading profiles from files or memory buffers
- Validating profile integrity and version compatibility
- Querying profile characteristics (color space, channels, capabilities)
- Supporting ICC v2, v4, v5, and iccMAX profiles

### Basic Profile Operations

@code{.cpp}
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/icc.hpp>

// Load an ICC profile from file
cv::IccProfile profile("path/to/profile.icc");

// Check if profile is valid
if (profile.isValid()) {
    std::cout << "Profile color space: " << profile.getColorSpace() << std::endl;
    std::cout << "Input channels: " << profile.getInputChannels() << std::endl;
    std::cout << "Supports HDR: " << (profile.supportsHDR() ? "Yes" : "No") << std::endl;
}
@endcode

### Standard Profiles

OpenCV provides built-in standard profiles for common workflows:

@code{.cpp}
// Create standard color profiles
cv::IccProfile srgbProfile = cv::createStandardProfile("sRGB");
cv::IccProfile adobeRgbProfile = cv::createStandardProfile("Adobe RGB");
cv::IccProfile prophotoProfile = cv::createStandardProfile("ProPhoto RGB");
cv::IccProfile rec2020Profile = cv::createStandardProfile("Rec. 2020");
@endcode

## Color Transformations

### Basic Profile Transformation

@code{.cpp}
cv::Mat inputImage = cv::imread("image.jpg", cv::IMREAD_COLOR);
inputImage.convertTo(inputImage, CV_32FC3, 1.0/255.0); // Convert to float

cv::Mat outputImage;
cv::colorProfileTransform(inputImage, outputImage,
                         srgbProfile, adobeRgbProfile,
                         cv::ICC_PERCEPTUAL);
@endcode

### Color Appearance Models

For perceptual accuracy under different viewing conditions:

@code{.cpp}
// Define viewing conditions
cv::ViewingConditions officeConditions = cv::getStandardViewingConditions("office");
cv::ViewingConditions cinemaConditions = cv::getStandardViewingConditions("cinema");

// Transform with CAM16 color appearance model
cv::Mat camTransformed;
cv::colorProfileTransform(inputImage, camTransformed,
                         srgbProfile, adobeRgbProfile,
                         cv::ICC_PERCEPTUAL, cv::CAM16,
                         officeConditions);
@endcode

## HDR and Wide-Gamut Support

### Working with HDR Images

@code{.cpp}
// Load HDR image (supports float values > 1.0)
cv::Mat hdrImage = cv::imread("image.hdr", cv::IMREAD_ANYDEPTH | cv::IMREAD_COLOR);

// Use iccMAX profiles for extended color gamuts
cv::IccProfile hdrProfile = cv::createStandardProfile("Rec. 2020");
cv::IccProfile displayProfile = cv::createStandardProfile("Display P3");

if (hdrProfile.supportsHDR() && hdrProfile.supportsFloat()) {
    cv::Mat displayImage;
    cv::colorProfileTransform(hdrImage, displayImage,
                             hdrProfile, displayProfile,
                             cv::ICC_PERCEPTUAL, cv::CAM16);
}
@endcode

## Professional Workflows

### Scientific Imaging

@code{.cpp}
// Load scientific measurement profile
cv::IccProfile measurementProfile("scientific_spectral.icc");

// Apply precise colorimetric transform
cv::Mat scientificImage, calibratedImage;
cv::colorProfileTransform(scientificImage, calibratedImage,
                         measurementProfile, srgbProfile,
                         cv::ICC_ABSOLUTE_COLORIMETRIC);
@endcode

### Print Production

@code{.cpp}
// Soft-proofing for print
cv::IccProfile printProfile("CMYK_printer.icc");
cv::IccProfile monitorProfile = cv::createStandardProfile("sRGB");

cv::Mat proofImage;
cv::colorProfileTransform(originalImage, proofImage,
                         printProfile, monitorProfile,
                         cv::ICC_PERCEPTUAL);
@endcode

## Integration with OpenCV Color Conversion

The ICC color management is integrated with OpenCV's standard color conversion system:

@code{.cpp}
// Using new color conversion codes
cv::cvtColor(inputImage, outputImage, cv::COLOR_ICC_PROFILE_TRANSFORM);
@endcode

## Best Practices

1. **Profile Validation**: Always check if profiles are valid before use
2. **Floating-Point Processing**: Use CV_32FC3 or CV_64FC3 for precision
3. **Rendering Intent**: Choose appropriate intent for your workflow
4. **Viewing Conditions**: Consider perceptual adaptation for critical applications
5. **HDR Handling**: Use iccMAX profiles for extended dynamic range

## Performance Considerations

- Profile loading is optimized with caching
- Transformations support multi-threading
- GPU acceleration (CUDA/OpenCL) is available for large images
- Memory usage is optimized for real-time applications

## Troubleshooting

### Common Issues

1. **Profile Not Found**: Check file paths and permissions
2. **Invalid Profile**: Verify profile format and version compatibility
3. **Color Shifts**: Ensure correct rendering intent and viewing conditions
4. **Performance**: Use appropriate data types and consider GPU acceleration

### Debugging

@code{.cpp}
// Enable ICC debugging (if available)
cv::setUseOptimized(true);

// Check profile details
if (!profile.isValid()) {
    std::cerr << "Invalid ICC profile" << std::endl;
    return -1;
}

std::cout << "Profile version: " << profile.getVersion() << std::endl;
std::cout << "Profile class: " << profile.getDeviceClass() << std::endl;
@endcode

## Conclusion

OpenCV's ICC color management brings professional-grade color accuracy to computer vision applications. This enables:

- Consistent color reproduction across devices
- Scientific and medical imaging precision
- Professional photography and print workflows
- Enhanced machine learning dataset quality
- HDR and wide-gamut image processing

The implementation supports modern ICC standards (v5/iccMAX) while maintaining backward compatibility and performance suitable for real-time applications.

## See Also

- @ref imgproc_color_icc "ICC Color Management API Reference"
- [ICC Consortium Specifications](http://www.color.org/)
- @ref tutorial_imgproc_basic_linear_transform
- @ref tutorial_imgproc_threshold
