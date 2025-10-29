#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/icc.hpp>
#include <iostream>

int main() {
    std::cout << "Testing OpenCV ICC v5/iccMAX Color Management Feature" << std::endl;
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;

    try {
        // Test 1: Create standard profiles
        std::cout << "\n=== Test 1: Creating Standard ICC Profiles ===" << std::endl;
        cv::IccProfile srgbProfile = cv::createStandardProfile("sRGB");
        cv::IccProfile adobeRgbProfile = cv::createStandardProfile("Adobe RGB");

        std::cout << "sRGB Profile - Valid: " << (srgbProfile.isValid() ? "Yes" : "No") << std::endl;
        std::cout << "sRGB Profile - Color Space: " << srgbProfile.getColorSpace() << std::endl;
        std::cout << "sRGB Profile - Input Channels: " << srgbProfile.getInputChannels() << std::endl;
        std::cout << "sRGB Profile - Supports Float: " << (srgbProfile.supportsFloat() ? "Yes" : "No") << std::endl;
        std::cout << "sRGB Profile - Supports HDR: " << (srgbProfile.supportsHDR() ? "Yes" : "No") << std::endl;

        std::cout << "\nAdobe RGB Profile - Valid: " << (adobeRgbProfile.isValid() ? "Yes" : "No") << std::endl;
        std::cout << "Adobe RGB Profile - Color Space: " << adobeRgbProfile.getColorSpace() << std::endl;

        // Test 2: Viewing conditions
        std::cout << "\n=== Test 2: Standard Viewing Conditions ===" << std::endl;
        cv::ViewingConditions officeConditions = cv::getStandardViewingConditions("office");
        cv::ViewingConditions cinemaConditions = cv::getStandardViewingConditions("cinema");

        std::cout << "Office Conditions - Adapting Luminance: " << officeConditions.adaptingLuminance << " cd/m²" << std::endl;
        std::cout << "Office Conditions - Surround: " << officeConditions.surround << std::endl;
        std::cout << "Cinema Conditions - Adapting Luminance: " << cinemaConditions.adaptingLuminance << " cd/m²" << std::endl;
        std::cout << "Cinema Conditions - Surround: " << cinemaConditions.surround << std::endl;

        // Test 3: Color transformation
        std::cout << "\n=== Test 3: ICC Color Transformation ===" << std::endl;

        // Create a test image (3x3 RGB image)
        cv::Mat testImage = (cv::Mat_<cv::Vec3f>(3, 3) <<
            cv::Vec3f(1.0f, 0.0f, 0.0f),  // Red
            cv::Vec3f(0.0f, 1.0f, 0.0f),  // Green
            cv::Vec3f(0.0f, 0.0f, 1.0f),  // Blue
            cv::Vec3f(1.0f, 1.0f, 1.0f),  // White
            cv::Vec3f(0.5f, 0.5f, 0.5f),  // Gray
            cv::Vec3f(0.0f, 0.0f, 0.0f),  // Black
            cv::Vec3f(1.0f, 0.5f, 0.0f),  // Orange
            cv::Vec3f(0.5f, 0.0f, 1.0f),  // Purple
            cv::Vec3f(0.0f, 1.0f, 1.0f)   // Cyan
        );

        cv::Mat transformedImage;

        std::cout << "Input image size: " << testImage.size() << std::endl;
        std::cout << "Input image type: " << testImage.type() << " (CV_32FC3 = " << CV_32FC3 << ")" << std::endl;

        // Test ICC profile transformation
        cv::colorProfileTransform(testImage, transformedImage,
                                 srgbProfile, adobeRgbProfile,
                                 cv::ICC_PERCEPTUAL, cv::CAM_NONE);

        std::cout << "Transformation completed successfully!" << std::endl;
        std::cout << "Output image size: " << transformedImage.size() << std::endl;
        std::cout << "Output image type: " << transformedImage.type() << std::endl;

        // Test 4: Color appearance models
        std::cout << "\n=== Test 4: Color Appearance Models ===" << std::endl;

        cv::Mat camTransformed;
        cv::colorProfileTransform(testImage, camTransformed,
                                 srgbProfile, adobeRgbProfile,
                                 cv::ICC_PERCEPTUAL, cv::CAM16,
                                 officeConditions);

        std::cout << "CAM16 transformation completed successfully!" << std::endl;

        // Test 5: New color conversion codes
        std::cout << "\n=== Test 5: Color Conversion Codes ===" << std::endl;
        std::cout << "COLOR_ICC_PROFILE_TRANSFORM = " << cv::COLOR_ICC_PROFILE_TRANSFORM << std::endl;
        std::cout << "COLOR_ICC_CAM16 = " << cv::COLOR_ICC_CAM16 << std::endl;
        std::cout << "COLOR_COLORCVT_MAX = " << cv::COLOR_COLORCVT_MAX << std::endl;

        std::cout << "\n✅ All ICC Color Management tests passed!" << std::endl;

    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV Exception: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Standard Exception: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
