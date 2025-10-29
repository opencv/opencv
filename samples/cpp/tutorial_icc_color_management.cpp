/**
 * @file icc_color_management_tutorial.cpp
 * @brief Tutorial demonstrating ICC v5/iccMAX color management in OpenCV
 *
 * This tutorial shows how to use the new ICC color management features:
 * - Loading and using ICC profiles
 * - Converting between color spaces with ICC profiles
 * - Using color appearance models (CAM02/CAM16)
 * - HDR and wide-gamut workflows
 */

#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    std::cout << "OpenCV ICC Color Management Tutorial\n";
    std::cout << "====================================\n\n";

    try {
        // 1. Create test image (HDR content)
        cv::Mat hdrImage(300, 400, CV_32FC3);

        // Generate HDR gradient with values > 1.0
        for (int y = 0; y < hdrImage.rows; y++) {
            for (int x = 0; x < hdrImage.cols; x++) {
                cv::Vec3f& pixel = hdrImage.at<cv::Vec3f>(y, x);
                pixel[0] = (float)x / hdrImage.cols * 2.0f;  // Blue: 0-2.0
                pixel[1] = (float)y / hdrImage.rows * 3.0f;  // Green: 0-3.0
                pixel[2] = 1.5f;                             // Red: constant 1.5
            }
        }

        std::cout << "1. Created HDR test image (" << hdrImage.rows << "x" << hdrImage.cols << ")\n";
        std::cout << "   - Floating-point values: 0.0 to 3.0 (extends beyond SDR range)\n\n";

        // 2. Create ICC profiles for different color spaces
        cv::IccProfile srgbProfile = cv::createStandardProfile("sRGB");
        cv::IccProfile adobeRgbProfile = cv::createStandardProfile("Adobe RGB");
        cv::IccProfile proPhotoProfile = cv::createStandardProfile("ProPhoto RGB");
        cv::IccProfile rec2020Profile = cv::createStandardProfile("Rec2020");

        std::cout << "2. Created ICC profiles:\n";
        std::cout << "   - sRGB: " << srgbProfile.getDescription() << "\n";
        std::cout << "   - Adobe RGB: " << adobeRgbProfile.getDescription() << "\n";
        std::cout << "   - ProPhoto RGB: " << proPhotoProfile.getDescription() << "\n";
        std::cout << "   - Rec2020: " << rec2020Profile.getDescription() << "\n\n";

        // 3. Set up viewing conditions for color appearance modeling
        cv::ViewingConditions officeConditions = cv::getStandardViewingConditions("office");
        cv::ViewingConditions cinemaConditions = cv::getStandardViewingConditions("cinema");

        std::cout << "3. Viewing conditions:\n";
        std::cout << "   - Office: " << officeConditions.adaptingLuminance << " cd/m²\n";
        std::cout << "   - Cinema: " << cinemaConditions.adaptingLuminance << " cd/m²\n\n";

        // 4. Basic color space conversion: sRGB -> Adobe RGB
        cv::Mat adobeRgbResult;
        cv::colorProfileTransform(hdrImage, adobeRgbResult,
                                 srgbProfile, adobeRgbProfile,
                                 cv::ICC_PERCEPTUAL);

        std::cout << "4. Basic transformation: sRGB -> Adobe RGB\n";
        std::cout << "   - Rendering intent: Perceptual\n";
        std::cout << "   - Result size: " << adobeRgbResult.size() << "\n\n";

        // 5. Wide-gamut transformation: sRGB -> ProPhoto RGB
        cv::Mat proPhotoResult;
        cv::colorProfileTransform(hdrImage, proPhotoResult,
                                 srgbProfile, proPhotoProfile,
                                 cv::ICC_RELATIVE_COLORIMETRIC);

        std::cout << "5. Wide-gamut transformation: sRGB -> ProPhoto RGB\n";
        std::cout << "   - Rendering intent: Relative Colorimetric\n";
        std::cout << "   - Preserves HDR values and wide color gamut\n\n";

        // 6. HDR workflow: sRGB -> Rec2020 with CAM16 color appearance model
        cv::Mat rec2020Result;
        cv::colorProfileTransform(hdrImage, rec2020Result,
                                 srgbProfile, rec2020Profile,
                                 cv::ICC_PERCEPTUAL, cv::CAM16,
                                 officeConditions);

        std::cout << "6. HDR workflow: sRGB -> Rec2020 with CAM16\n";
        std::cout << "   - Color appearance model: CAM16\n";
        std::cout << "   - Viewing conditions: Office lighting\n";
        std::cout << "   - Accurate perceptual color adaptation\n\n";

        // 7. Compare different rendering intents
        cv::Mat perceptualResult, colorimetricResult, saturationResult;

        cv::colorProfileTransform(hdrImage, perceptualResult,
                                 srgbProfile, adobeRgbProfile,
                                 cv::ICC_PERCEPTUAL);

        cv::colorProfileTransform(hdrImage, colorimetricResult,
                                 srgbProfile, adobeRgbProfile,
                                 cv::ICC_RELATIVE_COLORIMETRIC);

        cv::colorProfileTransform(hdrImage, saturationResult,
                                 srgbProfile, adobeRgbProfile,
                                 cv::ICC_SATURATION);

        std::cout << "7. Rendering intent comparison:\n";
        std::cout << "   - Perceptual: Optimized for photographic content\n";
        std::cout << "   - Relative Colorimetric: Preserves color accuracy\n";
        std::cout << "   - Saturation: Enhances vivid colors\n\n";

        // 8. Single color transformation (useful for UI/graphics)
        cv::Mat singleColor = (cv::Mat_<float>(1, 3) << 0.8f, 1.2f, 2.1f); // HDR color
        cv::Mat transformedColor;

        cv::colorProfileTransformSingle(singleColor, transformedColor,
                                       srgbProfile, proPhotoProfile,
                                       cv::ICC_PERCEPTUAL, cv::CAM02,
                                       officeConditions);

        std::cout << "8. Single color transformation:\n";
        std::cout << "   - Input HDR color: [" << singleColor.at<float>(0,0)
                  << ", " << singleColor.at<float>(0,1)
                  << ", " << singleColor.at<float>(0,2) << "]\n";
        std::cout << "   - Transformed with CIECAM02 appearance model\n\n";

        // 9. Professional workflow demonstration
        std::cout << "9. Professional Workflow Examples:\n\n";

        // Cinema workflow: Rec2020 -> sRGB with cinema viewing conditions
        cv::Mat cinemaToSrgb;
        cv::colorProfileTransform(hdrImage, cinemaToSrgb,
                                 rec2020Profile, srgbProfile,
                                 cv::ICC_PERCEPTUAL, cv::CAM16,
                                 cinemaConditions);
        std::cout << "   • Cinema mastering: Rec2020 -> sRGB with cinema conditions\n";

        // Photography workflow: Adobe RGB -> ProPhoto RGB
        cv::Mat photoWorkflow;
        cv::colorProfileTransform(hdrImage, photoWorkflow,
                                 adobeRgbProfile, proPhotoProfile,
                                 cv::ICC_RELATIVE_COLORIMETRIC);
        std::cout << "   • Photography: Adobe RGB -> ProPhoto RGB (archival)\n";

        // Print workflow: sRGB -> CMYK simulation (using Adobe RGB as proxy)
        cv::Mat printWorkflow;
        cv::colorProfileTransform(hdrImage, printWorkflow,
                                 srgbProfile, adobeRgbProfile,
                                 cv::ICC_PERCEPTUAL, cv::CAM_NONE,
                                 cv::getStandardViewingConditions("print"));
        std::cout << "   • Print preview: sRGB -> print simulation with print conditions\n\n";

        // 10. Performance and compatibility notes
        std::cout << "10. Performance Notes:\n";
        std::cout << "    • Use floating-point formats (CV_32F, CV_64F) for HDR content\n";
        std::cout << "    • CAM02/CAM16 models add computational overhead but improve accuracy\n";
        std::cout << "    • Relative colorimetric intent preserves out-of-gamut HDR values\n";
        std::cout << "    • Multi-threading supported for large images\n\n";

        std::cout << "Tutorial completed successfully!\n";
        std::cout << "ICC v5/iccMAX color management provides:\n";
        std::cout << "• True HDR and wide-gamut support\n";
        std::cout << "• Perceptual color appearance modeling\n";
        std::cout << "• Professional cinema/photography workflows\n";
        std::cout << "• Backward compatibility with existing code\n";

    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV Error: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
