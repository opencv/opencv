#!/usr/bin/env python3
"""
OpenCV ICC Color Management Tutorial (Python)
============================================

This tutorial demonstrates the new ICC v5/iccMAX color management features
in OpenCV Python bindings, including:
- HDR and wide-gamut color workflows
- Color appearance models (CAM02/CAM16)
- Professional cinema and photography pipelines
- Multi-channel and spectral color processing
"""

import cv2
import numpy as np

def main():
    print("OpenCV ICC Color Management Tutorial (Python)")
    print("=" * 50)
    print()

    try:
        # 1. Create HDR test image with extended dynamic range
        height, width = 300, 400
        hdr_image = np.zeros((height, width, 3), dtype=np.float32)

        # Generate HDR gradient (values > 1.0)
        for y in range(height):
            for x in range(width):
                hdr_image[y, x, 0] = (x / width) * 2.0    # Blue: 0-2.0
                hdr_image[y, x, 1] = (y / height) * 3.0   # Green: 0-3.0
                hdr_image[y, x, 2] = 1.5                  # Red: constant 1.5

        print(f"1. Created HDR test image ({height}x{width})")
        print(f"   - Floating-point values: 0.0 to 3.0 (extends beyond SDR)")
        print(f"   - Min value: {hdr_image.min():.2f}, Max value: {hdr_image.max():.2f}")
        print()

        # 2. Create ICC profiles for different color spaces
        srgb_profile = cv2.createStandardProfile("sRGB")
        adobe_rgb_profile = cv2.createStandardProfile("Adobe RGB")
        prophoto_profile = cv2.createStandardProfile("ProPhoto RGB")
        rec2020_profile = cv2.createStandardProfile("Rec2020")

        print("2. Created ICC profiles:")
        print(f"   - sRGB: {srgb_profile.getDescription()}")
        print(f"   - Adobe RGB: {adobe_rgb_profile.getDescription()}")
        print(f"   - ProPhoto RGB: {prophoto_profile.getDescription()}")
        print(f"   - Rec2020: {rec2020_profile.getDescription()}")
        print()

        # 3. Set up viewing conditions
        office_conditions = cv2.getStandardViewingConditions("office")
        cinema_conditions = cv2.getStandardViewingConditions("cinema")
        print_conditions = cv2.getStandardViewingConditions("print")

        print("3. Viewing conditions:")
        print(f"   - Office: {office_conditions.adaptingLuminance} cd/m²")
        print(f"   - Cinema: {cinema_conditions.adaptingLuminance} cd/m²")
        print(f"   - Print: {print_conditions.adaptingLuminance} cd/m²")
        print()

        # 4. Basic color space conversion
        adobe_result = cv2.colorProfileTransform(
            hdr_image, srgb_profile, adobe_rgb_profile,
            intent=cv2.ICC_PERCEPTUAL
        )

        print("4. Basic transformation: sRGB → Adobe RGB")
        print(f"   - Rendering intent: Perceptual")
        print(f"   - Result shape: {adobe_result.shape}")
        print(f"   - Value range: {adobe_result.min():.2f} to {adobe_result.max():.2f}")
        print()

        # 5. Wide-gamut HDR workflow
        prophoto_result = cv2.colorProfileTransform(
            hdr_image, srgb_profile, prophoto_profile,
            intent=cv2.ICC_RELATIVE_COLORIMETRIC
        )

        print("5. Wide-gamut transformation: sRGB → ProPhoto RGB")
        print(f"   - Preserves HDR values and wide color gamut")
        print(f"   - Value range: {prophoto_result.min():.2f} to {prophoto_result.max():.2f}")
        print()

        # 6. Color appearance model workflow
        rec2020_cam16 = cv2.colorProfileTransform(
            hdr_image, srgb_profile, rec2020_profile,
            intent=cv2.ICC_PERCEPTUAL,
            cam=cv2.CAM16,
            viewingConditions=office_conditions
        )

        print("6. HDR workflow with CAM16: sRGB → Rec2020")
        print(f"   - Color appearance model: CAM16")
        print(f"   - Viewing conditions: Office lighting")
        print(f"   - Perceptual adaptation applied")
        print()

        # 7. Compare different rendering intents
        intents = {
            'Perceptual': cv2.ICC_PERCEPTUAL,
            'Relative Colorimetric': cv2.ICC_RELATIVE_COLORIMETRIC,
            'Saturation': cv2.ICC_SATURATION,
            'Absolute Colorimetric': cv2.ICC_ABSOLUTE_COLORIMETRIC
        }

        print("7. Rendering intent comparison:")
        results = {}
        for name, intent in intents.items():
            result = cv2.colorProfileTransform(
                hdr_image, srgb_profile, adobe_rgb_profile, intent=intent
            )
            results[name] = result
            print(f"   - {name}: Range {result.min():.2f} to {result.max():.2f}")
        print()

        # 8. Single color transformation
        hdr_color = np.array([[[0.8, 1.2, 2.1]]], dtype=np.float32)
        transformed_color = cv2.colorProfileTransformSingle(
            hdr_color, srgb_profile, prophoto_profile,
            intent=cv2.ICC_PERCEPTUAL,
            cam=cv2.CAM02,
            viewingConditions=office_conditions
        )

        print("8. Single color transformation:")
        print(f"   - Input HDR color: {hdr_color[0,0]}")
        print(f"   - Transformed color: {transformed_color[0,0]}")
        print(f"   - Used CIECAM02 appearance model")
        print()

        # 9. Professional workflows
        print("9. Professional Workflow Examples:")
        print()

        # Cinema mastering workflow
        cinema_master = cv2.colorProfileTransform(
            hdr_image, rec2020_profile, srgb_profile,
            intent=cv2.ICC_PERCEPTUAL,
            cam=cv2.CAM16,
            viewingConditions=cinema_conditions
        )
        print("   • Cinema mastering: Rec2020 → sRGB with cinema viewing")

        # Photography workflow
        photo_workflow = cv2.colorProfileTransform(
            hdr_image, adobe_rgb_profile, prophoto_profile,
            intent=cv2.ICC_RELATIVE_COLORIMETRIC
        )
        print("   • Photography: Adobe RGB → ProPhoto RGB (archival)")

        # Print simulation
        print_sim = cv2.colorProfileTransform(
            hdr_image, srgb_profile, adobe_rgb_profile,
            intent=cv2.ICC_PERCEPTUAL,
            viewingConditions=print_conditions
        )
        print("   • Print preview: sRGB → print simulation")
        print()

        # 10. Machine learning dataset preparation
        print("10. Machine Learning Applications:")

        # Create dataset with consistent color management
        dataset_colors = []
        source_spaces = [srgb_profile, adobe_rgb_profile, prophoto_profile]
        target_space = rec2020_profile  # Standardize to wide gamut

        for i, source_profile in enumerate(source_spaces):
            # Simulate different camera/source color spaces
            sample_image = hdr_image * (0.8 + i * 0.1)  # Vary exposure

            standardized = cv2.colorProfileTransform(
                sample_image, source_profile, target_space,
                intent=cv2.ICC_RELATIVE_COLORIMETRIC,
                cam=cv2.CAM16,
                viewingConditions=office_conditions
            )

            dataset_colors.append(standardized)
            print(f"    • Standardized {source_profile.getColorSpace()} → Rec2020")

        print("    • Consistent color representation across training data")
        print("    • Improved model generalization with accurate colors")
        print()

        # 11. Performance optimization tips
        print("11. Performance Optimization:")
        print("    • Use numpy arrays with dtype=np.float32 for HDR content")
        print("    • Batch process multiple images for efficiency")
        print("    • Cache ICC profiles to avoid repeated parsing")
        print("    • Use appropriate rendering intent for your use case")
        print("    • Consider computational cost of color appearance models")
        print()

        # 12. Integration with existing OpenCV workflows
        print("12. Integration Examples:")

        # Traditional cvtColor still works
        srgb_traditional = cv2.cvtColor(hdr_image.astype(np.uint8), cv2.COLOR_BGR2RGB)
        print("    • Traditional cvtColor: Compatible with existing code")

        # ICC-aware processing
        icc_aware = cv2.colorProfileTransform(
            hdr_image, srgb_profile, adobe_rgb_profile
        )
        print("    • ICC-aware processing: Enhanced color accuracy")

        # Hybrid workflow
        gray = cv2.cvtColor(icc_aware, cv2.COLOR_BGR2GRAY)
        print("    • Hybrid: ICC transform → traditional processing")
        print()

        print("Tutorial completed successfully!")
        print()
        print("OpenCV ICC v5/iccMAX enables:")
        print("✓ True HDR and wide-gamut color processing")
        print("✓ Perceptual color appearance modeling")
        print("✓ Professional cinema/photography workflows")
        print("✓ Improved machine learning dataset quality")
        print("✓ Seamless integration with existing OpenCV code")

    except Exception as e:
        print(f"Error: {e}")
        return -1

    return 0

if __name__ == "__main__":
    main()
