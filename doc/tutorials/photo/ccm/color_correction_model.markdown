Color Correction Model{#tutorial_ccm_color_correction_model}
===========================

Introduction
----

The purpose of color correction is to adjust the color response of input and output devices to a known state. The device being calibrated is sometimes called the calibration source; the color space used as the standard is sometimes called the calibration target. Color calibration has been used in many industries, such as television production, games, photography, engineering, chemistry, medicine, etc. Due to the manufacturing process of the input and output equipment, the channel response has nonlinear distortion. In order to correct the picture output of the equipment, it is nessary to calibrate the captured color and the actual color.

In this tutorial you will learn how to use the 'Color Correction Model' to do a color correction in a image.

The color correction functionalities are included in:
```cpp
#include <opencv2/photo/ccm.hpp>
```

Reference
----

See details of ColorCorrection Algorithm at https://github.com/riskiest/color_calibration/tree/v4/doc/pdf/English/Algorithm

Source Code of the sample
-----------

The sample has two parts of code, the first is the color checker detector model, see details at tutorial_macbeth_chart_detection, the second part is to make color calibration.

```
Here are the parameters for ColorCorrectionModel
    src :
            detected colors of ColorChecker patches;
            NOTICE: the color type is RGB not BGR, and the color values are in [0, 1];
    constcolor :
            the Built-in color card;
            Supported list:
                Macbeth: Macbeth ColorChecker ;
                Vinyl: DKK ColorChecker ;
                DigitalSG: DigitalSG ColorChecker with 140 squares;
    Mat colors :
           the reference color values
           and corresponding color space
           NOTICE: the color values are in [0, 1]
    refColorSpace :
           the corresponding color space
                  If the color type is some RGB, the format is RGB not BGR;
    Supported Color Space:
            Must be one of the members of the ColorSpace enum.
            @snippet modules/photo/include/opencv2/photo/ccm.hpp ColorSpace
            For the full, up-to-date list see cv::ccm::ColorSpace in ccm.hpp.
```


## Code

@snippet samples/cpp/color_correction_model.cpp tutorial
