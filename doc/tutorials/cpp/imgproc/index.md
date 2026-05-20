# Image Processing (imgproc module)

The imgproc module is a collection of per-pixel image operations (color conversions, filters),
drawing (contours, objects, text), and geometry transformations (warping, resize) useful for
computer vision.

## Basic

```{list-table}
:class: opencv-module-table
:widths: 35 65
:header-rows: 1

* - Topic
  - Description

* - [Basic Drawing](basic_geometric_drawing.md)
  - Draw lines, ellipses, rectangles, circles, and polygons with `cv::line`, `cv::ellipse`, `cv::rectangle`, `cv::circle`, and `cv::fillPoly`.
* - [Random Generator and Text](random_generator_and_text.md)
  - Use `cv::RNG` to generate random colors and coordinates while drawing a mix of geometric shapes and text on an image.
* - [Smoothing Images](gausian_median_blur_bilateral_filter.md)
  - Apply Gaussian, median, and bilateral blur filters with `cv::GaussianBlur`, `cv::medianBlur`, and `cv::bilateralFilter`.
* - [Eroding and Dilating](erosion_dilatation.md)
  - Perform erosion and dilation morphological operations using `cv::erode` and `cv::dilate` with custom structuring elements.
* - [More Morphology Transformations](opening_closing_hats.md)
  - Opening, closing, morphological gradient, top-hat, and black-hat transforms via `cv::morphologyEx`.
* - [Hit-or-Miss](hitOrMiss.md)
  - Detect specific pixel patterns in binary images with the Hit-or-Miss transform using `MORPH_HITMISS`.
* - [Extract horizontal and vertical lines by using morphological operations](morph_lines_detection.md)
  - Use morphologically-shaped structuring elements to isolate horizontal or vertical lines in scanned documents.
* - [Image Pyramids](pyramids.md)
  - Downsample and upsample images with `cv::pyrDown` and `cv::pyrUp` to build Gaussian pyramid representations.
* - [Basic Thresholding Operations](threshold.md)
  - Apply binary, truncate, and zero thresholding with `cv::threshold` and explore the five threshold type variants.
* - [Thresholding Operations using inRange](threshold_inRange.md)
  - Segment objects by color range in HSV colorspace using `cv::inRange` with trackbar-controlled bounds.
```

```{toctree}
:hidden:
:maxdepth: 1

basic_geometric_drawing
random_generator_and_text
gausian_median_blur_bilateral_filter
erosion_dilatation
opening_closing_hats
hitOrMiss
morph_lines_detection
pyramids
threshold
threshold_inRange
```
