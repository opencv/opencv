// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "fuzz_precomp.hpp"

using namespace cv;

////////////////////////////////////////////////////////////////////////////////

static void FuzzDrawLine(int x1, int y1, int x2, int y2, int thickness,
                         int line_type, int shift) {
  InitializeOpenCV();
  try {
    // Load image and prepare canvas
    Mat1b matrix = Mat1b::zeros(10, 10);
    Scalar kBlue = Scalar(0, 0, 255);

    line(matrix, Point(x1, y1), Point(x2, y2), kBlue, thickness, line_type,
         shift);
  } catch (const Exception& e) {
  }
}

FUZZ_TEST(Imgproc, FuzzDrawLine);

////////////////////////////////////////////////////////////////////////////////

static void FuzzDrawRectangle(int x, int y, int width, int height,
                              int thickness, int line_type, int shift) {
  InitializeOpenCV();
  try {
    // Load image and prepare canvas
    Mat1b matrix = Mat1b::zeros(10, 10);
    Scalar kBlue = Scalar(0, 0, 255);

    Rect rect = Rect(x, y, width, height);
    rectangle(matrix, rect, kBlue, thickness, line_type, shift);
  } catch (const Exception& e) {
  }
}

FUZZ_TEST(Imgproc, FuzzDrawRectangle);

////////////////////////////////////////////////////////////////////////////////

static void FuzzDrawCircle(int x, int y, int radius, int thickness,
                           int line_type, int shift) {
  InitializeOpenCV();
  try {
    // Load image and prepare canvas
    Mat1b matrix = Mat1b::zeros(10, 10);
    Scalar kBlue = Scalar(0, 0, 255);

    circle(matrix, Point(x, y), radius, kBlue, thickness, line_type, shift);
  } catch (const Exception& e) {
  }
}

FUZZ_TEST(Imgproc, FuzzDrawCircle);

////////////////////////////////////////////////////////////////////////////////

static void FuzzPutText(int font_face, int line_type, float font_scale,
                 int font_thickness, int pos_x, int pos_y,
                 const std::string& annotation, const Mat& canvas) {
  InitializeOpenCV();
  try {
    // Copy the canvas to be able to write to it.
    Mat canvas_copy = canvas.clone();
    putText(canvas_copy, annotation, Point(pos_x, pos_y), font_face,
            font_scale, Scalar(255, 255, 255), font_thickness, line_type);
  } catch (const Exception& e) {
  }
}

FUZZ_TEST(PutTextTest, FuzzPutText)
    .WithDomains(/*font_face=*/fuzztest::ElementOf(
                     {FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_PLAIN,
                      FONT_HERSHEY_DUPLEX, FONT_HERSHEY_COMPLEX,
                      FONT_HERSHEY_TRIPLEX, FONT_HERSHEY_COMPLEX_SMALL,
                      FONT_HERSHEY_SCRIPT_SIMPLEX,
                      FONT_HERSHEY_SCRIPT_COMPLEX}),
                 /*line_type=*/
                 fuzztest::ElementOf({LINE_8, LINE_4, LINE_AA}),
                 /*font_scale=*/fuzztest::InRange<float>(0.f, 10.f),
                 /*font_thickness=*/fuzztest::InRange<int>(0, 10),
                 /*pos_x=*/fuzztest::InRange<int>(-10, 10),
                 /*pos_y=*/fuzztest::InRange<int>(-10, 10),
                 /*annotation=*/fuzztest::Arbitrary<std::string>(),
                 /*canvas=*/Arbitrary2DMat());
