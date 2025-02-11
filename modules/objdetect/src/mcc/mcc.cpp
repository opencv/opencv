// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

/*
 * MIT License
 *
 * Copyright (c) 2018 Pedro Diamel Marrero Fernández
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "opencv2/mcc.hpp"
#include <opencv2/objdetect/checker_model.hpp>
namespace cv
{
namespace mcc
{

/**
  *
  */
 DetectorParametersMCC::DetectorParametersMCC()
    : adaptiveThreshWinSizeMin(23),
      adaptiveThreshWinSizeMax(153),
      adaptiveThreshWinSizeStep(16),
      adaptiveThreshConstant(7),
      minContoursAreaRate(0.003),
      minContoursArea(100),
      confidenceThreshold(0.5),
      minContourSolidity(0.9),
      findCandidatesApproxPolyDPEpsMultiplier(0.05),
      borderWidth(0),
      B0factor(1.25f),
      maxError(0.1f),
      minContourPointsAllowed(4),
      minContourLengthAllowed(100),
      minInterContourDistance(100),
      minInterCheckerDistance(10000),
      minImageSize(1000),
      minGroupSize(4)

{
}

/**
  * @brief Create a new set of DetectorParametersMCC with default values.
  */
Ptr<DetectorParametersMCC> DetectorParametersMCC::create()
{
  Ptr<DetectorParametersMCC> params = makePtr<DetectorParametersMCC>();
  return params;
}
} // namespace mcc
} // namespace cv
