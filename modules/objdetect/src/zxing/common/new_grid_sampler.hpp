#pragma once

/*
 *  NewGridSampler.hpp
 *  zxing
 *
 *  Copyright 2010 ZXing authors All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http:// www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "counted.hpp"
#include "bit_matrix.hpp"
#include "perspective_transform.hpp"
#include "byte_matrix.hpp"
#include "../error_handler.hpp"

namespace zxing {

namespace wxcode {
    const int CODING_ROW_SUM = 72;
}

class NewGridSampler {
private:
    static NewGridSampler gridSampler;
    NewGridSampler();
    
public:
    Ref<BitMatrix> sampleGrid(Ref<BitMatrix> image, int dimension, Ref<PerspectiveTransform> transform, float fInitialMS, ErrorHandler &err_handler);
    
    Ref<ByteMatrix> sampleGrid(Ref<ByteMatrix> image, int dimension, Ref<PerspectiveTransform> transform, ErrorHandler & err_handler);
    
    static int checkAndNudgePoints(int width, int height, std::vector<float> &points, ErrorHandler & err_handler);
    static NewGridSampler &getInstance();
};
}  // namespace zxing
