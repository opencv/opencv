#ifndef __CORNER_FINDER_H__
#define __CORNER_FINDER_H__

/*
 *  CornerPoint.hpp
 *  zxing
 *
 *  Created by Luiz Silva on 09/02/2010.
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

#include "../../result_point.hpp"
#include <cmath>

namespace zxing {
namespace datamatrix {

class CornerPoint : public ResultPoint {
private:
    int counter_;
    
public:
    CornerPoint(float posX, float posY);
    int getCount() const;
    void incrementCount();
    bool equals(Ref<CornerPoint> other) const;
};
}  // namespace datamatrix
}  // namespace zxing

#endif  // __CORNER_FINDER_H__
