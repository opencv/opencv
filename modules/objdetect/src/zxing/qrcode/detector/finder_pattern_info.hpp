#ifndef __FINDER_PATTERN_INFO_H__
#define __FINDER_PATTERN_INFO_H__

/*
 *  FinderPatternInfo.hpp
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

#include "finder_pattern.hpp"
#include "../../common/counted.hpp"
#include "../../common/array.hpp"
#include <vector>

namespace zxing {
namespace qrcode {

class FinderPatternInfo : public Counted {
private:
    Ref<FinderPattern> bottomLeft_;
    Ref<FinderPattern> topLeft_;
    Ref<FinderPattern> topRight_;
    float possibleFix_;
	float anglePossibleFix_;

public:
    FinderPatternInfo(std::vector<Ref<FinderPattern> > patternCenters);

    Ref<FinderPattern> getBottomLeft();
    Ref<FinderPattern> getTopLeft();
    Ref<FinderPattern> getTopRight();
    void estimateFinderPatternInfo();
    float getPossibleFix();
	float getAnglePossibleFix();

    // to void code duplicated
    static void calculateSides(Ref<FinderPattern> centerA, Ref<FinderPattern> centerB,
                               Ref<FinderPattern> centerC, float &longSide, float &shortSide1,
                               float &shortSide2);
	void showDetail();
};
}  // namespace qrcode
}  // namespace zxing

#endif  // __FINDER_PATTERN_INFO_H__
