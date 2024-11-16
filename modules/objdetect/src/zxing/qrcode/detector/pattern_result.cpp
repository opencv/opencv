#include "pattern_result.hpp"

// VC++
using namespace zxing;
using namespace qrcode;

using std::abs;
using zxing::Ref;
using zxing::qrcode::FinderPattern;		
using zxing::qrcode::FinderPatternInfo;
using zxing::ResultPoint;
using zxing::qrcode::PatternResult;

PatternResult::PatternResult(Ref<FinderPatternInfo> info)
{
    finderPatternInfo = info;
    possibleAlignmentPatterns.clear();
    
    topLeftPoints.clear();
    topRightPoints.clear();
    bottomLeftPoints.clear();
}

void PatternResult::setConfirmedAlignmentPattern(int index){
    if (index >= static_cast<int>(possibleAlignmentPatterns.size()))
        return;
    confirmedAlignmentPattern = possibleAlignmentPatterns[index];
}
