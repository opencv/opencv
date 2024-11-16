#include "code25reader.hpp"
#include "one_dresult_point.hpp"
#include "../common/array.hpp"
#include "../reader_exception.hpp"
#include "../not_found_exception.hpp"
#include "../checksum_exception.hpp"
#include <math.h>
#include <limits.h>

#include<iostream>

using std::vector;
using zxing::Ref;
using zxing::Result;
using zxing::String;
using zxing::NotFoundException;
using zxing::ChecksumException;
using zxing::oned::Code25Reader;

// VC++
using zxing::BitArray;

#include "one_dconstant.hpp"
using namespace zxing;
using namespace oned;
using namespace zxing::oned::constant::Code25;

const int Code25Reader::MAX_AVG_VARIANCE = static_cast<int>(PATTERN_MATCH_RESULT_SCALE_FACTOR * 250 / 1000);
const int Code25Reader::MAX_INDIVIDUAL_VARIANCE = static_cast<int>(PATTERN_MATCH_RESULT_SCALE_FACTOR * 700 / 1000);

void Code25Reader::init(bool usingCheckDigit_, bool extendedMode_) {
    usingCheckDigit = usingCheckDigit_;
    extendedMode = extendedMode_;  // fake
    decodeRowResult.reserve(20);
    counters.resize(10);
}

/**
 * Creates a reader that assumes all encoded data is data, and does not treat
 * the final character as a check digit. It will not decoded "extended
 * Code 39" sequences.
 */
Code25Reader::Code25Reader() {
    init();
}

/**
 * Creates a reader that can be configured to check the last character as a
 * check digit. It will not decoded "extended Code 39" sequences.
 *
 * @param usingCheckDigit if true, treat the last data character as a check
 * digit, not data, and verify that the checksum passes.
 */
Code25Reader::Code25Reader(bool usingCheckDigit_) {
    init(usingCheckDigit_);
}

Code25Reader::Code25Reader(bool usingCheckDigit_, bool extendedMode_) {
    init(usingCheckDigit_, extendedMode_);
}

Ref<Result> Code25Reader::decodeRow(int rowNumber, Ref<BitArray> row) {
    
    ErrorHandler err_handler;
    std::vector<int>& theCounters(counters);
    std::string& result(decodeRowResult);
    result.clear();
    std::vector<int> startCounters;
    startCounters.resize(4);
    
    vector<int> start(findStartPattern(row, startCounters, _onedReaderData));
    
    if (start.size() == 0)
    {
        return Ref<Result>(NULL);
    }
    
    // Read off white space
    int nextStart = row->getNextSet(start[1]);
    // int end = row->getSize();
    
    char decodedChar;
    int lastStart;
    do {
        bool rp = recordPattern(row, nextStart, theCounters, _onedReaderData);
        
        if (rp == false)
        {
            break;
        }
        
        int whitePattern, blackPattern;
        if (!splitPattern(theCounters, blackPattern, whitePattern))
        {
            break;
        }
        decodedChar = patternToChar(blackPattern, err_handler);
        if (err_handler.ErrCode())   return Ref<Result>(NULL);
        result.append(1, decodedChar);
        
        decodedChar = patternToChar(whitePattern, err_handler);
        if (err_handler.ErrCode())   return Ref<Result>(NULL);
        result.append(1, decodedChar);
        
        lastStart = nextStart;
        for (int i = 0, end = theCounters.size(); i < end; i++) {
            nextStart += theCounters[i];
        }
        // Read off white space
        nextStart = row->getNextSet(nextStart);
    } while (true);
    
    std::vector<int> endCounters;
    endCounters.resize(3);
    
    // check end
    if (recordPattern(row, nextStart, endCounters, _onedReaderData))
    {
        int lastPatternSize = 0;
        int endSize = endCounters.size();
        
        float deltaA = 0.5;
        float deltaB = 1.5;
        if (!(endCounters[0] < (3+deltaA) * endCounters[2] && endCounters[0] > (3-deltaB) * endCounters[2]))
            return Ref<Result>(NULL);
        
        int minCounters = INT_MAX;
        for (int i = 1; i < endSize; ++i)
        {
            minCounters = (std::min)(minCounters, endCounters[i]);
        }
        
        if (abs(endCounters[1] - endCounters[2]) > minCounters)
            return Ref<Result>(NULL);
        
        if (result.length() <= 4)
        {
            // Almost false positive
            return Ref<Result>(NULL);
        }
        
        {
            ErrorHandler err_handler_;
            const int range_start = nextStart + (endCounters[0] + endCounters[1] + endCounters[2]);
            const int range_end = range_start + (endCounters[0] + endCounters[1]/2);
            if (range_end >= row->getSize() || !row->isRange(range_start, range_end, false, err_handler_))
                return Ref<Result>(NULL);
            if (err_handler_.ErrCode())   return Ref<Result>(NULL);
        }
        
        Ref<String> resultString = Ref<String>(new String(result));
        
        float left = static_cast<float>(start[1] + start[0]) / 2.0f;
        float right = lastStart + lastPatternSize / 2.0f;
        ArrayRef< Ref<ResultPoint> > resultPoints(2);
        resultPoints[0] =
        Ref<OneDResultPoint>(new OneDResultPoint(left, static_cast<float>(rowNumber)));
        resultPoints[1] =
        Ref<OneDResultPoint>(new OneDResultPoint(right, static_cast<float>(rowNumber)));
        return Ref<Result>(new Result(resultString, ArrayRef<char>(), resultPoints, BarcodeFormat::CODE_25));
        
        return Ref<Result>(NULL);
    }
    return Ref<Result>(NULL);
}

vector<int> Code25Reader::findStartPattern(Ref<BitArray> row, vector<int>& counters, ONED_READER_DATA* onedReaderData) {
    vector<int> counters_a;
    vector<int> counters_b;
    counters_a.resize(2);
    counters_b.resize(2);
    
    int counterOffset = onedReaderData->first_is_white ? 1 : 0;
    int patternLength = counters.size();
    int patternStart = onedReaderData->first_is_white ? onedReaderData->all_counters[0] : 0;
    
    for (int c = counterOffset; c < onedReaderData->counter_size - patternLength + 1; c += 2) {
        int i = patternStart;
        for (int ii = 0; ii < patternLength; ii++) {
            counters[ii] = onedReaderData->all_counters[c + ii];
            i += counters[ii];
            
            if (ii == 0 || ii == 2)
            {
                counters_a[ii/2] = counters[ii];
            }
            else
            {
                counters_b[ii/2] = counters[ii];
            }
        }
        
        // Look for whitespace before start pattern, >= 50% of width of
        // start pattern.
        
        ErrorHandler err_handler;
        if (
            counters_a[0] == counters_a[1] &&
            std::abs(counters_b[0] - counters_b[1]) < 2 &&
            counters_a[0] / counters_b[0] < 3 &&
            patternMatchVariance(counters, START_PATTERN, MAX_INDIVIDUAL_VARIANCE) < INT_MAX &&
            row->isRange((std::max)(0, patternStart - ((i - patternStart) >> 1)), patternStart, false , err_handler)&&
            true
           )
        {
                if (err_handler.ErrCode())   return vector<int>(0);
                vector<int> resultValue(2, 0);
                resultValue[0] = patternStart;
                resultValue[1] = i;
                return resultValue;
            }
        patternStart += counters[0] + counters[1];
    }
    
    return vector<int>(0);
}

// For efficiency, returns -1 on failure. Not throwing here saved as many as
// 700 exceptions per image when using some of our blackbox images.
bool Code25Reader::splitPattern(vector<int>& counters, int &blackPattern, int &whitePattern) {
    
    blackPattern = 0;
    whitePattern = 0;
    
    const int numCounters = counters.size();
    
    int minCounterBlack = INT_MAX;
    int minCounterWhite = INT_MAX;
    
    {  // get base module
        int realMinCounterBlack = INT_MAX;
        int realMinCounterWhite = INT_MAX;
        int realMaxCounterBlack = 0;
        int realMaxCounterWhite = 0;
        
        // find real min
        for (int i = 0; i < numCounters; i++) {
            int counter = counters[i];
            if (i & 1) {
                if (counter < realMinCounterWhite)
                    realMinCounterWhite = counter;
                if (counter > realMaxCounterWhite)
                    realMaxCounterWhite = counter;
            }
            else
            {
                if (counter < realMinCounterBlack)
                    realMinCounterBlack = counter;
                if (counter > realMaxCounterBlack)
                    realMaxCounterBlack = counter;
            }
        }
        
        minCounterBlack = realMaxCounterBlack;
        minCounterWhite = realMaxCounterWhite;
        
        // find base min
        for (int i = 0; i < numCounters; i++) {
            int counter = counters[i];
            if (i & 1)
            {
                if (counter < minCounterWhite && counter > realMinCounterWhite)
                    minCounterWhite = counter;
            }
            else
            {
                if (counter < minCounterBlack && counter > realMinCounterBlack)
                    minCounterBlack = counter;
            }
        }
        
        if (minCounterWhite == realMaxCounterWhite || minCounterWhite + 1 == realMaxCounterWhite)
            minCounterWhite = realMinCounterWhite;
        if (minCounterBlack == realMaxCounterBlack || minCounterBlack + 1 == realMaxCounterBlack)
            minCounterBlack= realMinCounterBlack;
    }
    
    float delta_big = 1;
    float delta_small= 1.8;
    for (int i = 0; i < numCounters; i++) {
        int counter = counters[i];
        
        int * piPattern = NULL;
        if (i & 1) {
            if (counter > (3 + delta_big) * minCounterWhite) return false;
            piPattern = &whitePattern;
        }
        else
        {
            if (counter > (3 + delta_big) * minCounterBlack) return false;
            piPattern = &blackPattern;
        }
        
        *piPattern = *piPattern << 1;
        
        if (i & 1)
        {
            if (counter > (3 - delta_small) * minCounterWhite)
                *piPattern |= 1;
        }
        else
        {
            if (counter > (3 - delta_small) * minCounterBlack)
                *piPattern |= 1;
        }
    }
    return true;
}

char Code25Reader::patternToChar(int pattern, ErrorHandler & err_handler) {
    for (int i = 0; i < NUMBER_ENCODINGS_LEN; i++) {
        if (NUMBER_ENCODINGS[i] == pattern) {
            return i + '0';
        }
    }
    
    err_handler = ErrorHandler(-1);
    return 0;
}
