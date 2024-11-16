#pragma once

#include "one_dreader.hpp"
#include "../common/bit_array.hpp"
#include "../error_handler.hpp"
#include "../result.hpp"

namespace zxing {
namespace oned {

/**
 * <p>Decodes Code 39 barcodes. This does not support "Full ASCII Code 39" yet.</p>
 * Ported form Java (author Sean Owen)
 * @author Lukasz Warchol
 */
class Code25Reader : public OneDReader {
private:
    bool usingCheckDigit;
    bool extendedMode;
    std::string decodeRowResult;
    std::vector<int> counters;
    
    void init(bool usingCheckDigit = false, bool extendedMode = false);
    
    static std::vector<int> findStartPattern(Ref<BitArray> row,
                                             std::vector<int>& counters, ONED_READER_DATA* onedReaderData);
    
    static bool splitPattern(std::vector<int>& counters, int &blackPattern, int &whitePattern);
    static char patternToChar(int pattern, ErrorHandler & err_handler);
    
    static const int MAX_AVG_VARIANCE;
    static const int MAX_INDIVIDUAL_VARIANCE;
    
public:
    Code25Reader();
    Code25Reader(bool usingCheckDigit_);
    Code25Reader(bool usingCheckDigit_, bool extendedMode_);
    
    Ref<Result> decodeRow(int rowNumber, Ref<BitArray> row);
};

}  // namespace oned
}  // namespace zxing
