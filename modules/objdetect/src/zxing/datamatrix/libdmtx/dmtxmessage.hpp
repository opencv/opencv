//
//  dmtxmessage.hpp
//  test_dm
//
//  Created by wechatcv on 2022/5/5.
//

#ifndef dmtxmessage_hpp
#define dmtxmessage_hpp

#include <stdio.h>
#include <vector>
#include "common.hpp"

namespace dmtx {

class DmtxMessage {
public:
    DmtxMessage() {}
    ~DmtxMessage();
    
    int Init(int sizeIdx, int symbolFormat);
    
    unsigned int DecodeDataStream(int sizeIdx, unsigned char *outputStart);
    
private:
    int GetEncodationScheme(unsigned char cw);
    DmtxBoolean ValidOutputWord(int value);
    unsigned int PushOutputWord(int value);
    unsigned int PushOutputC40TextWord(C40TextState *state, int value);
    unsigned int PushOutputMacroHeader(int macroType);
    void PushOutputMacroTrailer();
    unsigned char *DecodeSchemeAscii(unsigned char *ptr, unsigned char *dataEnd);
    unsigned char *DecodeSchemeC40Text(unsigned char *ptr, unsigned char *dataEnd, DmtxScheme encScheme);
    unsigned char *DecodeSchemeX12(unsigned char *ptr, unsigned char *dataEnd);
    unsigned char *DecodeSchemeEdifact(unsigned char *ptr, unsigned char *dataEnd);
    unsigned char *DecodeSchemeBase256(unsigned char *ptr, unsigned char *dataEnd);
    unsigned char UnRandomize255State(unsigned char value, int idx);
    
public:
    size_t          arraySize;     /* mappingRows * mappingCols */
    size_t          codeSize;      /* Size of encoded data (data words + error words) */
    size_t          outputSize;    /* Size of buffer used to hold decoded data */
    int             outputIdx;     /* Internal index used to store output progress */
    int             padCount;
    int             fnc1;          /* Character to represent FNC1, or DmtxUndefined */
    unsigned char  *array;         /* Pointer to internal representation of Data Matrix modules */
    unsigned char  *code;          /* Pointer to internal storage of code words (data and error) */
    unsigned char  *output;        /* Pointer to internal storage of decoded output */
    
    std::vector<DmtxPixelLoc> points;
};

}  // namespace dmtx

#endif /* dmtxmessage_hpp */
