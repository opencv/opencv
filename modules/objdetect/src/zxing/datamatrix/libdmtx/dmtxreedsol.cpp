//
//  dmtxreedsol.cpp
//  test_dm
//
//  Created by wechatcv on 2022/5/7.
//

#include "dmtxreedsol.hpp"
#include "dmtxbytelist.hpp"
#include "dmtxsymbol.hpp"

#include <math.h>

namespace dmtx {

#define NN                      255
#define MAX_ERROR_WORD_COUNT     68

/* GF add (a + b) */
#define GfAdd(a, b) \
((a) ^ (b))

/* GF multiply (a * b) */
#define GfMult(a, b) \
(((a) == 0 || (b) == 0) ? 0 : antilog301[(log301[(a)] + log301[(b)]) % NN])

/* GF multiply by antilog (a * alpha**b) */
#define GfMultAntilog(a, b) \
(((a) == 0) ? 0 : antilog301[(log301[(a)] + (b)) % NN])

/* GF(256) log values using primitive polynomial 301 */
static DmtxByte log301[] =
{ 255,   0,   1, 240,   2, 225, 241,  53,   3,  38, 226, 133, 242,  43,  54, 210,
    4, 195,  39, 114, 227, 106, 134,  28, 243, 140,  44,  23,  55, 118, 211, 234,
    5, 219, 196,  96,  40, 222, 115, 103, 228,  78, 107, 125, 135,   8,  29, 162,
    244, 186, 141, 180,  45,  99,  24,  49,  56,  13, 119, 153, 212, 199, 235,  91,
    6,  76, 220, 217, 197,  11,  97, 184,  41,  36, 223, 253, 116, 138, 104, 193,
    229,  86,  79, 171, 108, 165, 126, 145, 136,  34,   9,  74,  30,  32, 163,  84,
    245, 173, 187, 204, 142,  81, 181, 190,  46,  88, 100, 159,  25, 231,  50, 207,
    57, 147,  14,  67, 120, 128, 154, 248, 213, 167, 200,  63, 236, 110,  92, 176,
    7, 161,  77, 124, 221, 102, 218,  95, 198,  90,  12, 152,  98,  48, 185, 179,
    42, 209,  37, 132, 224,  52, 254, 239, 117, 233, 139,  22, 105,  27, 194, 113,
    230, 206,  87, 158,  80, 189, 172, 203, 109, 175, 166,  62, 127, 247, 146,  66,
    137, 192,  35, 252,  10, 183,  75, 216,  31,  83,  33,  73, 164, 144,  85, 170,
    246,  65, 174,  61, 188, 202, 205, 157, 143, 169,  82,  72, 182, 215, 191, 251,
    47, 178,  89, 151, 101,  94, 160, 123,  26, 112, 232,  21,  51, 238, 208, 131,
    58,  69, 148,  18,  15,  16,  68,  17, 121, 149, 129,  19, 155,  59, 249,  70,
    214, 250, 168,  71, 201, 156,  64,  60, 237, 130, 111,  20,  93, 122, 177, 150 };

/* GF(256) antilog values using primitive polynomial 301 */
static DmtxByte antilog301[] =
{   1,   2,   4,   8,  16,  32,  64, 128,  45,  90, 180,  69, 138,  57, 114, 228,
    229, 231, 227, 235, 251, 219, 155,  27,  54, 108, 216, 157,  23,  46,  92, 184,
    93, 186,  89, 178,  73, 146,   9,  18,  36,  72, 144,  13,  26,  52, 104, 208,
    141,  55, 110, 220, 149,   7,  14,  28,  56, 112, 224, 237, 247, 195, 171, 123,
    246, 193, 175, 115, 230, 225, 239, 243, 203, 187,  91, 182,  65, 130,  41,  82,
    164, 101, 202, 185,  95, 190,  81, 162, 105, 210, 137,  63, 126, 252, 213, 135,
    35,  70, 140,  53, 106, 212, 133,  39,  78, 156,  21,  42,  84, 168, 125, 250,
    217, 159,  19,  38,  76, 152,  29,  58, 116, 232, 253, 215, 131,  43,  86, 172,
    117, 234, 249, 223, 147,  11,  22,  44,  88, 176,  77, 154,  25,  50, 100, 200,
    189,  87, 174, 113, 226, 233, 255, 211, 139,  59, 118, 236, 245, 199, 163, 107,
    214, 129,  47,  94, 188,  85, 170, 121, 242, 201, 191,  83, 166,  97, 194, 169,
    127, 254, 209, 143,  51, 102, 204, 181,  71, 142,  49,  98, 196, 165, 103, 206,
    177,  79, 158,  17,  34,  68, 136,  61, 122, 244, 197, 167,  99, 198, 161, 111,
    222, 145,  15,  30,  60, 120, 240, 205, 183,  67, 134,  33,  66, 132,  37,  74,
    148,   5,  10,  20,  40,  80, 160, 109, 218, 153,  31,  62, 124, 248, 221, 151,
    3,   6,  12,  24,  48,  96, 192, 173, 119, 238, 241, 207, 179,  75, 150,   0 };

///

#undef CHKPASS
#define CHKPASS { if (passFail == DmtxFail) return DmtxFail; }

unsigned int RsDecode(unsigned char *code, int sizeIdx, int fix)
{
    (void)fix;
    int i;
    int blockStride, blockIdx;
    int blockDataWords, blockErrorWords, blockMaxCorrectable;
    
    int symbolDataWords, symbolErrorWords, symbolTotalWords;
    DmtxBoolean error, repairable;
    unsigned int passFail;
    unsigned char *word;
    DmtxByte elpStorage[MAX_ERROR_WORD_COUNT];
    DmtxByte synStorage[MAX_ERROR_WORD_COUNT+1];
    DmtxByte recStorage[NN];
    DmtxByte locStorage[NN];
    DmtxByteList elp = dmtxByteListBuild(elpStorage, sizeof(elpStorage));
    DmtxByteList syn = dmtxByteListBuild(synStorage, sizeof(synStorage));
    DmtxByteList rec = dmtxByteListBuild(recStorage, sizeof(recStorage));
    DmtxByteList loc = dmtxByteListBuild(locStorage, sizeof(locStorage));
    
    blockStride = dmtxGetSymbolAttribute(DmtxSymAttribInterleavedBlocks, sizeIdx);
    blockErrorWords = dmtxGetSymbolAttribute(DmtxSymAttribBlockErrorWords, sizeIdx);
    blockMaxCorrectable = dmtxGetSymbolAttribute(DmtxSymAttribBlockMaxCorrectable, sizeIdx);
    symbolDataWords = dmtxGetSymbolAttribute(DmtxSymAttribSymbolDataWords, sizeIdx);
    symbolErrorWords = dmtxGetSymbolAttribute(DmtxSymAttribSymbolErrorWords, sizeIdx);
    symbolTotalWords = symbolDataWords + symbolErrorWords;
    
    /* For each interleaved block */
    for (blockIdx = 0; blockIdx < blockStride; blockIdx++)
    {
        /* Data word count depends on blockIdx due to special case at 144x144 */
        blockDataWords = dmtxGetBlockDataSize(sizeIdx, blockIdx);
        
        /* Populate received list (rec) with data and error codewords */
        passFail = dmtxByteListInit(&rec, 0, 0); CHKPASS;
        
        /* Start with final error word and work backward */
        word = code + symbolTotalWords + blockIdx - blockStride;
        for (i = 0; i < blockErrorWords; i++)
        {
            passFail = dmtxByteListPush(&rec, *word); CHKPASS;
            word -= blockStride;
        }
        
        /* Start with final data word and work backward */
        word = code + blockIdx + (blockStride * (blockDataWords - 1));
        for (i = 0; i < blockDataWords; i++)
        {
            passFail = dmtxByteListPush(&rec, *word); CHKPASS;
            word -= blockStride;
        }
        
        /* Compute syndromes (syn) */
        error = RsComputeSyndromes(&syn, &rec, blockErrorWords);
        
        /* Error(s) detected: Attempt repair */
        if (error)
        {
            /* Find error locator polynomial (elp) */
            repairable = RsFindErrorLocatorPoly(&elp, &syn, blockErrorWords, blockMaxCorrectable);
            if (!repairable)
                return DmtxFail;
            
            /* Find error positions (loc) */
            repairable = RsFindErrorLocations(&loc, &elp);
            if (!repairable)
                return DmtxFail;
            
            /* Find error values and repair */
            RsRepairErrors(&rec, &loc, &elp, &syn);
        }
        
        /*
         * Overwrite output with correct/corrected values
         */
        
        /* Start with first data word and work forward */
        word = code + blockIdx;
        for (i = 0; i < blockDataWords; i++)
        {
            *word = dmtxByteListPop(&rec, &passFail); CHKPASS;
            word += blockStride;
        }
        
        /* Start with first error word and work forward */
        word = code + symbolDataWords + blockIdx;
        for (i = 0; i < blockErrorWords; i++)
        {
            *word = dmtxByteListPop(&rec, &passFail); CHKPASS;
            word += blockStride;
        }
    }
    
    return DmtxPass;
}

#undef CHKPASS
#define CHKPASS { if (passFail == DmtxFail) return DmtxTrue; }
static DmtxBoolean RsComputeSyndromes(DmtxByteList *syn, const DmtxByteList *rec, int blockErrorWords)
{
    int i, j;
    unsigned int passFail;
    DmtxBoolean error = DmtxFalse;
    
    /* Initialize all coefficients to 0 */
    passFail = dmtxByteListInit(syn, blockErrorWords + 1, 0); CHKPASS;
    
    for (i = 1; i < syn->length; i++)
    {
        /* Calculate syndrome at i */
        for (j = 0; j < rec->length; j++) /* alternatively: j < blockTotalWords */
            syn->b[i] = GfAdd(syn->b[i], GfMultAntilog(rec->b[j], i*j));
        
        /* Non-zero syndrome indicates presence of error(s) */
        if (syn->b[i] != 0)
            error = DmtxTrue;
    }
    
    return error;
}

#undef CHKPASS
#define CHKPASS { if (passFail == DmtxFail) return DmtxFalse; }
static DmtxBoolean RsFindErrorLocatorPoly(DmtxByteList *elpOut, const DmtxByteList *syn, int errorWordCount, int maxCorrectable)
{
    int i, iNext, j;
    int m, mCmp, lambda;
    DmtxByte disTmp, disStorage[MAX_ERROR_WORD_COUNT+1];
    DmtxByte elpStorage[MAX_ERROR_WORD_COUNT+2][MAX_ERROR_WORD_COUNT];
    DmtxByteList dis, elp[MAX_ERROR_WORD_COUNT+2];
    unsigned int passFail;
    
    dis = dmtxByteListBuild(disStorage, sizeof(disStorage));
    passFail = dmtxByteListInit(&dis, 0, 0); CHKPASS;
    
    for (i = 0; i < MAX_ERROR_WORD_COUNT + 2; i++)
    {
        elp[i] = dmtxByteListBuild(elpStorage[i], sizeof(elpStorage[i]));
        passFail = dmtxByteListInit(&elp[i], 0, 0); CHKPASS;
    }
    
    /* iNext = 0 */
    passFail = dmtxByteListPush(&elp[0], 1); CHKPASS;
    passFail = dmtxByteListPush(&dis, 1); CHKPASS;
    
    /* iNext = 1 */
    passFail = dmtxByteListPush(&elp[1], 1); CHKPASS;
    passFail = dmtxByteListPush(&dis, syn->b[1]); CHKPASS;
    
    for (iNext = 2, i = 1; /* explicit break */; i = iNext++)
    {
        if (dis.b[i] == 0)
        {
            /* Simple case: Copy directly from previous iteration */
            passFail = dmtxByteListCopy(&elp[iNext], &elp[i]); CHKPASS;
        }
        else
        {
            /* Find earlier iteration (m) that provides maximal (m - lambda) */
            for (m = 0, mCmp = 1; mCmp < i; mCmp++)
                if (dis.b[mCmp] != 0 && (mCmp - elp[mCmp].length) >= (m - elp[m].length))
                    m = mCmp;
            
            /* Calculate error location polynomial elp[i] (set 1st term) */
            for (lambda = elp[m].length - 1, j = 0; j <= lambda; j++)
                elp[iNext].b[j+i-m] = (elp[i - 1].b[j] == 0) ? 0 :
                antilog301[(NN - log301[dis.b[m]] +
                            log301[dis.b[i]] + log301[elp[m].b[j]]) % NN];
            
            /* Calculate error location polynomial elp[i] (add 2nd term) */
            for (lambda = elp[i].length - 1, j = 0; j <= lambda; j++)
                elp[iNext].b[j] = GfAdd(elp[iNext].b[j], elp[i].b[j]);
            
            elp[iNext].length = fmax(elp[i].length, elp[m].length + i - m);
        }
        
        lambda = elp[iNext].length - 1;
        if (i == errorWordCount || i >= lambda + maxCorrectable)
            break;
        
        /* Calculate discrepancy dis.b[i] */
        for (disTmp = syn->b[iNext], j = 1; j <= lambda; j++)
            disTmp = GfAdd(disTmp, GfMult(syn->b[iNext-j], elp[iNext].b[j]));
        
        // assert(dis.length == iNext);
        if (dis.length != iNext) return DmtxFalse;
        passFail = dmtxByteListPush(&dis, disTmp); CHKPASS;
    }
    
    passFail = dmtxByteListCopy(elpOut, &elp[iNext]); CHKPASS;
    
    return (lambda <= maxCorrectable) ? DmtxTrue : DmtxFalse;
}

#undef CHKPASS
#define CHKPASS { if (passFail == DmtxFail) return DmtxFalse; }
static DmtxBoolean RsFindErrorLocations(DmtxByteList *loc, const DmtxByteList *elp)
{
    int i, j;
    int lambda = elp->length - 1;
    unsigned int passFail;
    DmtxByte q, regStorage[MAX_ERROR_WORD_COUNT];
    DmtxByteList reg = dmtxByteListBuild(regStorage, sizeof(regStorage));
    
    passFail = dmtxByteListCopy(&reg, elp); CHKPASS;
    passFail = dmtxByteListInit(loc, 0, 0); CHKPASS;
    
    for (i = 1; i <= NN; i++)
    {
        for (q = 1, j = 1; j <= lambda; j++)
        {
            reg.b[j] = GfMultAntilog(reg.b[j], j);
            q = GfAdd(q, reg.b[j]);
        }
        
        if (q == 0)
        {
            passFail = dmtxByteListPush(loc, NN - i); CHKPASS;
        }
    }
    
    return (loc->length == lambda) ? DmtxTrue : DmtxFalse;
}

#undef CHKPASS
#define CHKPASS { if (passFail == DmtxFail) return DmtxFail; }
static unsigned int RsRepairErrors(DmtxByteList *rec, const DmtxByteList *loc, const DmtxByteList *elp, const DmtxByteList *syn)
{
    int i, j, q;
    int lambda = elp->length - 1;
    unsigned int passFail;
    DmtxByte zVal, root, err;
    DmtxByte zStorage[MAX_ERROR_WORD_COUNT+1];
    DmtxByteList z = dmtxByteListBuild(zStorage, sizeof(zStorage));
    
    /* Form polynomial z(x) */
    passFail = dmtxByteListPush(&z, 1); CHKPASS;
    for (i = 1; i <= lambda; i++)
    {
        for (zVal = GfAdd(syn->b[i], elp->b[i]), j = 1; j < i; j++)
            zVal= GfAdd(zVal, GfMult(elp->b[i-j], syn->b[j]));
        passFail = dmtxByteListPush(&z, zVal); CHKPASS;
    }
    
    for (i = 0; i < lambda; i++)
    {
        /* Calculate numerator of error term */
        root = NN - loc->b[i];
        
        for (err = 1, j = 1; j <= lambda; j++)
            err = GfAdd(err, GfMultAntilog(z.b[j], j * root));
        
        if (err == 0)
            continue;
        
        /* Calculate denominator of error term */
        for (q = 0, j = 0; j < lambda; j++)
        {
            if (j != i)
                q += log301[1 ^ antilog301[(loc->b[j] + root) % NN]];
        }
        q %= NN;
        
        err = GfMultAntilog(err, NN - q);
        rec->b[loc->b[i]] = GfAdd(rec->b[loc->b[i]], err);
    }
    
    return DmtxPass;
}

}  // namespace dmtx
