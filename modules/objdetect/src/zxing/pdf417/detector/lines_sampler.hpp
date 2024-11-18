// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_PDF417_DETECTOR_LINES_SAMPLER_HPP__
#define __ZXING_PDF417_DETECTOR_LINES_SAMPLER_HPP__

/*
 * Copyright 2010 ZXing authors All rights reserved.
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

#include <map>
#include "../../common/bit_matrix.hpp"
#include "../../result_point.hpp"
#include "../../common/point.hpp"
#include "../../error_handler.hpp"

namespace zxing {
namespace pdf417 {
namespace detector {

class PossibleCodeWords
{
public:
    PossibleCodeWords(const std::vector<std::vector<std::map<int, int> > > &vcVotes, int iRowCount);
    ~PossibleCodeWords() {};
    
    int getNextPossible(std::vector<std::vector<int> > &vcCodeWords);
    
private:
    int m_iHeight;
    int m_iWidth;
    int m_iForIndex;
    std::vector<int> m_vcBadPointList;
    std::vector<int> m_vcSecondIndex;
    std::vector<std::vector<std::map<int, int> > > m_vcVotes;
    std::vector<std::vector<int> > m_vcCodeWords;
};

class LinesSampler {
private:
    static const int MODULES_IN_SYMBOL = 17;
    static const int BARS_IN_SYMBOL = 8;
    static const int POSSIBLE_SYMBOLS = 2787;
    std::vector<float> RATIOS_TABLE;
    std::vector<float> init_ratios_table();
    static const int BARCODE_START_OFFSET = 2;
    const float MAX_EMPTY_LINE_RATIOS = 0.3;
    
    Ref<BitMatrix> linesMatrix_;
    // int symbolsPerLine_;
    int dimension_;
    PossibleCodeWords * m_ptPossibleCodeWords;
    
    void codewordsToBitMatrix(const std::vector<std::vector<int> > &codewords,
                              Ref<BitMatrix> &matrix);
    int calculateClusterNumber(int codeword);
    
    void computeSymbolWidths(std::vector<float>& symbolWidths,
                             const int symbolsPerLine, Ref<BitMatrix> linesMatrix);
   
    ErrorHandler linesMatrixToCodewords(std::vector<std::vector<int> > &clusterNumbers,
                                        const int symbolsPerLine,
                                        const std::vector<float> &symbolWidths,
                                        Ref<BitMatrix> linesMatrix,
                                        std::vector<std::vector<int> > &codewords);
    std::vector<std::vector<std::map<int, int> > >
    distributeVotes(const int symbolsPerLine,
                    const std::vector<std::vector<int> >& codewords,
                    const std::vector<std::vector<int> >& clusterNumbers);
    std::vector<int>
    findMissingLines(const int symbolsPerLine,
                     std::vector<std::vector<int> > &detectedCodeWords, int &iEmptyLine);
    int decodeRowCount(const int symbolsPerLine,
                       std::vector<std::vector<int> > &detectedCodeWords,
                       std::vector<int> &insertLinesAt,
                       int &iEmptyLine);
    
    int round(float d);
    Point intersection(Line a, Line b);
    int getBitCountSum(const std::vector<int> &vcModuleBitCount);
    std::vector<int> sampleBitCounts(const std::vector<int> &vcModuleBitCount);
    int getBitValue(const std::vector<int> &vcModuleBitCount);
    
public:
    LinesSampler(Ref<BitMatrix> linesMatrix, int dimension);
    ~LinesSampler()
    {
        if (m_ptPossibleCodeWords != NULL)
        {
            delete m_ptPossibleCodeWords;
            m_ptPossibleCodeWords = NULL;
        }
    }
    void setLineMatrix(Ref<BitMatrix> linesMatrix);
    ErrorHandler sample(Ref<BitMatrix> & bit_matrix);
    Ref<BitMatrix> getNextPossibleGrid(ErrorHandler &err_handler);
};

}  // namespace detector
}  // namespace pdf417
}  // namespace zxing

#endif  // __ZXING_PDF417_DETECTOR_LINES_SAMPLER_HPP__
