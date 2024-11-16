// -*- mode:c++; tab-width:2; indent-tabs-mode:nil; c-basic-offset:2 -*-
/*
 * Copyright 2010, 2012 ZXing authors All rights reserved.
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
#include "lines_sampler.hpp"
#include "../decoder/bit_matrix_parser.hpp"
#include "../../not_found_exception.hpp"
#include "../../common/point.hpp"
#include <iostream>
#include <cmath>
#include <cstdio>
#include <algorithm>

using std::map;
using std::vector;
using std::min;
using zxing::pdf417::detector::LinesSampler;
using zxing::pdf417::detector::PossibleCodeWords;
using zxing::pdf417::decoder::BitMatrixParser;
using zxing::Ref;
using zxing::BitMatrix;
using zxing::NotFoundException;
using zxing::NotFoundErrorHandler;
using zxing::ErrorHandler;
using zxing::Point;

// VC++
using zxing::Line;

const int LinesSampler::MODULES_IN_SYMBOL;
const int LinesSampler::BARS_IN_SYMBOL;
const int LinesSampler::POSSIBLE_SYMBOLS;
const int LinesSampler::BARCODE_START_OFFSET;
const float MAX_DIFF_FROM_LAST_SYMBOL = 0.0075;

namespace {

class VoteResult {
private:
    bool indecisive;
    int vote;
public:
    VoteResult() : indecisive(false), vote(0) {}
    bool isIndecisive() {
        return indecisive;
    }
    void setIndecisive(bool indecisive_) {
        this->indecisive = indecisive_;
    }
    int getVote() {
        return vote;
    }
    void setVote(int vote_) {
        this->vote = vote_;
    }
};

VoteResult getValueWithMaxVotes(map<int, int>& votes) {
    VoteResult result;
    int maxVotes = 0;
    for (map<int, int>::iterator i = votes.begin(); i != votes.end(); i++) {
        if (i->second > maxVotes)
        {
            maxVotes = i->second;
            result.setVote(i->first);
            result.setIndecisive(false);
        }
        else if (i->second == maxVotes)
        {
            result.setIndecisive(true);
        }
    }
    return result;
}

}  // namespace

PossibleCodeWords::PossibleCodeWords(const std::vector<std::vector<std::map<int, int> > > &vcVotes, int iRowCount) :
m_vcVotes(vcVotes)
{
    m_iForIndex = 0;
    m_iHeight = iRowCount;
    if (m_iHeight > static_cast<int>(m_vcVotes.size())) m_iHeight = m_vcVotes.size();
    m_iWidth = m_vcVotes[0].size();
    m_vcCodeWords.resize(iRowCount);
    for (int i = 0; i < m_iHeight; i++)
    {
        m_vcCodeWords[i].resize(m_iWidth, 0);
        for (size_t j = 0; j < m_vcVotes[i].size(); j++)
        {
            int iFirstVoteValue = -1;
            int iFirstVoteIndex = -1;
            int iSecondVoteValue = -1;
            int iSecondVoteIndex = -1;
            for (map<int, int>::iterator k = m_vcVotes[i][j].begin(); k != m_vcVotes[i][j].end(); k++)
            {
                if (k->second > iFirstVoteValue)
                {
                    iSecondVoteValue = iFirstVoteValue;
                    iSecondVoteIndex = iFirstVoteIndex;
                    iFirstVoteValue = k->second;
                    iFirstVoteIndex = k->first;
                }
                else if (k->second > iSecondVoteValue)
                {
                    iSecondVoteValue = k->second;
                    iSecondVoteIndex = k->first;
                }
            }
            if (iFirstVoteIndex != -1) m_vcCodeWords[i][j] = iFirstVoteIndex;
            if (iFirstVoteIndex != -1 && iSecondVoteIndex != -1 && iSecondVoteValue * 1.0 / iFirstVoteValue > 0.3)
            {
                m_vcBadPointList.push_back(i * m_iWidth + j);
                m_vcSecondIndex.push_back(iSecondVoteIndex);
            }
        }
    }
}

int PossibleCodeWords::GetNextPossible(std::vector<std::vector<int> > &vcCodeWords)
{
    int iListSize = m_vcBadPointList.size();
    vcCodeWords = m_vcCodeWords;

    if (iListSize > 6)
    {
        if (m_iForIndex < iListSize)
        {
            int iOffset = m_vcBadPointList[m_iForIndex];
            int iSecondCodeWord = m_vcSecondIndex[m_iForIndex];
            vcCodeWords[iOffset / m_iWidth][iOffset % m_iWidth] = iSecondCodeWord;
        }
        else if (m_iForIndex < iListSize * iListSize + iListSize)
        {
            int iRealIndex = m_iForIndex - iListSize;
            int iOffset = m_vcBadPointList[iRealIndex / iListSize];
            int iSecondCodeWord = m_vcSecondIndex[iRealIndex / iListSize];
            vcCodeWords[iOffset / m_iWidth][iOffset % m_iWidth] = iSecondCodeWord;
            
            iOffset = m_vcBadPointList[iRealIndex % iListSize];
            iSecondCodeWord = m_vcSecondIndex[iRealIndex % iListSize];
            vcCodeWords[iOffset / m_iWidth][iOffset % m_iWidth] = iSecondCodeWord;
        }
        else return 1;
    }
    else
    {
        if (m_iForIndex >= (1 << iListSize)) return 1;
        for (int i = 0; i < iListSize; ++i)
        {
            if (m_iForIndex&(1 << i))
            {
                int iOffset = m_vcBadPointList[i];
                int iSecondCodeWord = m_vcSecondIndex[i];
                vcCodeWords[iOffset / m_iWidth][iOffset % m_iWidth] = iSecondCodeWord;
            }
        }
    }
    ++m_iForIndex;
    return 0;
}

vector<float> LinesSampler::init_ratios_table() {
    // Pre-computes and outputs the symbol ratio table.
    vector<vector<float> > table (BitMatrixParser::SYMBOL_TABLE_LENGTH);
    for (size_t i = 0; i < table.size(); ++i) {
        table[i].resize(LinesSampler::BARS_IN_SYMBOL);
    }
    vector<float> RATIOS_TABLE_TMP (BitMatrixParser::SYMBOL_TABLE_LENGTH * LinesSampler::BARS_IN_SYMBOL);
    int x = 0;
    for (int i = 0; i < BitMatrixParser::SYMBOL_TABLE_LENGTH; i++) {
        int currentSymbol = BitMatrixParser::SYMBOL_TABLE[i];
        int currentBit = currentSymbol & 0x1;
        for (int j = 0; j < BARS_IN_SYMBOL; j++) {
            float size = 0.0f;
            while ((currentSymbol & 0x1) == currentBit) {
                size += 1.0f;
                currentSymbol >>= 1;
            }
            currentBit = currentSymbol & 0x1;
            table[i][BARS_IN_SYMBOL - j - 1] = size / MODULES_IN_SYMBOL;
        }
        for (int j = 0; j < BARS_IN_SYMBOL; j++) {
            RATIOS_TABLE_TMP[x] = table[i][j];
            x++;
        }
    }
    return RATIOS_TABLE_TMP;
}


LinesSampler::LinesSampler(Ref<BitMatrix> linesMatrix, int dimension)
: linesMatrix_(linesMatrix), dimension_(dimension) {
    m_ptPossibleCodeWords = NULL;
    RATIOS_TABLE = init_ratios_table();
}

void LinesSampler::SetLineMatrix(Ref<BitMatrix> linesMatrix)
{
    linesMatrix_ = linesMatrix;
}

/**
 * Samples a grid from a lines matrix.
 *
 * @return the potentially decodable bit matrix.
 */
// Ref<BitMatrix> LinesSampler::sample() {
ErrorHandler LinesSampler::sample(Ref<BitMatrix> & bit_matrix) {
    const int symbolsPerLine = dimension_ / MODULES_IN_SYMBOL;
    
    ErrorHandler err_handler;
    // XXX
    vector<float> symbolWidths;
    computeSymbolWidths(symbolWidths, symbolsPerLine, linesMatrix_);
    
    // XXX
    vector<vector<int> > codewords(linesMatrix_->getHeight());
    vector<vector<int> > clusterNumbers(linesMatrix_->getHeight());
    err_handler = linesMatrixToCodewords(clusterNumbers, symbolsPerLine, symbolWidths, linesMatrix_, codewords);
    if (err_handler.ErrCode()) return err_handler;
    
    vector<vector<map<int, int> > > votes =
    distributeVotes(symbolsPerLine, codewords, clusterNumbers);
    
    vector<vector<int> > detectedCodeWords(votes.size());
    for (size_t i = 0; i < votes.size(); i++) {
        detectedCodeWords[i].resize(votes[i].size(), 0);
        for (size_t j = 0; j < votes[i].size(); j++) {
            if (!votes[i][j].empty()) {
                detectedCodeWords[i][j] = getValueWithMaxVotes(votes[i][j]).getVote();
            }
        }
    }
    
    int iEmptyLine = 0;
    
    // XXX
    vector<int> insertLinesAt = findMissingLines(symbolsPerLine, detectedCodeWords, iEmptyLine);
    
    // XXX
    int rowCount = decodeRowCount(symbolsPerLine, detectedCodeWords, insertLinesAt, iEmptyLine);
    
    if (iEmptyLine * 1.0 / detectedCodeWords.size() > MAX_EMPTY_LINE_RATIOS)
    {
        bit_matrix = Ref<BitMatrix>();
        return err_handler;
    }
    
    if (iEmptyLine == 0) m_ptPossibleCodeWords = new PossibleCodeWords(votes, rowCount);
    
    detectedCodeWords.resize(rowCount);
    
    // XXX
    Ref<BitMatrix> grid(new BitMatrix(dimension_, detectedCodeWords.size(), err_handler));
    if (err_handler.ErrCode()) return err_handler;
    
    codewordsToBitMatrix(detectedCodeWords, grid);
    
    bit_matrix = grid;
    return err_handler;
}

Ref<BitMatrix> LinesSampler::GetNextPossibleGrid(ErrorHandler & err_handler)
{
    if (m_ptPossibleCodeWords == NULL) return Ref<BitMatrix>();
    vector<vector<int> > detectedCodeWords;
    if (m_ptPossibleCodeWords->GetNextPossible(detectedCodeWords)) return Ref<BitMatrix>();
    Ref<BitMatrix> grid(new BitMatrix(dimension_, detectedCodeWords.size(), err_handler));
    if (err_handler.ErrCode())   return Ref<BitMatrix>();
    codewordsToBitMatrix(detectedCodeWords, grid);
    return grid;
}

/**
 * @brief LinesSampler::codewordsToBitMatrix
 * @param codewords
 * @param matrix
 */
void LinesSampler::codewordsToBitMatrix(const vector<vector<int> > &codewords, Ref<BitMatrix> &matrix) {
    for (size_t i = 0; i < codewords.size(); i++) {
        for (size_t j = 0; j < codewords[i].size(); j++) {
            int moduleOffset = j * MODULES_IN_SYMBOL;
            for (int k = 0; k < MODULES_IN_SYMBOL; k++) {
                if ((codewords[i][j] & (1 << (MODULES_IN_SYMBOL - k - 1))) > 0) {
                    matrix->set(moduleOffset + k, i);
                }
            }
        }
    }
}

/**
 * @brief LinesSampler::calculateClusterNumber
 * @param codeword
 * @return
 */
int LinesSampler::calculateClusterNumber(int codeword) {
    if (codeword == 0) {
        return -1;
    }
    int barNumber = 0;
    bool blackBar = true;
    int clusterNumber = 0;
    for (int i = 0; i < MODULES_IN_SYMBOL; i++) {
        if ((codeword & (1 << i)) > 0) {
            if (!blackBar)
            {
                blackBar = true;
                barNumber++;
            }
            if (barNumber % 2 == 0)
            {
                clusterNumber++;
            }
            else
            {
                clusterNumber--;
            }
        }
        else
        {
            if (blackBar)
            {
                blackBar = false;
            }
        }
    }
    return (clusterNumber + 9) % 9;
}

void LinesSampler::computeSymbolWidths(vector<float> &symbolWidths, const int symbolsPerLine, Ref<BitMatrix> linesMatrix)
{
    int symbolStart = 0;
    bool lastWasSymbolStart = true;
    const float symbolWidth = symbolsPerLine > 0 ? static_cast<float>(linesMatrix->getWidth()) / static_cast<float>(symbolsPerLine) : static_cast<float>(linesMatrix->getWidth());
    
    // Use the following property of PDF417 barcodes to detect symbols:
    // Every symbol starts with a black module and every symbol is 17 modules wide,
    // therefore there have to be columns in the line matrix that are completely composed of black pixels.
    vector<int> blackCount(linesMatrix->getWidth(), 0);
    vector<int> whiteCount(linesMatrix->getWidth(), 0);
    int iLastWhiteCol = 0;
    for (int x = BARCODE_START_OFFSET; x < linesMatrix->getWidth(); x++) {
        int iMaxLen = 0;
        for (int y = 0; y < linesMatrix->getHeight(); y++) {
            if (linesMatrix->get(x, y))
            {
                blackCount[x]++;
                iMaxLen = (std::max)(iMaxLen, blackCount[x]);
            }
            else
            {
                whiteCount[x]++;
                blackCount[x] = 0;
            }
        }
        blackCount[x] = iMaxLen;
       
        if (blackCount[x] * 1.0 / linesMatrix->getHeight() > 0.9 && x - iLastWhiteCol <= 5)
        {
            if (!lastWasSymbolStart)
            {
                float currentWidth = static_cast<float>(x - symbolStart);
                // Make sure we really found a symbol by asserting a minimal size of 75% of the expected symbol width.
                // This might break highly distorted barcodes, but fixes an issue with barcodes where there is a
                // full black column from top to bottom within a symbol.
                if (currentWidth > 0.75 * symbolWidth)
                {
                    // The actual symbol width might be slightly bigger than the expected symbol width,
                    // but if we are more than half an expected symbol width bigger, we assume that
                    // we missed one or more symbols and assume that they were the expected symbol width.
                    while (currentWidth > 1.5 * symbolWidth) {
                        symbolWidths.push_back(symbolWidth);
                        currentWidth -= symbolWidth;
                    }
                    symbolWidths.push_back(currentWidth);
                    lastWasSymbolStart = true;
                    symbolStart = x;
                }
            }
        }
        else
        {
            if (lastWasSymbolStart)
            {
                lastWasSymbolStart = false;
            }
        }
        if (whiteCount[x] * 1.0 / linesMatrix->getHeight() > 0.95)
        {
            iLastWhiteCol = x;
        }
    }
    
    // The last symbol ends at the right edge of the matrix, where there usually is no black bar.
    float currentWidth = static_cast<float>(linesMatrix->getWidth() - symbolStart);
    while (currentWidth > 1.5 * symbolWidth) {
        symbolWidths.push_back(symbolWidth);
        currentWidth -= symbolWidth;
    }
    symbolWidths.push_back(currentWidth);
}

int LinesSampler::GetBitCountSum(const vector<int> &vcModuleBitCount)
{
    int iBitCountSum = 0;
    for (size_t i = 0; i < vcModuleBitCount.size(); ++i)
    {
        iBitCountSum += vcModuleBitCount[i];
    }
    return iBitCountSum;
}

vector<int> LinesSampler::SampleBitCounts(const vector<int> &vcModuleBitCount)
{
    float fBitCountSum = GetBitCountSum(vcModuleBitCount);
    vector<int> vcResult( LinesSampler::BARS_IN_SYMBOL);
    int iBitCountIndex = 0;
    int iSumPreviousBits = 0;
    for (int i = 0; i < MODULES_IN_SYMBOL; ++i)
    {
        float fSampleIndex = fBitCountSum / (2 * MODULES_IN_SYMBOL)
        + (i * fBitCountSum) / MODULES_IN_SYMBOL;
        if (iSumPreviousBits + vcModuleBitCount[iBitCountIndex] <= fSampleIndex)
        {
            iSumPreviousBits += vcModuleBitCount[iBitCountIndex];
            iBitCountIndex++;
        }
        vcResult[iBitCountIndex]++;
    }
    return vcResult;
}

int LinesSampler::GetBitValue(const vector<int> &vcModuleBitCount)
{
    int iResult = 0;
    for (size_t i = 0; i < vcModuleBitCount.size(); ++i)
    {
        for (int j = 0; j < vcModuleBitCount[i]; ++j)
        {
            iResult = (iResult << 1) | (i % 2 == 0 ? 1:0);
        }
    }
    return iResult;
}

ErrorHandler LinesSampler::linesMatrixToCodewords(vector<vector<int> >& clusterNumbers,
                                                  const int symbolsPerLine,
                                                  const vector<float>& symbolWidths,
                                                  Ref<BitMatrix> linesMatrix,
                                                  vector<vector<int> >& codewords)
{
    ErrorHandler err_handler;
    vector<int> aiPreSymbolIndex(symbolsPerLine, -1);
    for (int y = 0; y < linesMatrix->getHeight(); y++) {
        // Not sure if this is the right way to handle this but avoids an error:
        if (symbolsPerLine > static_cast<int>(symbolWidths.size())) {
            return NotFoundErrorHandler("Inconsistent number of symbols in this line.");
        }
        
        // TODO(sofiawu) use symbolWidths.size() instead of symbolsPerLine to at least decode some codewords
        
        codewords[y].resize(symbolsPerLine, 0);
        clusterNumbers[y].resize(symbolsPerLine, -1);
        int line = y;
        vector<int> barWidths(1, 0);
        int barCount = 0;
        // Runlength encode the bars in the scanned linesMatrix.
        // We assume that the first bar is black, as determined by the PDF417 standard.
        bool isSetBar = true;
        // Filter small white bars at the beginning of the barcode.
        // Small white bars may occur due to small deviations in scan line sampling.
        barWidths[0] += BARCODE_START_OFFSET;
        for (int x = BARCODE_START_OFFSET; x < linesMatrix->getWidth(); x++) {
            if (linesMatrix->get(x, line))
            {
                if (!isSetBar)
                {
                    isSetBar = true;
                    barCount++;
                    barWidths.resize(barWidths.size() + 1);
                }
            }
            else
            {
                if (isSetBar) {
                    isSetBar = false;
                    barCount++;
                    barWidths.resize(barWidths.size() + 1);
                }
                
            }
            barWidths[barCount]++;
        }
        // Don't forget the last bar.
        barCount++;
        barWidths.resize(barWidths.size() + 1);
        
        //////////////////////////////////////////////////
        
        // Find the symbols in the line by counting bar lengths until we reach symbolWidth.
        // We make sure, that the last bar of a symbol is always white, as determined by the PDF417 standard.
        // This helps to reduce the amount of errors done during the symbol recognition.
        // The symbolWidth usually is not constant over the width of the barcode.
        int cwWidth = 0;
        int cwCount = 0;
        vector<int> cwStarts(symbolsPerLine, 0);
        cwStarts[0] = 0;
        cwCount++;
        for (int i = 0; i < barCount && cwCount < symbolsPerLine; i++) {
            cwWidth += barWidths[i];
            if (static_cast<float>(cwWidth) > symbolWidths[cwCount - 1]) {
                if ((i % 2) == 1) {  // check if bar is white
                    i++;
                }
                cwWidth = barWidths[i];
                cwStarts[cwCount] = i;
                cwCount++;
            }
        }
        
        vector<vector<float> > cwRatios(symbolsPerLine);
        // Distribute bar widths to modules of a codeword.sss
        for (int i = 0; i < symbolsPerLine; i++) {
            cwRatios[i].resize(BARS_IN_SYMBOL, 0.0f);
            const int cwStart = cwStarts[i];
            const int cwEnd = (i == symbolsPerLine - 1) ? barCount : cwStarts[i + 1];
            const int cwLength = cwEnd - cwStart;
            
            if (cwLength < 7 || cwLength > 9) {
                // We try to recover smybols with 7 or 9 bars and spaces with heuristics, but everything else is beyond repair.
                continue;
            }
            
            float cwWidth_ = 0;
            
            // For symbols with 9 bar length simply ignore the last bar.
            for (int j = 0; j < min(BARS_IN_SYMBOL, cwLength); ++j) {
                cwWidth_ += static_cast<float>(barWidths[cwStart + j]);
            }
            
            if (cwLength == LinesSampler::BARS_IN_SYMBOL)
            {
                vector<int> vcModuleBitCount;
                for (int  j = 0; j < cwLength; ++j)
                {
                    vcModuleBitCount.push_back(barWidths[cwStart + j]);
                }
                vcModuleBitCount = SampleBitCounts(vcModuleBitCount);
                
                int iSymbol = GetBitValue(vcModuleBitCount);
                bool bFound  = std::binary_search(BitMatrixParser::SYMBOL_TABLE, BitMatrixParser::SYMBOL_TABLE + BitMatrixParser::SYMBOL_TABLE_LENGTH, iSymbol);
                if (bFound)
                {
                    codewords[y][i] = iSymbol;
                    clusterNumbers[y][i] = calculateClusterNumber(codewords[y][i]);
                    continue;
                }
            }
            
            
            
            // If there were only 7 bars and spaces detected use the following heuristic:
            // Assume the length of the symbol is symbolWidth and the last (unrecognized) bar uses all remaining space.
            if (cwLength == 7)
            {
                for (int j = 0; j < cwLength; ++j) {
                    cwRatios[i][j] = static_cast<float>(barWidths[cwStart + j]) / symbolWidths[i];
                }
                cwRatios[i][7] = (symbolWidths[i] - cwWidth_) / symbolWidths[i];
            }
            else
            {
                if (cwWidth_ == 0) continue;
                for (size_t j = 0; j < cwRatios[i].size(); ++j) {
                    cwRatios[i][j] = static_cast<float>(barWidths[cwStart + j]) / cwWidth_;
                }
            }
            
            float bestMatchError = std::numeric_limits<float>::max();
            int bestMatch = 0;
            
            // Search for the most possible codeword by comparing the ratios of bar size to symbol width.
            // The sum of the squared differences is used as similarity metric.
            // (Picture it as the square euclidian distance in the space of eight tuples where a tuple represents the bar ratios.)
            if (aiPreSymbolIndex[i] != -1)
            {
                float error = 0.0f;
                for (int k = 0; k < BARS_IN_SYMBOL; k++) {
                    float diff = RATIOS_TABLE[aiPreSymbolIndex[i] * BARS_IN_SYMBOL + k] - cwRatios[i][k];
                    error += diff * diff;
                    if (error >= bestMatchError) {
                        break;
                    }
                }
                if (error < MAX_DIFF_FROM_LAST_SYMBOL) {
                    bestMatchError = error;
                    bestMatch = BitMatrixParser::SYMBOL_TABLE[aiPreSymbolIndex[i]];
                }
            }
            if (bestMatch == 0)
            {
                aiPreSymbolIndex[i] = -1;
                for (int j = 0; j < POSSIBLE_SYMBOLS; j++) {
                    float error = 0.0f;
                    for (int k = 0; k < BARS_IN_SYMBOL; k++) {
                        float diff = RATIOS_TABLE[j * BARS_IN_SYMBOL + k] - cwRatios[i][k];
                        error += diff * diff;
                        if (error >= bestMatchError) {
                            break;
                        }
                    }
                    if (error < bestMatchError) {
                        bestMatchError = error;
                        bestMatch = BitMatrixParser::SYMBOL_TABLE[j];
                        aiPreSymbolIndex[i] = j;
                    }
                }
            }
            codewords[y][i] = bestMatch;
            clusterNumbers[y][i] = calculateClusterNumber(bestMatch);
        }
    }
    
    return err_handler;
}

vector<vector<map<int,  int> > >
LinesSampler::distributeVotes(const int symbolsPerLine,
                              const vector<vector<int> >& codewords,
                              const vector<vector<int> >& clusterNumbers)
{
    // Matrix of votes for codewords which are possible at this position.
    vector<vector<map<int, int> > > votes(1);
    votes[0].resize(symbolsPerLine);
    
    int currentRow = 0;
    map<int, int> clusterNumberVotes;
    int lastLineClusterNumber = -1;
    
    for (size_t y = 0; y < codewords.size(); y++) {
        clusterNumberVotes.clear();
        for (size_t i = 0; i < codewords[y].size(); i++) {
            if (clusterNumbers[y][i] != -1) {
                clusterNumberVotes[clusterNumbers[y][i]] = clusterNumberVotes[clusterNumbers[y][i]] + 1;
            }
        }
        
        if (!clusterNumberVotes.empty()) {
            VoteResult voteResult = getValueWithMaxVotes(clusterNumberVotes);
            bool lineClusterNumberIsIndecisive = voteResult.isIndecisive();
            int lineClusterNumber = voteResult.getVote();
            
            // This avoids switching lines because of damaged inter line readings, but
            // may cause problems for barcodes with four or less rows.
            if (lineClusterNumberIsIndecisive) {
                lineClusterNumber = lastLineClusterNumber;
            }
            
            if ((lineClusterNumber != ((lastLineClusterNumber + 3) % 9)) && (lastLineClusterNumber != -1)) {
                lineClusterNumber = lastLineClusterNumber;
            }
            
            if (clusterNumberVotes[lineClusterNumber] * 1.0 / symbolsPerLine < 0.25)
            {
                lineClusterNumber = lastLineClusterNumber;
            }
            
            // Ignore broken lines at the beginning of the barcode.
            if ((lineClusterNumber == 0 && lastLineClusterNumber == -1) || (lastLineClusterNumber != -1)) {
                if ((lineClusterNumber == ((lastLineClusterNumber + 3) % 9)) && (lastLineClusterNumber != -1)) {
                    currentRow++;
                    if (static_cast<int>(votes.size()) < currentRow + 1) {
                        votes.resize(currentRow + 1);
                        votes[currentRow].resize(symbolsPerLine);
                    }
                }
                if ((lineClusterNumber == ((lastLineClusterNumber + 6) % 9)) && (lastLineClusterNumber != -1)) {
                    currentRow += 2;
                    if (static_cast<int>(votes.size()) < currentRow + 1) {
                        votes.resize(currentRow + 1);
                        votes[currentRow].resize(symbolsPerLine);
                    }
                }
                
                for (size_t i = 0; i < codewords[y].size(); i++) {
                    if (clusterNumbers[y][i] != -1)
                    {
                        if (clusterNumbers[y][i] == lineClusterNumber)
                        {
                            votes[currentRow][i][codewords[y][i]] = votes[currentRow][i][codewords[y][i]] + 1;
                        }
                        else if (clusterNumbers[y][i] == ((lineClusterNumber + 3) % 9))
                        {
                            if (static_cast<int>(votes.size()) < currentRow + 2)
                            {
                                votes.resize(currentRow + 2);
                                votes[currentRow + 1].resize(symbolsPerLine);
                            }
                            votes[currentRow + 1][i][codewords[y][i]] = votes[currentRow + 1][i][codewords[y][i]] + 1;
                        }
                        else if ((clusterNumbers[y][i] == ((lineClusterNumber + 6) % 9)) && (currentRow > 0))
                        {
                            votes[currentRow - 1][i][codewords[y][i]] = votes[currentRow - 1][i][codewords[y][i]] + 1;
                        }
                    }
                }
            
                lastLineClusterNumber = lineClusterNumber;
            }
        }
    }
    
    return votes;
}

vector<int>
LinesSampler::findMissingLines(const int symbolsPerLine, vector<vector<int> > &detectedCodeWords, int &iEmptyLine) {
    vector<int> insertLinesAt;
    if (detectedCodeWords.size() > 1) {
        for (size_t i = 0; i < detectedCodeWords.size() - 1; i++) {
            int clusterNumberRow = -1;
            for (size_t j = 0; j < detectedCodeWords[i].size() && clusterNumberRow == -1; j++) {
                int clusterNumber = calculateClusterNumber(detectedCodeWords[i][j]);
                if (clusterNumber != -1) {
                    clusterNumberRow = clusterNumber;
                }
            }
            if (i == 0) {
                // The first line must have the cluster number 0. Insert empty lines to match this.
                if (clusterNumberRow > 0) {
                    insertLinesAt.push_back(0);
                    if (clusterNumberRow > 3) {
                        insertLinesAt.push_back(0);
                    }
                }
            }
            int clusterNumberNextRow = -1;
            for (size_t j = 0; j < detectedCodeWords[i + 1].size() && clusterNumberNextRow == -1; j++) {
                int clusterNumber = calculateClusterNumber(detectedCodeWords[i + 1][j]);
                if (clusterNumber != -1) {
                    clusterNumberNextRow = clusterNumber;
                }
            }
            if ((clusterNumberRow + 3) % 9 != clusterNumberNextRow
                && clusterNumberRow != -1
                && clusterNumberNextRow != -1) {
                // The cluster numbers are not consecutive. Insert an empty line between them.
                insertLinesAt.push_back(i + 1);
                if (clusterNumberRow == clusterNumberNextRow) {
                    // There may be two lines missing. This is detected when two consecutive lines have the same cluster number.
                    insertLinesAt.push_back(i + 1);
                }
            }
        }
    }
    
    for (size_t i = 0; i < insertLinesAt.size(); i++) {
        detectedCodeWords.insert(detectedCodeWords.begin() + insertLinesAt[i] + i, vector<int>(symbolsPerLine, 0));
    }
    iEmptyLine += insertLinesAt.size();
    
    return insertLinesAt;
}

int LinesSampler::decodeRowCount(const int symbolsPerLine, vector<vector<int> > &detectedCodeWords, vector<int> &insertLinesAt, int &iEmptyLine)
{
    // Use the information in the first and last column to determin the number of rows and find more missing rows.
    // For missing rows insert blank space, so the error correction can try to fill them in.
    
    map<int, int> rowCountVotes;
    map<int, int> ecLevelVotes;
    map<int, int> rowNumberVotes;
    int lastRowNumber = -1;
    insertLinesAt.clear();
    
    for (size_t i = 0; i + 2 < detectedCodeWords.size(); i += 3) {
        rowNumberVotes.clear();
        int firstCodewordDecodedLeft = -1;
        int secondCodewordDecodedLeft = -1;
        int thirdCodewordDecodedLeft = -1;
        int firstCodewordDecodedRight = -1;
        int secondCodewordDecodedRight = -1;
        int thirdCodewordDecodedRight = -1;
        
        if (detectedCodeWords[i][0] != 0) {
            firstCodewordDecodedLeft = BitMatrixParser::getCodeword(detectedCodeWords[i][0]);
        }
        if (detectedCodeWords[i + 1][0] != 0) {
            secondCodewordDecodedLeft = BitMatrixParser::getCodeword(detectedCodeWords[i + 1][0]);
        }
        if (detectedCodeWords[i + 2][0] != 0) {
            thirdCodewordDecodedLeft = BitMatrixParser::getCodeword(detectedCodeWords[i + 2][0]);
        }
        
        if (detectedCodeWords[i][detectedCodeWords[i].size() - 1] != 0) {
            firstCodewordDecodedRight = BitMatrixParser::getCodeword(detectedCodeWords[i][detectedCodeWords[i].size() - 1]);
        }
        if (detectedCodeWords[i + 1][detectedCodeWords[i + 1].size() - 1] != 0) {
            secondCodewordDecodedRight = BitMatrixParser::getCodeword(detectedCodeWords[i + 1][detectedCodeWords[i + 1].size() - 1]);
        }
        if (detectedCodeWords[i + 2][detectedCodeWords[i + 2].size() - 1] != 0) {
            thirdCodewordDecodedRight = BitMatrixParser::getCodeword(detectedCodeWords[i + 2][detectedCodeWords[i + 2].size() - 1]);
        }
        
        if (firstCodewordDecodedLeft != -1 && secondCodewordDecodedLeft != -1) {
            int leftRowCount = ((firstCodewordDecodedLeft % 30) * 3) + ((secondCodewordDecodedLeft % 30) % 3);
            int leftECLevel = (secondCodewordDecodedLeft % 30) / 3;
            
            rowCountVotes[leftRowCount] = rowCountVotes[leftRowCount] + 1;
            ecLevelVotes[leftECLevel] = ecLevelVotes[leftECLevel] + 1;
        }
        
        if (secondCodewordDecodedRight != -1 && thirdCodewordDecodedRight != -1) {
            int rightRowCount = ((secondCodewordDecodedRight % 30) * 3) + ((thirdCodewordDecodedRight % 30) % 3);
            int rightECLevel = (thirdCodewordDecodedRight % 30) / 3;
            
            rowCountVotes[rightRowCount] = rowCountVotes[rightRowCount] + 1;
            ecLevelVotes[rightECLevel] = ecLevelVotes[rightECLevel] + 1;
        }
        
        if (firstCodewordDecodedLeft != -1) {
            int rowNumber = firstCodewordDecodedLeft / 30;
            rowNumberVotes[rowNumber] = rowNumberVotes[rowNumber] + 1;
        }
        if (secondCodewordDecodedLeft != -1) {
            int rowNumber = secondCodewordDecodedLeft / 30;
            rowNumberVotes[rowNumber] = rowNumberVotes[rowNumber] + 1;
        }
        if (thirdCodewordDecodedLeft != -1) {
            int rowNumber = thirdCodewordDecodedLeft / 30;
            rowNumberVotes[rowNumber] = rowNumberVotes[rowNumber] + 1;
        }
        if (firstCodewordDecodedRight != -1) {
            int rowNumber = firstCodewordDecodedRight / 30;
            rowNumberVotes[rowNumber] = rowNumberVotes[rowNumber] + 1;
        }
        if (secondCodewordDecodedRight != -1) {
            int rowNumber = secondCodewordDecodedRight / 30;
            rowNumberVotes[rowNumber] = rowNumberVotes[rowNumber] + 1;
        }
        if (thirdCodewordDecodedRight != -1) {
            int rowNumber = thirdCodewordDecodedRight / 30;
            rowNumberVotes[rowNumber] = rowNumberVotes[rowNumber] + 1;
        }
        int rowNumber = getValueWithMaxVotes(rowNumberVotes).getVote();
        if (lastRowNumber + 1 < rowNumber) {
            for (int j = lastRowNumber + 1; j < rowNumber; j++) {
                insertLinesAt.push_back(i);
                insertLinesAt.push_back(i);
                insertLinesAt.push_back(i);
            }
        }
        lastRowNumber = rowNumber;
    }
    
    for (size_t i = 0; i < insertLinesAt.size(); i++) {
        detectedCodeWords.insert(detectedCodeWords.begin() + insertLinesAt[i] + i, vector<int>(symbolsPerLine, 0));
    }
    iEmptyLine += insertLinesAt.size();
    
    int rowCount = getValueWithMaxVotes(rowCountVotes).getVote();
    
    rowCount += 1;
    return rowCount;
}

/**
 * Ends up being a bit faster than Math.round(). This merely rounds its
 * argument to the nearest int, where x.5 rounds up.
 */
int LinesSampler::round(float d)
{
    return static_cast<int>(d + 0.5f);
}

Point LinesSampler::intersection(Line a, Line b) {
    float dxa = a.start.x - a.end.x;
    float dxb = b.start.x - b.end.x;
    float dya = a.start.y - a.end.y;
    float dyb = b.start.y - b.end.y;
    
    float p = a.start.x * a.end.y - a.start.y * a.end.x;
    float q = b.start.x * b.end.y - b.start.y * b.end.x;
    float denom = dxa * dyb - dya * dxb;
    if (abs(denom) < 1e-12)  // Lines don't intersect (replaces "denom == 0")
        return Point(std::numeric_limits<float>::infinity(),
                     std::numeric_limits<float>::infinity());
    
    float x = (p * dxb - dxa * q) / denom;
    float y = (p * dyb - dya * q) / denom;
    
    return Point(x, y);
}
