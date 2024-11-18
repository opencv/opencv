// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

/*
 *  BitMatrixParser.cpp
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

#include "bit_matrix_parser.hpp"
#include "../../common/illegal_argument_exception.hpp"

#include <iostream>

namespace zxing {
namespace datamatrix {

int BitMatrixParser::copyBit(size_t x, size_t y, int versionBits) {
    return bitMatrix_->get(x, y) ? (versionBits << 1) | 0x1 : versionBits << 1;
}

BitMatrixParser::BitMatrixParser(Ref<BitMatrix> bitMatrix, ErrorHandler & err_handler) : bitMatrix_(NULL),
parsedVersion_(NULL),
readBitMatrix_(NULL) {
    size_t dimension = bitMatrix->getHeight();
    if (dimension < 8 || dimension > 144 || (dimension & 0x01) != 0)
    {
        err_handler = ErrorHandler("Dimensio n must be even, > 8 < 144");
        return;
    }
    
    parsedVersion_ = readVersion(bitMatrix, err_handler);
    if (err_handler.errCode()) return;
    
    bitMatrix_ = extractDataRegion(bitMatrix, err_handler);
    if (err_handler.errCode()) return;
    
    readBitMatrix_ = new BitMatrix(bitMatrix_->getWidth(), bitMatrix_->getHeight() , err_handler);
    if (err_handler.errCode())   return;
}

Version * BitMatrixParser::readVersion(Ref<BitMatrix> bitMatrix, ErrorHandler & err_handler) {
    if (parsedVersion_ != 0)
    {
        return parsedVersion_;
    }
    
    int numRows = bitMatrix->getHeight();
    int numColumns = bitMatrix->getWidth();
    
    Version * version = parsedVersion_->getVersionForDimensions(numRows, numColumns, err_handler);
    
    if (err_handler.errCode())
    {
        err_handler = ErrorHandler("Couldn't decode versio");
        return NULL;
    }
    else
    {
        return version;
    }
}

ArrayRef<char> BitMatrixParser::readCodewords(ErrorHandler& err_handler) {
    ArrayRef<char> result(parsedVersion_->getTotalCodewords());
    int resultOffset = 0;
    int row = 4;
    int column = 0;
    
    int numRows = bitMatrix_->getHeight();
    int numColumns = bitMatrix_->getWidth();
    
    bool corner1Read = false;
    bool corner2Read = false;
    bool corner3Read = false;
    bool corner4Read = false;
    
    // Read all of the codewords
    do {
        // Check the four corner cases
        if ((row == numRows) && (column == 0) && !corner1Read)
        {
            result[resultOffset++] = static_cast<char>(readCorner1(numRows, numColumns));
            row -= 2;
            column +=2;
            corner1Read = true;
        }
        else if ((row == numRows-2) && (column == 0) && ((numColumns & 0x03) != 0) && !corner2Read)
        {
            result[resultOffset++] = static_cast<char>(readCorner2(numRows, numColumns));
            row -= 2;
            column +=2;
            corner2Read = true;
        }
        else if ((row == numRows+4) && (column == 2) && ((numColumns & 0x07) == 0) && !corner3Read)
        {
            result[resultOffset++] = static_cast<char>(readCorner3(numRows, numColumns));
            row -= 2;
            column +=2;
            corner3Read = true;
        }
        else if ((row == numRows-2) && (column == 0) && ((numColumns & 0x07) == 4) && !corner4Read)
        {
            result[resultOffset++] = static_cast<char>(readCorner4(numRows, numColumns));
            row -= 2;
            column +=2;
            corner4Read = true;
        }
        else
        {
            // Sweep upward diagonally to the right
            do {
                if ((row < numRows) && (column >= 0) && !readBitMatrix_->get(column, row))
                {
                    result[resultOffset++] = static_cast<char>(readUtah(row, column, numRows, numColumns));
                }
                row -= 2;
                column +=2;
            } while ((row >= 0) && (column < numColumns));
            row += 1;
            column +=3;
            
            // Sweep downward diagonally to the left
            do {
                if ((row >= 0) && (column < numColumns) && !readBitMatrix_->get(column, row))
                {
                    result[resultOffset++] = static_cast<char>(readUtah(row, column, numRows, numColumns));
                }
                row += 2;
                column -= 2;
            } while ((row < numRows) && (column >= 0));
            row += 3;
            column +=1;
        }
    } while ((row < numRows) || (column < numColumns));
    
    if (resultOffset != parsedVersion_->getTotalCodewords())
    {
        err_handler = zxing::ReaderErrorHandler("Did not read all codewords");
        return ArrayRef<char>();
    }
    return result;
}

bool BitMatrixParser::readModule(int row, int column, int numRows, int numColumns) {
    // Adjust the row and column indices based on boundary wrapping
    if (row < 0)
    {
        row += numRows;
        column += 4 - ((numRows + 4) & 0x07);
    }
    if (column < 0)
    {
        column += numColumns;
        row += 4 - ((numColumns + 4) & 0x07);
    }
    if (row >= numRows)
    {
        row -= numRows;
    }
    readBitMatrix_->set(column, row);
    return bitMatrix_->get(column, row);
}

int BitMatrixParser::readUtah(int row, int column, int numRows, int numColumns) {
    int currentByte = 0;
    if (readModule(row - 2, column - 2, numRows, numColumns))
    {
        currentByte |= 1;
    }
    currentByte <<= 1;
    if (readModule(row - 2, column - 1, numRows, numColumns))
    {
        currentByte |= 1;
    }
    currentByte <<= 1;
    if (readModule(row - 1, column - 2, numRows, numColumns))
    {
        currentByte |= 1;
    }
    currentByte <<= 1;
    if (readModule(row - 1, column - 1, numRows, numColumns))
    {
        currentByte |= 1;
    }
    currentByte <<= 1;
    if (readModule(row - 1, column, numRows, numColumns))
    {
        currentByte |= 1;
    }
    currentByte <<= 1;
    if (readModule(row, column - 2, numRows, numColumns))
    {
        currentByte |= 1;
    }
    currentByte <<= 1;
    if (readModule(row, column - 1, numRows, numColumns))
    {
        currentByte |= 1;
    }
    currentByte <<= 1;
    if (readModule(row, column, numRows, numColumns))
    {
        currentByte |= 1;
    }
    return currentByte;
}

int BitMatrixParser::readCorner1(int numRows, int numColumns) {
    int currentByte = 0;
    if (readModule(numRows - 1, 0, numRows, numColumns))
    {
        currentByte |= 1;
    }
    currentByte <<= 1;
    if (readModule(numRows - 1, 1, numRows, numColumns))
    {
        currentByte |= 1;
    }
    currentByte <<= 1;
    if (readModule(numRows - 1, 2, numRows, numColumns))
    {
        currentByte |= 1;
    }
    currentByte <<= 1;
    if (readModule(0, numColumns - 2, numRows, numColumns))
    {
        currentByte |= 1;
    }
    currentByte <<= 1;
    if (readModule(0, numColumns - 1, numRows, numColumns))
    {
        currentByte |= 1;
    }
    currentByte <<= 1;
    if (readModule(1, numColumns - 1, numRows, numColumns))
    {
        currentByte |= 1;
    }
    currentByte <<= 1;
    if (readModule(2, numColumns - 1, numRows, numColumns))
    {
        currentByte |= 1;
    }
    currentByte <<= 1;
    if (readModule(3, numColumns - 1, numRows, numColumns))
    {
        currentByte |= 1;
    }
    return currentByte;
}

int BitMatrixParser::readCorner2(int numRows, int numColumns)
{
    int currentByte = 0;
    if (readModule(numRows - 3, 0, numRows, numColumns))
    {
        currentByte |= 1;
    }
    currentByte <<= 1;
    if (readModule(numRows - 2, 0, numRows, numColumns))
    {
        currentByte |= 1;
    }
    currentByte <<= 1;
    if (readModule(numRows - 1, 0, numRows, numColumns))
    {
        currentByte |= 1;
    }
    currentByte <<= 1;
    if (readModule(0, numColumns - 4, numRows, numColumns))
    {
        currentByte |= 1;
    }
    currentByte <<= 1;
    if (readModule(0, numColumns - 3, numRows, numColumns))
    {
        currentByte |= 1;
    }
    currentByte <<= 1;
    if (readModule(0, numColumns - 2, numRows, numColumns))
    {
        currentByte |= 1;
    }
    currentByte <<= 1;
    if (readModule(0, numColumns - 1, numRows, numColumns))
    {
        currentByte |= 1;
    }
    currentByte <<= 1;
    if (readModule(1, numColumns - 1, numRows, numColumns))
    {
        currentByte |= 1;
    }
    return currentByte;
}

int BitMatrixParser::readCorner3(int numRows, int numColumns)
{
    int currentByte = 0;
    if (readModule(numRows - 1, 0, numRows, numColumns))
    {
        currentByte |= 1;
    }
    currentByte <<= 1;
    if (readModule(numRows - 1, numColumns - 1, numRows, numColumns))
    {
        currentByte |= 1;
    }
    currentByte <<= 1;
    if (readModule(0, numColumns - 3, numRows, numColumns))
    {
        currentByte |= 1;
    }
    currentByte <<= 1;
    if (readModule(0, numColumns - 2, numRows, numColumns))
    {
        currentByte |= 1;
    }
    currentByte <<= 1;
    if (readModule(0, numColumns - 1, numRows, numColumns))
    {
        currentByte |= 1;
    }
    currentByte <<= 1;
    if (readModule(1, numColumns - 3, numRows, numColumns))
    {
        currentByte |= 1;
    }
    currentByte <<= 1;
    if (readModule(1, numColumns - 2, numRows, numColumns))
    {
        currentByte |= 1;
    }
    currentByte <<= 1;
    if (readModule(1, numColumns - 1, numRows, numColumns))
    {
        currentByte |= 1;
    }
    return currentByte;
}

int BitMatrixParser::readCorner4(int numRows, int numColumns) {
    int currentByte = 0;
    if (readModule(numRows - 3, 0, numRows, numColumns))
    {
        currentByte |= 1;
    }
    currentByte <<= 1;
    if (readModule(numRows - 2, 0, numRows, numColumns))
    {
        currentByte |= 1;
    }
    currentByte <<= 1;
    if (readModule(numRows - 1, 0, numRows, numColumns))
    {
        currentByte |= 1;
    }
    currentByte <<= 1;
    if (readModule(0, numColumns - 2, numRows, numColumns))
    {
        currentByte |= 1;
    }
    currentByte <<= 1;
    if (readModule(0, numColumns - 1, numRows, numColumns))
    {
        currentByte |= 1;
    }
    currentByte <<= 1;
    if (readModule(1, numColumns - 1, numRows, numColumns))
    {
        currentByte |= 1;
    }
    currentByte <<= 1;
    if (readModule(2, numColumns - 1, numRows, numColumns))
    {
        currentByte |= 1;
    }
    currentByte <<= 1;
    if (readModule(3, numColumns - 1, numRows, numColumns))
    {
        currentByte |= 1;
    }
    return currentByte;
}

Ref<BitMatrix> BitMatrixParser::extractDataRegion(Ref<BitMatrix> bitMatrix, ErrorHandler & err_handler) {
    int symbolSizeRows = parsedVersion_->getSymbolSizeRows();
    int symbolSizeColumns = parsedVersion_->getSymbolSizeColumns();
    
    if (static_cast<int>(bitMatrix->getHeight()) != symbolSizeRows)
    {
        err_handler = ErrorHandler("Dimension of bitMatrix must match the version siz");
        return Ref<BitMatrix>();
    }
    
    int dataRegionSizeRows = parsedVersion_->getDataRegionSizeRows();
    int dataRegionSizeColumns = parsedVersion_->getDataRegionSizeColumns();
    
    int numDataRegionsRow = symbolSizeRows / dataRegionSizeRows;
    int numDataRegionsColumn = symbolSizeColumns / dataRegionSizeColumns;
    
    int sizeDataRegionRow = numDataRegionsRow * dataRegionSizeRows;
    int sizeDataRegionColumn = numDataRegionsColumn * dataRegionSizeColumns;
    
    Ref<BitMatrix> bitMatrixWithoutAlignment(new BitMatrix(sizeDataRegionColumn, sizeDataRegionRow, err_handler));
    if (err_handler.errCode())   return Ref<BitMatrix>();
    for (int dataRegionRow = 0; dataRegionRow < numDataRegionsRow; ++dataRegionRow)
    {
        int dataRegionRowOffset = dataRegionRow * dataRegionSizeRows;
        for (int dataRegionColumn = 0; dataRegionColumn < numDataRegionsColumn; ++dataRegionColumn) {
            int dataRegionColumnOffset = dataRegionColumn * dataRegionSizeColumns;
            for (int i = 0; i < dataRegionSizeRows; ++i) {
                int readRowOffset = dataRegionRow * (dataRegionSizeRows + 2) + 1 + i;
                int writeRowOffset = dataRegionRowOffset + i;
                for (int j = 0; j < dataRegionSizeColumns; ++j) {
                    int readColumnOffset = dataRegionColumn * (dataRegionSizeColumns + 2) + 1 + j;
                    if (bitMatrix->get(readColumnOffset, readRowOffset))
                    {
                        int writeColumnOffset = dataRegionColumnOffset + j;
                        bitMatrixWithoutAlignment->set(writeColumnOffset, writeRowOffset);
                    }
                }
            }
        }
    }
    return bitMatrixWithoutAlignment;
}

}  // namespace datamatrix
}  // namespace zxing
