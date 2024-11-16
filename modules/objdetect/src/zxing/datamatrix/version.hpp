#ifndef __VERSION_H_DM__
#define __VERSION_H_DM__

/*
 *  Version.hpp
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

#include "../reader_exception.hpp"
#include "../common/bit_matrix.hpp"
#include "../common/counted.hpp"
#include "../error_handler.hpp"
#include <vector>

namespace zxing {
namespace datamatrix {

class ECB {
private:
    int count_;
    int dataCodewords_;
public:
    ECB(int count, int dataCodewords);
    int getCount();
    int getDataCodewords();
};

class ECBlocks {
private:
    int ecCodewords_;
    std::vector<ECB*> ecBlocks_;
public:
    ECBlocks(int ecCodewords, ECB *ecBlocks);
    ECBlocks(int ecCodewords, ECB *ecBlocks1, ECB *ecBlocks2);
    int getECCodewords();
    std::vector<ECB*>& getECBlocks();
    ~ECBlocks();
};

class Version : public Counted {
private:
    int versionNumber_;
    int symbolSizeRows_;
    int symbolSizeColumns_;
    int dataRegionSizeRows_;
    int dataRegionSizeColumns_;
    ECBlocks* ecBlocks_;
    int totalCodewords_;
    Version(int versionNumber, int symbolSizeRows, int symbolSizeColumns, int dataRegionSizeRows,
            int dataRegionSizeColumns, ECBlocks *ecBlocks);
    
public:
    static std::vector<Ref<Version> > VERSIONS;
    
    ~Version();
    int getVersionNumber();
    int getSymbolSizeRows();
    int getSymbolSizeColumns();
    int getDataRegionSizeRows();
    int getDataRegionSizeColumns();
    int getTotalCodewords();
    ECBlocks* getECBlocks();
    static int  buildVersions();
    Version * getVersionForDimensions(int numRows, int numColumns, ErrorHandler & err_handler);
    
private:
    Version(const Version&);
    Version & operator=(const Version&);
};
}  // namespace datamatrix
}  // namespace zxing

#endif  // __VERSION_H__
