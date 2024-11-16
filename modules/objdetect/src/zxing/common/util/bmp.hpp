//
//  BMP.hpp
//  QQView
//
//  Created by Tencent Research on 9/30/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

bool SaveBMP(const char* BMPfname, int nWidth, int nHeight, unsigned char* buffer);

bool LoadBMP(const char* BMPfname, int &nWidth, int &nHeight, unsigned char* buffer);