// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

#ifndef __ZXING_COMMON_INTEGER_HPP__
#define __ZXING_COMMON_INTEGER_HPP__

#include <iostream>

namespace zxing
{

class Integer
{
public:
	static int parseInt(Ref<String> strInteger)
	{
        int integer = parseInt(strInteger->getText());
		return integer;
	}
    
    static int parseInt(std::string strInteger)
    {
        int integer = 0;
        
        integer = atoi(strInteger.c_str());

        return integer;
    }
};

}

#endif // __ZXING_COMMON_INTEGER_HPP__
