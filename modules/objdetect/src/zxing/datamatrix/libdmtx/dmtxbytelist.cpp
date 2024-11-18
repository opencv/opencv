// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

//
//  dmtxbytelist.cpp
//  test_dm
//
//  Created by wechatcv on 2022/5/7.
//

#include "dmtxbytelist.hpp"

#include <string.h>

namespace dmtx {

DmtxByteList dmtxByteListBuild(DmtxByte *storage, int capacity)
{
    DmtxByteList list;
    
    list.b = storage;
    list.capacity = capacity;
    list.length = 0;
    
    return list;
}

unsigned int dmtxByteListInit(DmtxByteList *list, int length, DmtxByte value)
{
    if (length > list->capacity)
    {
        return DmtxFail;
    }
    else
    {
        list->length = length;
        memset(list->b, value, sizeof(DmtxByte) * list->capacity);
        return DmtxPass;
    }
}

unsigned int dmtxByteListCopy(DmtxByteList *dst, const DmtxByteList *src)
{
    int length;
    
    if (dst->capacity < src->length)
    {
        return DmtxFail; /* dst must be large enough to hold src data */
    }
    else
    {
        /* Copy as many bytes as dst can hold or src can provide (smaller of two) */
        length = (dst->capacity < src->capacity) ? dst->capacity : src->capacity;
        
        dst->length = src->length;
        memcpy(dst->b, src->b, sizeof(unsigned char) * length);
        return DmtxPass;
    }
}

unsigned int dmtxByteListPush(DmtxByteList *list, DmtxByte value)
{
    if (list->length >= list->capacity)
    {
        return DmtxFail;
    }
    else
    {
        list->b[list->length++] = value;
        return DmtxPass;
    }
}

DmtxByte dmtxByteListPop(DmtxByteList *list, unsigned int *passFail)
{
    *passFail = (list->length > 0) ? 1 : 0;
    
    return list->b[--(list->length)];
}

}  // namespace dmtx
