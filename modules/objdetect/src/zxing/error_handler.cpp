// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Modified from ZXing. Copyright ZXing authors.
// Licensed under the Apache License, Version 2.0 (the "License").

// =====================================================================================
// 
//       Filename:  ErrorHandler.cpp
// 
//    Description:  
// 
//        Version:  1.0
//        Created:  07/03/2018 12:31:23 AM
//       Revision:  none
//       Compiler:  g++
// 
//         Author:  changoran (), changoran@tencent.com
//        Company:  
// 
// =====================================================================================

#include <stdio.h>

#include "error_handler.hpp"

using namespace zxing;

ErrorHandler::ErrorHandler() 
:err_code_(0), err_msg_("")
{
    init();
}

ErrorHandler::ErrorHandler(const char * err_msg) 
:err_code_(-1), err_msg_(err_msg)
{
    init();
}

ErrorHandler::ErrorHandler(std::string & err_msg) 
:err_code_(-1), err_msg_(err_msg)
{
    init();
}

ErrorHandler::ErrorHandler(int err_code) 
:err_code_(err_code), err_msg_("error")
{
    init();
}

ErrorHandler::ErrorHandler(int err_code, const char * err_msg) 
:err_code_(err_code), err_msg_(err_msg)
{
    init();
}

ErrorHandler::ErrorHandler(const ErrorHandler & other)
{
    err_code_ = other.errCode();
    err_msg_.assign(other.errMsg());
    init();
}

ErrorHandler& ErrorHandler::operator=(const ErrorHandler & other)
{
    err_code_ = other.errCode();
    err_msg_.assign(other.errMsg());
    init();
    return *this;
}

void ErrorHandler::init()
{
    handler_type_ = KErrorHandler;
}

void ErrorHandler::reset()
{
    err_code_ = 0;
    err_msg_.assign("");
}

void ErrorHandler::printInfo()
{
    printf("handler_tpye %d, error code %d, errmsg %s\n", handler_type_, err_code_, err_msg_.c_str());
}
