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
//       Filename:  ErrorHandler.hpp
// 
//    Description:  
// 
//        Version:  1.0
//        Created:  07/02/2018 11:14:37 PM
//       Revision:  none
//       Compiler:  g++
// 
//         Author:  changoran (), changoran@tencent.com
//        Company:  
// 
// =====================================================================================

#ifndef __ZXING_ERROR_HANDLER_HPP__
#define __ZXING_ERROR_HANDLER_HPP__

#include <string>

namespace zxing {

enum{
    KErrorHandler = 0,
    KErrorHandler_NotFound = 1,
    KErrorHandler_CheckSum = 2,
    KErrorHandler_Reader = 3,
    KErrorHandler_IllegalArgument = 4,
    KErrorHandler_ReedSolomon = 5,
    KErrorHandler_Format = 6,
    KErrorHandler_Detector = 7,
    KErrorHandler_IllegalState = 8,
};

class ErrorHandler 
{
public:
    ErrorHandler();
    ErrorHandler(std::string & err_msg);
    ErrorHandler(const char * err_msg);
    ErrorHandler(int err_code);
    ErrorHandler(int err_code, std::string & err_msg);
    ErrorHandler(int err_code, const char * err_msg);
    
    virtual ~ErrorHandler() {};
    
    virtual inline int errCode() const {return err_code_;}
    virtual inline const std::string & errMsg() const  {return err_msg_;}
    virtual inline int handlerType() const {return handler_type_;}
    
    
    virtual void init();
    ErrorHandler(const ErrorHandler & other);
    ErrorHandler& operator=(const ErrorHandler & other);
    
    virtual void printInfo();
    virtual void reset();
    
protected:
    int handler_type_;
    
private:
    int err_code_;
    std::string err_msg_;
};


#define DECLARE_ERROR_HANDLER(__HANDLER__)  \
class __HANDLER__##ErrorHandler : public ErrorHandler{  \
public: \
__HANDLER__##ErrorHandler() : ErrorHandler() { init();};    \
__HANDLER__##ErrorHandler(std::string & err_msg) : ErrorHandler(err_msg) { init();};    \
__HANDLER__##ErrorHandler(const char * err_msg) : ErrorHandler(err_msg) { init();}; \
__HANDLER__##ErrorHandler(int err_code) : ErrorHandler(err_code) { init();};    \
__HANDLER__##ErrorHandler(int err_code, std::string & err_msg) : ErrorHandler(err_code, err_msg) { init();};    \
__HANDLER__##ErrorHandler(int err_code, const char * err_msg) : ErrorHandler(err_code, err_msg) {init();}; \
__HANDLER__##ErrorHandler(const ErrorHandler & other) : ErrorHandler(other) { init();}; \
void init(){    \
handler_type_ = KErrorHandler_##__HANDLER__;    \
}   \
};

DECLARE_ERROR_HANDLER(Reader)
DECLARE_ERROR_HANDLER(IllegalArgument)
DECLARE_ERROR_HANDLER(ReedSolomon)
DECLARE_ERROR_HANDLER(Format)
DECLARE_ERROR_HANDLER(Detector)
DECLARE_ERROR_HANDLER(NotFound)
DECLARE_ERROR_HANDLER(CheckSum)
DECLARE_ERROR_HANDLER(IllegalState)

#undef DECLARE_ERROR_HANDLER
}   // namespace zxing


#endif  // __ZXING_ERROR_HANDLER_HPP__
