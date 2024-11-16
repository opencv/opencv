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
    Init();
}

ErrorHandler::ErrorHandler(const char * err_msg) 
:err_code_(-1), err_msg_(err_msg)
{
    Init();
}

ErrorHandler::ErrorHandler(std::string & err_msg) 
:err_code_(-1), err_msg_(err_msg)
{
    Init();
}

ErrorHandler::ErrorHandler(int err_code) 
:err_code_(err_code), err_msg_("error")
{
    Init();
}

ErrorHandler::ErrorHandler(int err_code, const char * err_msg) 
:err_code_(err_code), err_msg_(err_msg)
{
    Init();
}

ErrorHandler::ErrorHandler(const ErrorHandler & other)
{
    err_code_ = other.ErrCode();
    err_msg_.assign(other.ErrMsg());
    Init();
}

ErrorHandler& ErrorHandler::operator=(const ErrorHandler & other)
{
    err_code_ = other.ErrCode();
    err_msg_.assign(other.ErrMsg());
    Init();
    return *this;
}

void ErrorHandler::Init()
{
    handler_type_ = KErrorHandler;
}

void ErrorHandler::Reset()
{
    err_code_ = 0;
    err_msg_.assign("");
}

void ErrorHandler::PrintInfo()
{
    printf("handler_tpye %d, error code %d, errmsg %s\n", handler_type_, err_code_, err_msg_.c_str());
}
