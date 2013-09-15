/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

namespace cv
{

template<typename _KeyTp, typename _ValueTp> struct sorted_vector
{
    sorted_vector() {}
    void clear() { vec.clear(); }
    size_t size() const { return vec.size(); }
    _ValueTp& operator [](size_t idx) { return vec[idx]; }
    const _ValueTp& operator [](size_t idx) const { return vec[idx]; }

    void add(const _KeyTp& k, const _ValueTp& val)
    {
        std::pair<_KeyTp, _ValueTp> p(k, val);
        vec.push_back(p);
        size_t i = vec.size()-1;
        for( ; i > 0 && vec[i].first < vec[i-1].first; i-- )
            std::swap(vec[i-1], vec[i]);
        CV_Assert( i == 0 || vec[i].first != vec[i-1].first );
    }

    bool find(const _KeyTp& key, _ValueTp& value) const
    {
        size_t a = 0, b = vec.size();
        while( b > a )
        {
            size_t c = (a + b)/2;
            if( vec[c].first < key )
                a = c+1;
            else
                b = c;
        }

        if( a < vec.size() && vec[a].first == key )
        {
            value = vec[a].second;
            return true;
        }
        return false;
    }

    void get_keys(std::vector<_KeyTp>& keys) const
    {
        size_t i = 0, n = vec.size();
        keys.resize(n);

        for( i = 0; i < n; i++ )
            keys[i] = vec[i].first;
    }

    std::vector<std::pair<_KeyTp, _ValueTp> > vec;
};


template<typename _ValueTp> inline const _ValueTp* findstr(const sorted_vector<String, _ValueTp>& vec,
                                                           const char* key)
{
    if( !key )
        return 0;

    size_t a = 0, b = vec.vec.size();
    while( b > a )
    {
        size_t c = (a + b)/2;
        if( strcmp(vec.vec[c].first.c_str(), key) < 0 )
            a = c+1;
        else
            b = c;
    }

    if( ( a < vec.vec.size() ) && ( strcmp(vec.vec[a].first.c_str(), key) == 0 ))
        return &vec.vec[a].second;
    return 0;
}


Param::Param()
{
    type = 0;
    offset = 0;
    readonly = false;
    getter = 0;
    setter = 0;
}


Param::Param(int _type, bool _readonly, int _offset,
             Algorithm::Getter _getter, Algorithm::Setter _setter,
             const String& _help)
{
    type = _type;
    readonly = _readonly;
    offset = _offset;
    getter = _getter;
    setter = _setter;
    help = _help;
}

struct CV_EXPORTS AlgorithmInfoData
{
    sorted_vector<String, Param> params;
    String _name;
};


static sorted_vector<String, Algorithm::Constructor>& alglist()
{
    static sorted_vector<String, Algorithm::Constructor> alglist_var;
    return alglist_var;
}

void Algorithm::getList(std::vector<String>& algorithms)
{
    alglist().get_keys(algorithms);
}

Ptr<Algorithm> Algorithm::_create(const String& name)
{
    Algorithm::Constructor c = 0;
    if( !alglist().find(name, c) )
        return Ptr<Algorithm>();
    return Ptr<Algorithm>(c());
}

Algorithm::Algorithm()
{
}

Algorithm::~Algorithm()
{
}

String Algorithm::name() const
{
    return info()->name();
}

void Algorithm::set(const String& parameter, int value)
{
    info()->set(this, parameter.c_str(), ParamType<int>::type, &value);
}

void Algorithm::set(const String& parameter, double value)
{
    info()->set(this, parameter.c_str(), ParamType<double>::type, &value);
}

void Algorithm::set(const String& parameter, bool value)
{
    info()->set(this, parameter.c_str(), ParamType<bool>::type, &value);
}

void Algorithm::set(const String& parameter, const String& value)
{
    info()->set(this, parameter.c_str(), ParamType<String>::type, &value);
}

void Algorithm::set(const String& parameter, const Mat& value)
{
    info()->set(this, parameter.c_str(), ParamType<Mat>::type, &value);
}

void Algorithm::set(const String& parameter, const std::vector<Mat>& value)
{
    info()->set(this, parameter.c_str(), ParamType<std::vector<Mat> >::type, &value);
}

void Algorithm::set(const String& parameter, const Ptr<Algorithm>& value)
{
    info()->set(this, parameter.c_str(), ParamType<Algorithm>::type, &value);
}

void Algorithm::set(const char* parameter, int value)
{
    info()->set(this, parameter, ParamType<int>::type, &value);
}

void Algorithm::set(const char* parameter, double value)
{
    info()->set(this, parameter, ParamType<double>::type, &value);
}

void Algorithm::set(const char* parameter, bool value)
{
    info()->set(this, parameter, ParamType<bool>::type, &value);
}

void Algorithm::set(const char* parameter, const String& value)
{
    info()->set(this, parameter, ParamType<String>::type, &value);
}

void Algorithm::set(const char* parameter, const Mat& value)
{
    info()->set(this, parameter, ParamType<Mat>::type, &value);
}

void Algorithm::set(const char* parameter, const std::vector<Mat>& value)
{
    info()->set(this, parameter, ParamType<std::vector<Mat> >::type, &value);
}

void Algorithm::set(const char* parameter, const Ptr<Algorithm>& value)
{
    info()->set(this, parameter, ParamType<Algorithm>::type, &value);
}


void Algorithm::setInt(const String& parameter, int value)
{
    info()->set(this, parameter.c_str(), ParamType<int>::type, &value);
}

void Algorithm::setDouble(const String& parameter, double value)
{
    info()->set(this, parameter.c_str(), ParamType<double>::type, &value);
}

void Algorithm::setBool(const String& parameter, bool value)
{
    info()->set(this, parameter.c_str(), ParamType<bool>::type, &value);
}

void Algorithm::setString(const String& parameter, const String& value)
{
    info()->set(this, parameter.c_str(), ParamType<String>::type, &value);
}

void Algorithm::setMat(const String& parameter, const Mat& value)
{
    info()->set(this, parameter.c_str(), ParamType<Mat>::type, &value);
}

void Algorithm::setMatVector(const String& parameter, const std::vector<Mat>& value)
{
    info()->set(this, parameter.c_str(), ParamType<std::vector<Mat> >::type, &value);
}

void Algorithm::setAlgorithm(const String& parameter, const Ptr<Algorithm>& value)
{
    info()->set(this, parameter.c_str(), ParamType<Algorithm>::type, &value);
}

void Algorithm::setInt(const char* parameter, int value)
{
    info()->set(this, parameter, ParamType<int>::type, &value);
}

void Algorithm::setDouble(const char* parameter, double value)
{
    info()->set(this, parameter, ParamType<double>::type, &value);
}

void Algorithm::setBool(const char* parameter, bool value)
{
    info()->set(this, parameter, ParamType<bool>::type, &value);
}

void Algorithm::setString(const char* parameter, const String& value)
{
    info()->set(this, parameter, ParamType<String>::type, &value);
}

void Algorithm::setMat(const char* parameter, const Mat& value)
{
    info()->set(this, parameter, ParamType<Mat>::type, &value);
}

void Algorithm::setMatVector(const char* parameter, const std::vector<Mat>& value)
{
    info()->set(this, parameter, ParamType<std::vector<Mat> >::type, &value);
}

void Algorithm::setAlgorithm(const char* parameter, const Ptr<Algorithm>& value)
{
    info()->set(this, parameter, ParamType<Algorithm>::type, &value);
}



int Algorithm::getInt(const String& parameter) const
{
    return get<int>(parameter);
}

double Algorithm::getDouble(const String& parameter) const
{
    return get<double>(parameter);
}

bool Algorithm::getBool(const String& parameter) const
{
    return get<bool>(parameter);
}

String Algorithm::getString(const String& parameter) const
{
    return get<String>(parameter);
}

Mat Algorithm::getMat(const String& parameter) const
{
    return get<Mat>(parameter);
}

std::vector<Mat> Algorithm::getMatVector(const String& parameter) const
{
    return get<std::vector<Mat> >(parameter);
}

Ptr<Algorithm> Algorithm::getAlgorithm(const String& parameter) const
{
    return get<Algorithm>(parameter);
}

String Algorithm::paramHelp(const String& parameter) const
{
    return info()->paramHelp(parameter.c_str());
}

int Algorithm::paramType(const String& parameter) const
{
    return info()->paramType(parameter.c_str());
}

int Algorithm::paramType(const char* parameter) const
{
    return info()->paramType(parameter);
}

void Algorithm::getParams(std::vector<String>& names) const
{
    info()->getParams(names);
}

void Algorithm::write(FileStorage& fs) const
{
    info()->write(this, fs);
}

void Algorithm::read(const FileNode& fn)
{
    info()->read(this, fn);
}


AlgorithmInfo::AlgorithmInfo(const String& _name, Algorithm::Constructor create)
{
    data = new AlgorithmInfoData;
    data->_name = _name;
    if (!alglist().find(_name, create))
        alglist().add(_name, create);
}

AlgorithmInfo::~AlgorithmInfo()
{
    delete data;
}

void AlgorithmInfo::write(const Algorithm* algo, FileStorage& fs) const
{
    size_t i = 0, nparams = data->params.vec.size();
    cv::write(fs, "name", algo->name());
    for( i = 0; i < nparams; i++ )
    {
        const Param& p = data->params.vec[i].second;
        const String& pname = data->params.vec[i].first;
        if( p.type == Param::INT )
            cv::write(fs, pname, algo->get<int>(pname));
        else if( p.type == Param::BOOLEAN )
            cv::write(fs, pname, (int)algo->get<bool>(pname));
        else if( p.type == Param::REAL )
            cv::write(fs, pname, algo->get<double>(pname));
        else if( p.type == Param::STRING )
            cv::write(fs, pname, algo->get<String>(pname));
        else if( p.type == Param::MAT )
            cv::write(fs, pname, algo->get<Mat>(pname));
        else if( p.type == Param::MAT_VECTOR )
            cv::write(fs, pname, algo->get<std::vector<Mat> >(pname));
        else if( p.type == Param::ALGORITHM )
        {
            internal::WriteStructContext ws(fs, pname, CV_NODE_MAP);
            Ptr<Algorithm> nestedAlgo = algo->get<Algorithm>(pname);
            nestedAlgo->write(fs);
        }
        else if( p.type == Param::FLOAT)
            cv::write(fs, pname, algo->getDouble(pname));
        else if( p.type == Param::UNSIGNED_INT)
            cv::write(fs, pname, algo->getInt(pname));//TODO: implement cv::write(, , unsigned int)
        else if( p.type == Param::UINT64)
            cv::write(fs, pname, algo->getInt(pname));//TODO: implement cv::write(, , uint64)
        else if( p.type == Param::UCHAR)
            cv::write(fs, pname, algo->getInt(pname));
        else
        {
            String msg = format("unknown/unsupported type of '%s' parameter == %d", pname.c_str(), p.type);
            CV_Error( CV_StsUnsupportedFormat, msg.c_str());
        }
    }
}

void AlgorithmInfo::read(Algorithm* algo, const FileNode& fn) const
{
    size_t i = 0, nparams = data->params.vec.size();
    AlgorithmInfo* info = algo->info();

    for( i = 0; i < nparams; i++ )
    {
        const Param& p = data->params.vec[i].second;
        const String& pname = data->params.vec[i].first;
        const FileNode n = fn[pname];
        if( n.empty() )
            continue;
        if( p.type == Param::INT )
        {
            int val = (int)n;
            info->set(algo, pname.c_str(), p.type, &val, true);
        }
        else if( p.type == Param::BOOLEAN )
        {
            bool val = (int)n != 0;
            info->set(algo, pname.c_str(), p.type, &val, true);
        }
        else if( p.type == Param::REAL )
        {
            double val = (double)n;
            info->set(algo, pname.c_str(), p.type, &val, true);
        }
        else if( p.type == Param::STRING )
        {
            String val = (String)n;
            info->set(algo, pname.c_str(), p.type, &val, true);
        }
        else if( p.type == Param::MAT )
        {
            Mat m;
            cv::read(n, m);
            info->set(algo, pname.c_str(), p.type, &m, true);
        }
        else if( p.type == Param::MAT_VECTOR )
        {
            std::vector<Mat> mv;
            cv::read(n, mv);
            info->set(algo, pname.c_str(), p.type, &mv, true);
        }
        else if( p.type == Param::ALGORITHM )
        {
            Ptr<Algorithm> nestedAlgo = Algorithm::_create((String)n["name"]);
            CV_Assert( nestedAlgo );
            nestedAlgo->read(n);
            info->set(algo, pname.c_str(), p.type, &nestedAlgo, true);
        }
        else if( p.type == Param::FLOAT )
        {
            float val = (float)n;
            info->set(algo, pname.c_str(), p.type, &val, true);
        }
        else if( p.type == Param::UNSIGNED_INT )
        {
            unsigned int val = (unsigned int)((int)n);//TODO: implement conversion (unsigned int)FileNode
            info->set(algo, pname.c_str(), p.type, &val, true);
        }
        else if( p.type == Param::UINT64)
        {
            uint64 val = (uint64)((int)n);//TODO: implement conversion (uint64)FileNode
            info->set(algo, pname.c_str(), p.type, &val, true);
        }
        else if( p.type == Param::UCHAR)
        {
            uchar val = (uchar)((int)n);
            info->set(algo, pname.c_str(), p.type, &val, true);
        }
        else
        {
            String msg = format("unknown/unsupported type of '%s' parameter == %d", pname.c_str(), p.type);
            CV_Error( CV_StsUnsupportedFormat, msg.c_str());
        }
    }
}

String AlgorithmInfo::name() const
{
    return data->_name;
}

union GetSetParam
{
    int (Algorithm::*get_int)() const;
    bool (Algorithm::*get_bool)() const;
    double (Algorithm::*get_double)() const;
    String (Algorithm::*get_string)() const;
    Mat (Algorithm::*get_mat)() const;
    std::vector<Mat> (Algorithm::*get_mat_vector)() const;
    Ptr<Algorithm> (Algorithm::*get_algo)() const;
    float (Algorithm::*get_float)() const;
    unsigned int (Algorithm::*get_uint)() const;
    uint64 (Algorithm::*get_uint64)() const;
    uchar (Algorithm::*get_uchar)() const;

    void (Algorithm::*set_int)(int);
    void (Algorithm::*set_bool)(bool);
    void (Algorithm::*set_double)(double);
    void (Algorithm::*set_string)(const String&);
    void (Algorithm::*set_mat)(const Mat&);
    void (Algorithm::*set_mat_vector)(const std::vector<Mat>&);
    void (Algorithm::*set_algo)(const Ptr<Algorithm>&);
    void (Algorithm::*set_float)(float);
    void (Algorithm::*set_uint)(unsigned int);
    void (Algorithm::*set_uint64)(uint64);
    void (Algorithm::*set_uchar)(uchar);
};

static String getNameOfType(int argType);

static String getNameOfType(int argType)
{
    switch(argType)
    {
        case Param::INT: return "integer";
        case Param::BOOLEAN: return "boolean";
        case Param::REAL: return "double";
        case Param::STRING: return "string";
        case Param::MAT: return "cv::Mat";
        case Param::MAT_VECTOR: return "std::vector<cv::Mat>";
        case Param::ALGORITHM: return "algorithm";
        case Param::FLOAT: return "float";
        case Param::UNSIGNED_INT: return "unsigned int";
        case Param::UINT64: return "unsigned int64";
        case Param::UCHAR: return "unsigned char";
        default: CV_Error(CV_StsBadArg, "Wrong argument type");
    }
    return "";
}

static String getErrorMessageForWrongArgumentInSetter(String algoName, String paramName, int paramType, int argType)
{
    String message = String("Argument error: the setter")
        + " method was called for the parameter '" + paramName + "' of the algorithm '" + algoName
        +"', the parameter has " + getNameOfType(paramType) + " type, ";

    if (paramType == Param::INT || paramType == Param::BOOLEAN || paramType == Param::REAL
            || paramType == Param::FLOAT || paramType == Param::UNSIGNED_INT || paramType == Param::UINT64 || paramType == Param::UCHAR)
    {
        message = message + "so it should be set by integer, unsigned integer, uint64, unsigned char, boolean, float or double value, ";
    }
    message = message + "but the setter was called with " + getNameOfType(argType) + " value";

    return message;
}

static String getErrorMessageForWrongArgumentInGetter(String algoName, String paramName, int paramType, int argType)
{
    String message = String("Argument error: the getter")
        + " method was called for the parameter '" + paramName + "' of the algorithm '" + algoName
        +"', the parameter has " + getNameOfType(paramType) + " type, ";

    if (paramType == Param::BOOLEAN)
    {
        message = message + "so it should be get as integer, unsigned integer, uint64, boolean, unsigned char, float or double value, ";
    }
    else if (paramType == Param::INT || paramType == Param::UNSIGNED_INT || paramType == Param::UINT64 || paramType == Param::UCHAR)
    {
        message = message + "so it should be get as integer, unsigned integer, uint64, unsigned char, float or double value, ";
    }
    message = message + "but the getter was called to get a " + getNameOfType(argType) + " value";

    return message;
}

void AlgorithmInfo::set(Algorithm* algo, const char* parameter, int argType, const void* value, bool force) const
{
    const Param* p = findstr(data->params, parameter);

    if( !p )
        CV_Error_( CV_StsBadArg, ("No parameter '%s' is found", parameter ? parameter : "<NULL>") );

    if( !force && p->readonly )
        CV_Error_( CV_StsError, ("Parameter '%s' is readonly", parameter));

    GetSetParam f;
    f.set_int = p->setter;

    if( argType == Param::INT || argType == Param::BOOLEAN || argType == Param::REAL
            || argType == Param::FLOAT || argType == Param::UNSIGNED_INT || argType == Param::UINT64 || argType == Param::UCHAR)
    {
        if ( !( p->type == Param::INT || p->type == Param::REAL || p->type == Param::BOOLEAN
                || p->type == Param::UNSIGNED_INT || p->type == Param::UINT64 || p->type == Param::FLOAT || argType == Param::UCHAR) )
        {
            String message = getErrorMessageForWrongArgumentInSetter(algo->name(), parameter, p->type, argType);
            CV_Error(CV_StsBadArg, message);
        }

        if( p->type == Param::INT )
        {
            bool is_ok = true;
            int val = argType == Param::INT ? *(const int*)value :
            argType == Param::BOOLEAN ? (int)*(const bool*)value :
                argType == Param::REAL ? saturate_cast<int>(*(const double*)value) :
                argType == Param::FLOAT ?  saturate_cast<int>(*(const float*)value) :
                argType == Param::UNSIGNED_INT ? (int)*(const unsigned int*)value :
                argType == Param::UINT64 ? (int)*(const uint64*)value :
                argType == Param::UCHAR ? (int)*(const uchar*)value :
                (int)(is_ok = false);

            if (!is_ok)
            {
                CV_Error(CV_StsBadArg, "Wrong argument type in the setter");
            }

            if( p->setter )
                (algo->*f.set_int)(val);
            else
                *(int*)((uchar*)algo + p->offset) = val;
        }
        else if( p->type == Param::BOOLEAN )
        {
            bool is_ok = true;
            bool val = argType == Param::INT ? *(const int*)value != 0 :
                    argType == Param::BOOLEAN ? *(const bool*)value :
                    argType == Param::REAL ? (*(const double*)value != 0) :
                    argType == Param::FLOAT ?  (*(const float*)value != 0) :
                    argType == Param::UNSIGNED_INT ? (*(const unsigned int*)value != 0):
                    argType == Param::UINT64 ? (*(const uint64*)value != 0):
                    argType == Param::UCHAR ? (*(const uchar*)value != 0):
                    (int)(is_ok = false);

            if (!is_ok)
            {
                CV_Error(CV_StsBadArg, "Wrong argument type in the setter");
            }

            if( p->setter )
                (algo->*f.set_bool)(val);
            else
                *(bool*)((uchar*)algo + p->offset) = val;
        }
        else if( p->type == Param::REAL )
        {
            bool is_ok = true;
            double val = argType == Param::INT ? (double)*(const int*)value :
                         argType == Param::BOOLEAN ? (double)*(const bool*)value :
                         argType == Param::REAL ? (double)(*(const double*)value ) :
                         argType == Param::FLOAT ?  (double)(*(const float*)value ) :
                         argType == Param::UNSIGNED_INT ? (double)(*(const unsigned int*)value ) :
                         argType == Param::UINT64 ? (double)(*(const uint64*)value ) :
                         argType == Param::UCHAR ? (double)(*(const uchar*)value ) :
                         (double)(is_ok = false);

            if (!is_ok)
            {
                CV_Error(CV_StsBadArg, "Wrong argument type in the setter");
            }
            if( p->setter )
                (algo->*f.set_double)(val);
            else
                *(double*)((uchar*)algo + p->offset) = val;
        }
        else if( p->type == Param::FLOAT )
        {
            bool is_ok = true;
            double val = argType == Param::INT ? (double)*(const int*)value :
                         argType == Param::BOOLEAN ? (double)*(const bool*)value :
                         argType == Param::REAL ? (double)(*(const double*)value ) :
                         argType == Param::FLOAT ?  (double)(*(const float*)value ) :
                         argType == Param::UNSIGNED_INT ? (double)(*(const unsigned int*)value ) :
                         argType == Param::UINT64 ? (double)(*(const uint64*)value ) :
                         argType == Param::UCHAR ? (double)(*(const uchar*)value ) :
                         (double)(is_ok = false);

            if (!is_ok)
            {
                CV_Error(CV_StsBadArg, "Wrong argument type in the setter");
            }
            if( p->setter )
                (algo->*f.set_float)((float)val);
            else
                *(float*)((uchar*)algo + p->offset) = (float)val;
        }
        else if( p->type == Param::UNSIGNED_INT )
        {
            bool is_ok = true;
            unsigned int val = argType == Param::INT ? (unsigned int)*(const int*)value :
                         argType == Param::BOOLEAN ? (unsigned int)*(const bool*)value :
                         argType == Param::REAL ? saturate_cast<unsigned int>(*(const double*)value ) :
                         argType == Param::FLOAT ?  saturate_cast<unsigned int>(*(const float*)value ) :
                         argType == Param::UNSIGNED_INT ? (unsigned int)(*(const unsigned int*)value ) :
                         argType == Param::UINT64 ? (unsigned int)(*(const uint64*)value ) :
                         argType == Param::UCHAR ? (unsigned int)(*(const uchar*)value ) :
                         (int)(is_ok = false);

            if (!is_ok)
            {
                CV_Error(CV_StsBadArg, "Wrong argument type in the setter");
            }
            if( p->setter )
                (algo->*f.set_uint)(val);
            else
                *(unsigned int*)((uchar*)algo + p->offset) = val;
        }
        else if( p->type == Param::UINT64 )
        {
            bool is_ok = true;
            uint64 val = argType == Param::INT ? (uint64)*(const int*)value :
                         argType == Param::BOOLEAN ? (uint64)*(const bool*)value :
                         argType == Param::REAL ? saturate_cast<uint64>(*(const double*)value ) :
                         argType == Param::FLOAT ?  saturate_cast<uint64>(*(const float*)value ) :
                         argType == Param::UNSIGNED_INT ? (uint64)(*(const unsigned int*)value ) :
                         argType == Param::UINT64 ? (uint64)(*(const uint64*)value ) :
                         argType == Param::UCHAR ? (uint64)(*(const uchar*)value ) :
                         (int)(is_ok = false);

            if (!is_ok)
            {
                CV_Error(CV_StsBadArg, "Wrong argument type in the setter");
            }
            if( p->setter )
                (algo->*f.set_uint64)(val);
            else
                *(uint64*)((uchar*)algo + p->offset) = val;
        }
        else if( p->type == Param::UCHAR )
        {
            bool is_ok = true;
            uchar val = argType == Param::INT ? (uchar)*(const int*)value :
                         argType == Param::BOOLEAN ? (uchar)*(const bool*)value :
                         argType == Param::REAL ? saturate_cast<uchar>(*(const double*)value ) :
                         argType == Param::FLOAT ?  saturate_cast<uchar>(*(const float*)value ) :
                         argType == Param::UNSIGNED_INT ? (uchar)(*(const unsigned int*)value ) :
                         argType == Param::UINT64 ? (uchar)(*(const uint64*)value ) :
                         argType == Param::UCHAR ? (uchar)(*(const uchar*)value ) :
                         (int)(is_ok = false);

            if (!is_ok)
            {
                CV_Error(CV_StsBadArg, "Wrong argument type in the setter");
            }
            if( p->setter )
                (algo->*f.set_uchar)(val);
            else
                *(uchar*)((uchar*)algo + p->offset) = val;
        }
        else
            CV_Error(CV_StsBadArg, "Wrong parameter type in the setter");
    }
    else if( argType == Param::STRING )
    {
        if( p->type != Param::STRING )
        {
            String message = getErrorMessageForWrongArgumentInSetter(algo->name(), parameter, p->type, argType);
            CV_Error(CV_StsBadArg, message);
        }

        const String& val = *(const String*)value;
        if( p->setter )
            (algo->*f.set_string)(val);
        else
            *(String*)((uchar*)algo + p->offset) = val;
    }
    else if( argType == Param::MAT )
    {
        if( p->type != Param::MAT )
        {
            String message = getErrorMessageForWrongArgumentInSetter(algo->name(), parameter, p->type, argType);
            CV_Error(CV_StsBadArg, message);
        }

        const Mat& val = *(const Mat*)value;
        if( p->setter )
            (algo->*f.set_mat)(val);
        else
            *(Mat*)((uchar*)algo + p->offset) = val;
    }
    else if( argType == Param::MAT_VECTOR )
    {
        if( p->type != Param::MAT_VECTOR )
        {
            String message = getErrorMessageForWrongArgumentInSetter(algo->name(), parameter, p->type, argType);
            CV_Error(CV_StsBadArg, message);
        }

        const std::vector<Mat>& val = *(const std::vector<Mat>*)value;
        if( p->setter )
            (algo->*f.set_mat_vector)(val);
        else
            *(std::vector<Mat>*)((uchar*)algo + p->offset) = val;
    }
    else if( argType == Param::ALGORITHM )
    {
        if( p->type != Param::ALGORITHM )
        {
            String message = getErrorMessageForWrongArgumentInSetter(algo->name(), parameter, p->type, argType);
            CV_Error(CV_StsBadArg, message);
        }

        const Ptr<Algorithm>& val = *(const Ptr<Algorithm>*)value;
        if( p->setter )
            (algo->*f.set_algo)(val);
        else
            *(Ptr<Algorithm>*)((uchar*)algo + p->offset) = val;
    }
    else
        CV_Error(CV_StsBadArg, "Unknown/unsupported parameter type");
}

void AlgorithmInfo::get(const Algorithm* algo, const char* parameter, int argType, void* value) const
{
    const Param* p = findstr(data->params, parameter);
    if( !p )
        CV_Error_( CV_StsBadArg, ("No parameter '%s' is found", parameter ? parameter : "<NULL>") );

    GetSetParam f;
    f.get_int = p->getter;

    if( argType == Param::INT || argType == Param::BOOLEAN || argType == Param::REAL
            || argType == Param::FLOAT || argType == Param::UNSIGNED_INT || argType == Param::UINT64 || argType == Param::UCHAR)
    {
        if( p->type == Param::INT )
        {
            if (!( argType == Param::INT || argType == Param::REAL || argType == Param::FLOAT || argType == Param::UNSIGNED_INT || argType == Param::UINT64 || argType == Param::UCHAR))
            {
                String message = getErrorMessageForWrongArgumentInGetter(algo->name(), parameter, p->type, argType);
                CV_Error(CV_StsBadArg, message);
            }
            int val = p->getter ? (algo->*f.get_int)() : *(int*)((uchar*)algo + p->offset);

            if( argType == Param::INT )
                *(int*)value = (int)val;
            else if ( argType == Param::REAL )
                *(double*)value = (double)val;
            else if ( argType == Param::FLOAT)
                *(float*)value = (float)val;
            else if ( argType == Param::UNSIGNED_INT )
                *(unsigned int*)value = (unsigned int)val;
            else if ( argType == Param::UINT64 )
                *(uint64*)value = (uint64)val;
            else if ( argType == Param::UCHAR)
                *(uchar*)value = (uchar)val;
            else
                CV_Error(CV_StsBadArg, "Wrong argument type");

        }
        else if( p->type == Param::BOOLEAN )
        {
            if (!( argType == Param::INT || argType == Param::BOOLEAN || argType == Param::REAL || argType == Param::FLOAT || argType == Param::UNSIGNED_INT || argType == Param::UINT64 || argType == Param::UCHAR))
            {
                String message = getErrorMessageForWrongArgumentInGetter(algo->name(), parameter, p->type, argType);
                CV_Error(CV_StsBadArg, message);
            }
            bool val = p->getter ? (algo->*f.get_bool)() : *(bool*)((uchar*)algo + p->offset);

            if( argType == Param::INT )
                *(int*)value = (int)val;
            else if( argType == Param::BOOLEAN )
                *(bool*)value = val;
            else if ( argType == Param::REAL )
                *(double*)value = (int)val;
            else if ( argType == Param::FLOAT)
                *(float*)value = (float)((int)val);
            else if ( argType == Param::UNSIGNED_INT )
                *(unsigned int*)value = (unsigned int)val;
            else if ( argType == Param::UINT64 )
                *(uint64*)value = (int)val;
            else if ( argType == Param::UCHAR)
                *(uchar*)value = (uchar)val;
            else
                CV_Error(CV_StsBadArg, "Wrong argument type");
        }
        else if( p->type == Param::REAL )
        {
            if(!( argType == Param::REAL || argType == Param::FLOAT))
            {
                String message = getErrorMessageForWrongArgumentInGetter(algo->name(), parameter, p->type, argType);
                CV_Error(CV_StsBadArg, message);
            }
            double val = p->getter ? (algo->*f.get_double)() : *(double*)((uchar*)algo + p->offset);

            if ( argType == Param::REAL )
                *(double*)value = val;
            else if ( argType == Param::FLOAT)
                *(float*)value = (float)val;
            else
                CV_Error(CV_StsBadArg, "Wrong argument type");
        }
        else if( p->type == Param::FLOAT )
        {
            if(!( argType == Param::REAL || argType == Param::FLOAT))
            {
                String message = getErrorMessageForWrongArgumentInGetter(algo->name(), parameter, p->type, argType);
                CV_Error(CV_StsBadArg, message);
            }
            float val = p->getter ? (algo->*f.get_float)() : *(float*)((uchar*)algo + p->offset);

            if ( argType == Param::REAL )
                *(double*)value = (double)val;
            else if ( argType == Param::FLOAT)
                *(float*)value = (float)val;
            else
                CV_Error(CV_StsBadArg, "Wrong argument type");
        }
        else if( p->type == Param::UNSIGNED_INT )
        {
            if (!( argType == Param::INT || argType == Param::REAL || argType == Param::FLOAT || argType == Param::UNSIGNED_INT || argType == Param::UINT64 || argType == Param::UCHAR))
            {
                String message = getErrorMessageForWrongArgumentInGetter(algo->name(), parameter, p->type, argType);
                CV_Error(CV_StsBadArg, message);
            }
            unsigned int val = p->getter ? (algo->*f.get_uint)() : *(unsigned int*)((uchar*)algo + p->offset);

            if( argType == Param::INT )
                *(int*)value = (int)val;
            else if ( argType == Param::REAL )
                *(double*)value = (double)val;
            else if ( argType == Param::FLOAT)
                *(float*)value = (float)val;
            else if ( argType == Param::UNSIGNED_INT )
                *(unsigned int*)value = (unsigned int)val;
            else if ( argType == Param::UINT64 )
                *(uint64*)value = (uint64)val;
            else if ( argType == Param::UCHAR)
                *(uchar*)value = (uchar)val;
            else
                CV_Error(CV_StsBadArg, "Wrong argument type");
        }
        else if( p->type == Param::UINT64 )
        {
            if (!( argType == Param::INT || argType == Param::REAL || argType == Param::FLOAT || argType == Param::UNSIGNED_INT || argType == Param::UINT64 || argType == Param::UCHAR))
            {
                String message = getErrorMessageForWrongArgumentInGetter(algo->name(), parameter, p->type, argType);
                CV_Error(CV_StsBadArg, message);
        }
            uint64 val = p->getter ? (algo->*f.get_uint64)() : *(uint64*)((uchar*)algo + p->offset);

            if( argType == Param::INT )
                *(int*)value = (int)val;
            else if ( argType == Param::REAL )
                *(double*)value = (double)val;
            else if ( argType == Param::FLOAT)
                *(float*)value = (float)val;
            else if ( argType == Param::UNSIGNED_INT )
                *(unsigned int*)value = (unsigned int)val;
            else if ( argType == Param::UINT64 )
                *(uint64*)value = (uint64)val;
            else if ( argType == Param::UCHAR)
                *(uchar*)value = (uchar)val;
        else
                CV_Error(CV_StsBadArg, "Wrong argument type");
        }
        else if( p->type == Param::UCHAR )
        {
            if (!( argType == Param::INT || argType == Param::REAL || argType == Param::FLOAT || argType == Param::UNSIGNED_INT || argType == Param::UINT64 || argType == Param::UCHAR))
            {
                String message = getErrorMessageForWrongArgumentInGetter(algo->name(), parameter, p->type, argType);
                CV_Error(CV_StsBadArg, message);
            }
            uchar val = p->getter ? (algo->*f.get_uchar)() : *(uchar*)((uchar*)algo + p->offset);

            if( argType == Param::INT )
                *(int*)value = val;
            else if ( argType == Param::REAL )
            *(double*)value = val;
            else if ( argType == Param::FLOAT)
                *(float*)value = val;
            else if ( argType == Param::UNSIGNED_INT )
                *(unsigned int*)value = val;
            else if ( argType == Param::UINT64 )
                *(uint64*)value = val;
            else if ( argType == Param::UCHAR)
                *(uchar*)value = val;
            else
                CV_Error(CV_StsBadArg, "Wrong argument type");

        }
        else
            CV_Error(CV_StsBadArg, "Unknown/unsupported parameter type");
    }
    else if( argType == Param::STRING )
    {
        if( p->type != Param::STRING )
        {
            String message = getErrorMessageForWrongArgumentInGetter(algo->name(), parameter, p->type, argType);
            CV_Error(CV_StsBadArg, message);
        }

        *(String*)value = p->getter ? (algo->*f.get_string)() :
            *(String*)((uchar*)algo + p->offset);
    }
    else if( argType == Param::MAT )
    {
        if( p->type != Param::MAT )
        {
            String message = getErrorMessageForWrongArgumentInGetter(algo->name(), parameter, p->type, argType);
            CV_Error(CV_StsBadArg, message);
        }

        *(Mat*)value = p->getter ? (algo->*f.get_mat)() :
            *(Mat*)((uchar*)algo + p->offset);
    }
    else if( argType == Param::MAT_VECTOR )
    {
        if( p->type != Param::MAT_VECTOR )
        {
            String message = getErrorMessageForWrongArgumentInGetter(algo->name(), parameter, p->type, argType);
            CV_Error(CV_StsBadArg, message);
        }

        *(std::vector<Mat>*)value = p->getter ? (algo->*f.get_mat_vector)() :
        *(std::vector<Mat>*)((uchar*)algo + p->offset);
    }
    else if( argType == Param::ALGORITHM )
    {
        if( p->type != Param::ALGORITHM )
        {
            String message = getErrorMessageForWrongArgumentInGetter(algo->name(), parameter, p->type, argType);
            CV_Error(CV_StsBadArg, message);
        }

        *(Ptr<Algorithm>*)value = p->getter ? (algo->*f.get_algo)() :
            *(Ptr<Algorithm>*)((uchar*)algo + p->offset);
    }
    else
    {
        String message = getErrorMessageForWrongArgumentInGetter(algo->name(), parameter, p->type, argType);
        CV_Error(CV_StsBadArg, message);
    }
}


int AlgorithmInfo::paramType(const char* parameter) const
{
    const Param* p = findstr(data->params, parameter);
    if( !p )
        CV_Error_( CV_StsBadArg, ("No parameter '%s' is found", parameter ? parameter : "<NULL>") );
    return p->type;
}


String AlgorithmInfo::paramHelp(const char* parameter) const
{
    const Param* p = findstr(data->params, parameter);
    if( !p )
        CV_Error_( CV_StsBadArg, ("No parameter '%s' is found", parameter ? parameter : "<NULL>") );
    return p->help;
}


void AlgorithmInfo::getParams(std::vector<String>& names) const
{
    data->params.get_keys(names);
}


void AlgorithmInfo::addParam_(Algorithm& algo, const char* parameter, int argType,
                              void* value, bool readOnly,
                              Algorithm::Getter getter, Algorithm::Setter setter,
                              const String& help)
{
    CV_Assert( argType == Param::INT || argType == Param::BOOLEAN ||
               argType == Param::REAL || argType == Param::STRING ||
               argType == Param::MAT || argType == Param::MAT_VECTOR ||
               argType == Param::ALGORITHM
               || argType == Param::FLOAT || argType == Param::UNSIGNED_INT || argType == Param::UINT64
               || argType == Param::UCHAR);
    data->params.add(String(parameter), Param(argType, readOnly,
                     (int)((size_t)value - (size_t)(void*)&algo),
                     getter, setter, help));
}


void AlgorithmInfo::addParam(Algorithm& algo, const char* parameter,
                             int& value, bool readOnly,
                             int (Algorithm::*getter)(),
                             void (Algorithm::*setter)(int),
                             const String& help)
{
    addParam_(algo, parameter, ParamType<int>::type, &value, readOnly,
              (Algorithm::Getter)getter, (Algorithm::Setter)setter, help);
}

void AlgorithmInfo::addParam(Algorithm& algo, const char* parameter,
                             bool& value, bool readOnly,
                             int (Algorithm::*getter)(),
                             void (Algorithm::*setter)(int),
                             const String& help)
{
    addParam_(algo, parameter, ParamType<bool>::type, &value, readOnly,
              (Algorithm::Getter)getter, (Algorithm::Setter)setter, help);
}

void AlgorithmInfo::addParam(Algorithm& algo, const char* parameter,
                             double& value, bool readOnly,
                             double (Algorithm::*getter)(),
                             void (Algorithm::*setter)(double),
                             const String& help)
{
    addParam_(algo, parameter, ParamType<double>::type, &value, readOnly,
              (Algorithm::Getter)getter, (Algorithm::Setter)setter, help);
}

void AlgorithmInfo::addParam(Algorithm& algo, const char* parameter,
                             String& value, bool readOnly,
                             String (Algorithm::*getter)(),
                             void (Algorithm::*setter)(const String&),
                             const String& help)
{
    addParam_(algo, parameter, ParamType<String>::type, &value, readOnly,
              (Algorithm::Getter)getter, (Algorithm::Setter)setter, help);
}

void AlgorithmInfo::addParam(Algorithm& algo, const char* parameter,
                             Mat& value, bool readOnly,
                             Mat (Algorithm::*getter)(),
                             void (Algorithm::*setter)(const Mat&),
                             const String& help)
{
    addParam_(algo, parameter, ParamType<Mat>::type, &value, readOnly,
              (Algorithm::Getter)getter, (Algorithm::Setter)setter, help);
}

void AlgorithmInfo::addParam(Algorithm& algo, const char* parameter,
                             std::vector<Mat>& value, bool readOnly,
                             std::vector<Mat> (Algorithm::*getter)(),
                             void (Algorithm::*setter)(const std::vector<Mat>&),
                             const String& help)
{
    addParam_(algo, parameter, ParamType<std::vector<Mat> >::type, &value, readOnly,
              (Algorithm::Getter)getter, (Algorithm::Setter)setter, help);
}

void AlgorithmInfo::addParam(Algorithm& algo, const char* parameter,
                             Ptr<Algorithm>& value, bool readOnly,
                             Ptr<Algorithm> (Algorithm::*getter)(),
                             void (Algorithm::*setter)(const Ptr<Algorithm>&),
                             const String& help)
{
    addParam_(algo, parameter, ParamType<Algorithm>::type, &value, readOnly,
              (Algorithm::Getter)getter, (Algorithm::Setter)setter, help);
}

void AlgorithmInfo::addParam(Algorithm& algo, const char* parameter,
                             float& value, bool readOnly,
                             float (Algorithm::*getter)(),
                             void (Algorithm::*setter)(float),
                             const String& help)
{
    addParam_(algo, parameter, ParamType<float>::type, &value, readOnly,
              (Algorithm::Getter)getter, (Algorithm::Setter)setter, help);
}

void AlgorithmInfo::addParam(Algorithm& algo, const char* parameter,
                             unsigned int& value, bool readOnly,
                             unsigned int (Algorithm::*getter)(),
                             void (Algorithm::*setter)(unsigned int),
                             const String& help)
{
    addParam_(algo, parameter, ParamType<unsigned int>::type, &value, readOnly,
              (Algorithm::Getter)getter, (Algorithm::Setter)setter, help);
}

void AlgorithmInfo::addParam(Algorithm& algo, const char* parameter,
                             uint64& value, bool readOnly,
                             uint64 (Algorithm::*getter)(),
                             void (Algorithm::*setter)(uint64),
                             const String& help)
{
    addParam_(algo, parameter, ParamType<uint64>::type, &value, readOnly,
              (Algorithm::Getter)getter, (Algorithm::Setter)setter, help);
}

void AlgorithmInfo::addParam(Algorithm& algo, const char* parameter,
                             uchar& value, bool readOnly,
                             uchar (Algorithm::*getter)(),
                             void (Algorithm::*setter)(uchar),
                             const String& help)
{
    addParam_(algo, parameter, ParamType<uchar>::type, &value, readOnly,
              (Algorithm::Getter)getter, (Algorithm::Setter)setter, help);
}

}

/* End of file. */
