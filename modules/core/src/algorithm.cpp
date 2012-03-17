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

using std::pair;
    
template<typename _KeyTp, typename _ValueTp> struct sorted_vector
{
    sorted_vector() {}
    void clear() { vec.clear(); }
    size_t size() const { return vec.size(); }
    _ValueTp& operator [](size_t idx) { return vec[idx]; }
    const _ValueTp& operator [](size_t idx) const { return vec[idx]; }
    
    void add(const _KeyTp& k, const _ValueTp& val)
    {
        pair<_KeyTp, _ValueTp> p(k, val);
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
    
    void get_keys(vector<_KeyTp>& keys) const
    {
        size_t i = 0, n = vec.size();
        keys.resize(n);
        
        for( i = 0; i < n; i++ )
            keys[i] = vec[i].first;
    }
    
    vector<pair<_KeyTp, _ValueTp> > vec;
};

    
template<typename _ValueTp> inline const _ValueTp* findstr(const sorted_vector<string, _ValueTp>& vec,
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
    
    if( strcmp(vec.vec[a].first.c_str(), key) == 0 )
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
             const string& _help)
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
    sorted_vector<string, Param> params;
    string _name;
};

    
static sorted_vector<string, Algorithm::Constructor> alglist;

void Algorithm::getList(vector<string>& algorithms)
{
    alglist.get_keys(algorithms);
}

Ptr<Algorithm> Algorithm::_create(const string& name)
{
    Algorithm::Constructor c = 0;
    if( !alglist.find(name, c) )
        return Ptr<Algorithm>();
    return c();
}

Algorithm::Algorithm()
{
}
    
Algorithm::~Algorithm()
{
}
    
string Algorithm::name() const
{
    return info()->name();
}
  
void Algorithm::set(const string& name, int value)
{
    info()->set(this, name.c_str(), ParamType<int>::type, &value);
}

void Algorithm::set(const string& name, double value)
{
    info()->set(this, name.c_str(), ParamType<double>::type, &value);
}

void Algorithm::set(const string& name, bool value)
{
    info()->set(this, name.c_str(), ParamType<bool>::type, &value);
}

void Algorithm::set(const string& name, const string& value)
{
    info()->set(this, name.c_str(), ParamType<string>::type, &value);
}

void Algorithm::set(const string& name, const Mat& value)
{
    info()->set(this, name.c_str(), ParamType<Mat>::type, &value);
}

void Algorithm::set(const string& name, const Ptr<Algorithm>& value)
{
    info()->set(this, name.c_str(), ParamType<Algorithm>::type, &value);
}

void Algorithm::set(const char* name, int value)
{
    info()->set(this, name, ParamType<int>::type, &value);
}

void Algorithm::set(const char* name, double value)
{
    info()->set(this, name, ParamType<double>::type, &value);
}

void Algorithm::set(const char* name, bool value)
{
    info()->set(this, name, ParamType<bool>::type, &value);
}

void Algorithm::set(const char* name, const string& value)
{
    info()->set(this, name, ParamType<string>::type, &value);
}

void Algorithm::set(const char* name, const Mat& value)
{
    info()->set(this, name, ParamType<Mat>::type, &value);
}

void Algorithm::set(const char* name, const Ptr<Algorithm>& value)
{
    info()->set(this, name, ParamType<Algorithm>::type, &value);
}
    
string Algorithm::paramHelp(const string& name) const
{
    return info()->paramHelp(name.c_str());
}
    
int Algorithm::paramType(const string& name) const
{
    return info()->paramType(name.c_str());
}

int Algorithm::paramType(const char* name) const
{
    return info()->paramType(name);
}    
    
void Algorithm::getParams(vector<string>& names) const
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

    
AlgorithmInfo::AlgorithmInfo(const string& _name, Algorithm::Constructor create)
{
    data = new AlgorithmInfoData;
    data->_name = _name;
    alglist.add(_name, create);
}

AlgorithmInfo::~AlgorithmInfo()
{
    delete data;
}    
    
void AlgorithmInfo::write(const Algorithm* algo, FileStorage& fs) const
{
    size_t i = 0, n = data->params.vec.size();
    cv::write(fs, "name", algo->name());
    for( i = 0; i < n; i++ )
    {
        const Param& p = data->params.vec[i].second;
        const string& pname = data->params.vec[i].first;
        if( p.type == Param::INT )
            cv::write(fs, pname, algo->get<int>(pname));
        else if( p.type == Param::BOOLEAN )
            cv::write(fs, pname, (int)algo->get<bool>(pname));
        else if( p.type == Param::REAL )
            cv::write(fs, pname, algo->get<double>(pname));
        else if( p.type == Param::STRING )
            cv::write(fs, pname, algo->get<string>(pname));
        else if( p.type == Param::MAT )
            cv::write(fs, pname, algo->get<Mat>(pname));
        else if( p.type == Param::ALGORITHM )
        {
            WriteStructContext ws(fs, pname, CV_NODE_MAP);
            Ptr<Algorithm> nestedAlgo = algo->get<Algorithm>(pname);
            nestedAlgo->write(fs);
        }
        else
            CV_Error( CV_StsUnsupportedFormat, "unknown/unsupported parameter type");
    }
}

void AlgorithmInfo::read(Algorithm* algo, const FileNode& fn) const
{
    size_t i = 0, n = data->params.vec.size();
    
    for( i = 0; i < n; i++ )
    {
        const Param& p = data->params.vec[i].second;
        const string& pname = data->params.vec[i].first;
        FileNode n = fn[pname];
        if( n.empty() )
            continue;
        if( p.type == Param::INT )
            algo->set(pname, (int)n);
        else if( p.type == Param::BOOLEAN )
            algo->set(pname, (int)n != 0);
        else if( p.type == Param::REAL )
            algo->set(pname, (double)n);
        else if( p.type == Param::STRING )
            algo->set(pname, (string)n);
        else if( p.type == Param::MAT )
        {
            Mat m;
            cv::read(fn, m);
            algo->set(pname, m);
        }
        else if( p.type == Param::ALGORITHM )
        {
            Ptr<Algorithm> nestedAlgo = Algorithm::_create((string)n["name"]);
            CV_Assert( !nestedAlgo.empty() );
            nestedAlgo->read(n);
            algo->set(pname, nestedAlgo);
        }
        else
            CV_Error( CV_StsUnsupportedFormat, "unknown/unsupported parameter type");
    }
}    

string AlgorithmInfo::name() const
{
    return data->_name;
}
    
union GetSetParam
{
    int (Algorithm::*get_int)() const;
    bool (Algorithm::*get_bool)() const;
    double (Algorithm::*get_double)() const;
    string (Algorithm::*get_string)() const;
    Mat (Algorithm::*get_mat)() const;
    Ptr<Algorithm> (Algorithm::*get_algo)() const;
    
    void (Algorithm::*set_int)(int);
    void (Algorithm::*set_bool)(bool);
    void (Algorithm::*set_double)(double);
    void (Algorithm::*set_string)(const string&);
    void (Algorithm::*set_mat)(const Mat&);
    void (Algorithm::*set_algo)(const Ptr<Algorithm>&);
};
    
void AlgorithmInfo::set(Algorithm* algo, const char* name, int argType, const void* value) const
{
    const Param* p = findstr(data->params, name);
    
    if( !p )
        CV_Error_( CV_StsBadArg, ("No parameter '%s' is found", name ? name : "<NULL>") );
    
    if( p->readonly )
        CV_Error_( CV_StsError, ("Parameter '%s' is readonly", name));
    
    GetSetParam f;
    f.set_int = p->setter;
    
    if( argType == Param::INT || argType == Param::BOOLEAN || argType == Param::REAL )
    {
        CV_Assert( p->type == Param::INT || p->type == Param::REAL || p->type == Param::BOOLEAN );
        
        if( p->type == Param::INT )
        {
            int val = argType == Param::INT ? *(const int*)value :
                argType == Param::BOOLEAN ? (int)*(const bool*)value :
                saturate_cast<int>(*(const double*)value);
            if( p->setter )
                (algo->*f.set_int)(val);
            else
                *(int*)((uchar*)algo + p->offset) = val;
        }
        else if( p->type == Param::BOOLEAN )
        {
            bool val = argType == Param::INT ? *(const int*)value != 0 :
                    argType == Param::BOOLEAN ? *(const bool*)value :
                    *(const double*)value != 0;
            if( p->setter )
                (algo->*f.set_bool)(val);
            else
                *(bool*)((uchar*)algo + p->offset) = val;
        }
        else
        {
            double val = argType == Param::INT ? (double)*(const int*)value :
                         argType == Param::BOOLEAN ? (double)*(const bool*)value :
                        *(const double*)value;
            if( p->setter )
                (algo->*f.set_double)(val);
            else
                *(double*)((uchar*)algo + p->offset) = val;
        }
    }
    else if( argType == Param::STRING )
    {
        CV_Assert( p->type == Param::STRING );
        
        const string& val = *(const string*)value;
        if( p->setter )
            (algo->*f.set_string)(val);
        else
            *(string*)((uchar*)algo + p->offset) = val;
    }
    else if( argType == Param::MAT )
    {
        CV_Assert( p->type == Param::MAT );
        
        const Mat& val = *(const Mat*)value;
        if( p->setter )
            (algo->*f.set_mat)(val);
        else
            *(Mat*)((uchar*)algo + p->offset) = val;
    }
    else if( argType == Param::ALGORITHM )
    {
        CV_Assert( p->type == Param::ALGORITHM );
        
        const Ptr<Algorithm>& val = *(const Ptr<Algorithm>*)value;
        if( p->setter )
            (algo->*f.set_algo)(val);
        else
            *(Ptr<Algorithm>*)((uchar*)algo + p->offset) = val;
    }
    else
        CV_Error(CV_StsBadArg, "Unknown/unsupported parameter type");
}
    
void AlgorithmInfo::get(const Algorithm* algo, const char* name, int argType, void* value) const
{
    const Param* p = findstr(data->params, name);
    if( !p )
        CV_Error_( CV_StsBadArg, ("No parameter '%s' is found", name ? name : "<NULL>") );
    
    GetSetParam f;
    f.get_int = p->getter;
    
    if( argType == Param::INT || argType == Param::BOOLEAN || argType == Param::REAL )
    {
        if( p->type == Param::INT )
        {
            CV_Assert( argType == Param::INT || argType == Param::REAL );
            int val = p->getter ? (algo->*f.get_int)() : *(int*)((uchar*)algo + p->offset);
            
            if( argType == Param::INT )
                *(int*)value = val;
            else
                *(double*)value = val;
        }
        else if( p->type == Param::BOOLEAN )
        {
            CV_Assert( argType == Param::INT || argType == Param::BOOLEAN || argType == Param::REAL );
            bool val = p->getter ? (algo->*f.get_bool)() : *(bool*)((uchar*)algo + p->offset);
            
            if( argType == Param::INT )
                *(int*)value = (int)val;
            else if( argType == Param::BOOLEAN )
                *(bool*)value = val;
            else
                *(double*)value = (int)val;
        }
        else
        {
            CV_Assert( argType == Param::REAL );
            double val = p->getter ? (algo->*f.get_double)() : *(double*)((uchar*)algo + p->offset);
            
            *(double*)value = val;
        }
    }
    else if( argType == Param::STRING )
    {
        CV_Assert( p->type == Param::STRING );
        
        *(string*)value = p->getter ? (algo->*f.get_string)() :
            *(string*)((uchar*)algo + p->offset);
    }
    else if( argType == Param::MAT )
    {
        CV_Assert( p->type == Param::MAT );
        
        *(Mat*)value = p->getter ? (algo->*f.get_mat)() :
            *(Mat*)((uchar*)algo + p->offset);
    }
    else if( argType == Param::ALGORITHM )
    {
        CV_Assert( p->type == Param::ALGORITHM );
        
        *(Ptr<Algorithm>*)value = p->getter ? (algo->*f.get_algo)() :
            *(Ptr<Algorithm>*)((uchar*)algo + p->offset);
    }
    else
        CV_Error(CV_StsBadArg, "Unknown/unsupported parameter type");
}

    
int AlgorithmInfo::paramType(const char* name) const
{
    const Param* p = findstr(data->params, name);
    if( !p )
        CV_Error_( CV_StsBadArg, ("No parameter '%s' is found", name ? name : "<NULL>") );
    return p->type;
}
    
    
string AlgorithmInfo::paramHelp(const char* name) const
{
    const Param* p = findstr(data->params, name);
    if( !p )
        CV_Error_( CV_StsBadArg, ("No parameter '%s' is found", name ? name : "<NULL>") );
    return p->help;
}


void AlgorithmInfo::getParams(vector<string>& names) const
{
    data->params.get_keys(names);
}
    
    
void AlgorithmInfo::addParam_(Algorithm& algo, const char* name, int argType,
                              void* value, bool readOnly, 
                              Algorithm::Getter getter, Algorithm::Setter setter,
                              const string& help)
{
    CV_Assert( argType == Param::INT || argType == Param::BOOLEAN ||
               argType == Param::REAL || argType == Param::STRING ||
               argType == Param::MAT || argType == Param::ALGORITHM );
    data->params.add(string(name), Param(argType, readOnly,
                     (int)((size_t)value - (size_t)(void*)&algo),
                     getter, setter, help));
}
    
    
void AlgorithmInfo::addParam(Algorithm& algo, const char* name,
                             int& value, bool readOnly, 
                             int (Algorithm::*getter)(),
                             void (Algorithm::*setter)(int),
                             const string& help)
{
    addParam_(algo, name, ParamType<int>::type, &value, readOnly,
              (Algorithm::Getter)getter, (Algorithm::Setter)setter, help);
}

void AlgorithmInfo::addParam(Algorithm& algo, const char* name,
                             bool& value, bool readOnly, 
                             int (Algorithm::*getter)(),
                             void (Algorithm::*setter)(int),
                             const string& help)
{
    addParam_(algo, name, ParamType<bool>::type, &value, readOnly,
              (Algorithm::Getter)getter, (Algorithm::Setter)setter, help);
}
    
void AlgorithmInfo::addParam(Algorithm& algo, const char* name,
                             double& value, bool readOnly, 
                             double (Algorithm::*getter)(),
                             void (Algorithm::*setter)(double),
                             const string& help)
{
    addParam_(algo, name, ParamType<double>::type, &value, readOnly,
              (Algorithm::Getter)getter, (Algorithm::Setter)setter, help);
}

void AlgorithmInfo::addParam(Algorithm& algo, const char* name,
                             string& value, bool readOnly, 
                             string (Algorithm::*getter)(),
                             void (Algorithm::*setter)(const string&),
                             const string& help)
{
    addParam_(algo, name, ParamType<string>::type, &value, readOnly,
              (Algorithm::Getter)getter, (Algorithm::Setter)setter, help);
}

void AlgorithmInfo::addParam(Algorithm& algo, const char* name,
                             Mat& value, bool readOnly, 
                             Mat (Algorithm::*getter)(),
                             void (Algorithm::*setter)(const Mat&),
                             const string& help)
{
    addParam_(algo, name, ParamType<Mat>::type, &value, readOnly,
              (Algorithm::Getter)getter, (Algorithm::Setter)setter, help);
}
    
void AlgorithmInfo::addParam(Algorithm& algo, const char* name,
                             Ptr<Algorithm>& value, bool readOnly, 
                             Ptr<Algorithm> (Algorithm::*getter)(),
                             void (Algorithm::*setter)(const Ptr<Algorithm>&),
                             const string& help)
{
    addParam_(algo, name, ParamType<Algorithm>::type, &value, readOnly,
              (Algorithm::Getter)getter, (Algorithm::Setter)setter, help);
}    

}
    
/* End of file. */
