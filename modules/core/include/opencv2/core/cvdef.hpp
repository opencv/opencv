//
//  cvdef.hpp
//  OpenCV
//
//  Created by Jasper Shemilt on 05/09/2013.
//
//

#ifndef OpenCV_cvdef_hpp
#define OpenCV_cvdef_hpp


#ifndef __cplusplus
#  error core.hpp header must be compiled as C++
#endif

namespace cv
{

    struct cv_2bitType{
        union{
            struct{
                unsigned char val : 2;
                unsigned char     : 6;
            };
            unsigned char raw;
        };
        cv_2bitType()=default;
        explicit constexpr cv_2bitType(unsigned char a_):raw(a_){};
        operator CV_8U_TYPE() const {return CV_8U_TYPE(raw);};
        cv_2bitType& operator=(CV_8U_TYPE a_){raw = a_;return *this;}
        cv_2bitType& operator=(CV_8S_TYPE a_){raw = a_;return *this;}
        cv_2bitType& operator=(CV_16U_TYPE a_){raw = a_;return *this;}
        cv_2bitType& operator=(CV_16S_TYPE a_){raw = a_;return *this;}
        cv_2bitType& operator=(CV_32U_TYPE a_){raw = a_;return *this;}
        cv_2bitType& operator=(CV_32S_TYPE a_){raw = a_;return *this;}
        cv_2bitType& operator=(CV_64U_TYPE a_){raw = a_;return *this;}
        cv_2bitType& operator=(CV_64S_TYPE a_){raw = a_;return *this;}
    };

    struct cv_4bitType{
        union{
            struct{
                unsigned char val : 4;
                unsigned char     : 4;
            };
            unsigned char raw;
        };
#  if defined __cplusplus
        explicit constexpr cv_4bitType(unsigned char a_):val(a_){};
        cv_4bitType()=default;
        operator CV_8U_TYPE() const {return CV_8U_TYPE(raw);};
        cv_4bitType& operator=(CV_8U_TYPE  a_){raw = a_;return *this;}
        cv_4bitType& operator=(CV_8S_TYPE  a_){raw = a_;return *this;}
        cv_4bitType& operator=(CV_16U_TYPE a_){raw = a_;return *this;}
        cv_4bitType& operator=(CV_16S_TYPE a_){raw = a_;return *this;}
        cv_4bitType& operator=(CV_32U_TYPE a_){raw = a_;return *this;}
        cv_4bitType& operator=(CV_32S_TYPE a_){raw = a_;return *this;}
        cv_4bitType& operator=(CV_64U_TYPE a_){raw = a_;return *this;}
        cv_4bitType& operator=(CV_64S_TYPE a_){raw = a_;return *this;}
#  endif
    };
};


#  if defined __cplusplus
template<int t> struct cv_Data_Type{
    using type = unsigned char;
    const static int dataType = t;
    const static int bitDepth  = CV_MAT_DEPTH_BITS(t);
    const static int byteDepth = CV_MAT_DEPTH_BYTES(t);
    //    constexpr static type max  = CV_MAT_MAX(t);
    //    constexpr static type min  = CV_MAT_MIN(t);

};
template<> struct cv_Data_Type<CV_2U>{
    using type = CV_2U_TYPE;
    const static int dataType = CV_2U;
    const static int bitDepth  = CV_MAT_DEPTH_BITS(CV_2U);
    const static int byteDepth = CV_MAT_DEPTH_BYTES(CV_2U);
    constexpr static type max  = CV_2U_TYPE(CV_2U_MAX);
    constexpr static type min  = CV_2U_TYPE(CV_2U_MIN);
    const char* fmt = "hhu";
};
template<> struct cv_Data_Type<CV_4U>{
    using type = CV_4U_TYPE;
    const static int dataType = CV_4U;
    const static int bitDepth  = CV_MAT_DEPTH_BITS(CV_4U);
    const static int byteDepth = CV_MAT_DEPTH_BYTES(CV_4U);
    constexpr static type max  = CV_4U_TYPE(CV_4U_MAX);
    constexpr static type min  = CV_4U_TYPE(CV_4U_MIN);
    const char* fmt = "hhu";
};
template<> struct cv_Data_Type<CV_8U>{
    using type = CV_8U_TYPE;
    const static int dataType = CV_8U;
    const static int bitDepth  = CV_MAT_DEPTH_BITS(CV_8U);
    const static int byteDepth = CV_MAT_DEPTH_BYTES(CV_8U);
    constexpr static type max  = CV_8U_MAX;
    constexpr static type min  = CV_8U_MIN;
    const char* fmt = "hhu";
};
template<> struct cv_Data_Type<CV_8S>{
    using type = CV_8S_TYPE;
    const static int dataType = CV_8S;
    const static int bitDepth  = CV_MAT_DEPTH_BITS(CV_8S);
    const static int byteDepth = CV_MAT_DEPTH_BYTES(CV_8S);
    constexpr static type max  = CV_8S_MAX;
    constexpr static type min  = CV_8S_MIN;
    const char* fmt = "hhi";
};
template<> struct cv_Data_Type<CV_16U>{
    using type = CV_16U_TYPE;
    const static int dataType = CV_16U;
    const static int bitDepth  = CV_MAT_DEPTH_BITS(CV_16U);
    const static int byteDepth = CV_MAT_DEPTH_BYTES(CV_16U);
    constexpr static type max  = CV_16U_MAX;
    constexpr static type min  = CV_16U_MIN;
    const char* fmt = "hu";
};
template<> struct cv_Data_Type<CV_16S>{
    using type = CV_16S_TYPE;
    const static int dataType = CV_16S;
    const static int bitDepth  = CV_MAT_DEPTH_BITS(CV_16S);
    const static int byteDepth = CV_MAT_DEPTH_BYTES(CV_16S);
    constexpr static type max  = CV_16S_MAX;
    constexpr static type min  = CV_16S_MIN;
    const char* fmt = "hi";
};
template<> struct cv_Data_Type<CV_32U>{
    using type = CV_32U_TYPE;
    const static int dataType = CV_32U;
    const static int bitDepth  = CV_MAT_DEPTH_BITS(CV_32U);
    const static int byteDepth = CV_MAT_DEPTH_BYTES(CV_32U);
    constexpr static type max  = CV_32U_MAX;
    constexpr static type min  = CV_32U_MIN;
    const char* fmt = "u";
};
template<> struct cv_Data_Type<CV_32S>{
    using type = CV_32S_TYPE;
    const static int dataType = CV_32S;
    const static int bitDepth  = CV_MAT_DEPTH_BITS(CV_32S);
    const static int byteDepth = CV_MAT_DEPTH_BYTES(CV_32S);
    constexpr static type max  = CV_32S_MAX;
    constexpr static type min  = CV_32S_MIN;
    const char* fmt = "i";
};
template<> struct cv_Data_Type<CV_64U>{
    using type = CV_64U_TYPE;
    const static int dataType = CV_64U;
    const static int bitDepth  = CV_MAT_DEPTH_BITS(CV_64U);
    const static int byteDepth = CV_MAT_DEPTH_BYTES(CV_64U);
    constexpr static type max  = CV_64U_MAX;
    constexpr static type min  = CV_64U_MIN;
    const char* fmt = "llu";
};
template<> struct cv_Data_Type<CV_64S>{
    using type = CV_64S_TYPE;
    const static int dataType = CV_64S;
    const static int bitDepth  = CV_MAT_DEPTH_BITS(CV_64S);
    const static int byteDepth = CV_MAT_DEPTH_BYTES(CV_64S);
    const static type max  = CV_64S_MAX;
    const static type min  = CV_64S_MIN;
    const char* fmt = "lli";
};
template<> struct cv_Data_Type<CV_32F>{
    using type = CV_32F_TYPE;
    const static int dataType = CV_32F;
    const static int bitDepth  = CV_MAT_DEPTH_BITS(CV_32F);
    const static int byteDepth = CV_MAT_DEPTH_BYTES(CV_32F);
    constexpr static type max  = CV_32F_MAX;
    constexpr static type min  = CV_32F_MIN;
    const char* fmt = "f";
};
template<> struct cv_Data_Type<CV_64F>{
    using type = CV_64F_TYPE;
    const static int dataType = CV_64F;
    const static int bitDepth  = CV_MAT_DEPTH_BITS(CV_64F);
    const static int byteDepth = CV_MAT_DEPTH_BYTES(CV_64F);
    constexpr static type max  = CV_64F_MAX;
    constexpr static type min  = CV_64F_MIN;
    const char* fmt = "f";
};

template<int t1, int t2> struct cv_Work_Type;

template<> struct cv_Work_Type<CV_2U, CV_2U> : cv_Data_Type<CV_4U>{};
template<> struct cv_Work_Type<CV_2U, CV_4U> : cv_Data_Type<CV_8U>{};
template<> struct cv_Work_Type<CV_2U, CV_8U> : cv_Data_Type<CV_16U>{};
template<> struct cv_Work_Type<CV_2U,CV_16U> : cv_Data_Type<CV_32U>{};
template<> struct cv_Work_Type<CV_2U,CV_32U> : cv_Data_Type<CV_64U>{};
template<> struct cv_Work_Type<CV_2U,CV_64U> : cv_Data_Type<CV_64U>{};

template<> struct cv_Work_Type<CV_2U, CV_8S> : cv_Data_Type<CV_16S>{};
template<> struct cv_Work_Type<CV_2U,CV_16S> : cv_Data_Type<CV_32S>{};
template<> struct cv_Work_Type<CV_2U,CV_32S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_2U,CV_64S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_2U,CV_32F> : cv_Data_Type<CV_64F>{};
template<> struct cv_Work_Type<CV_2U,CV_64F> : cv_Data_Type<CV_64F>{};

template<> struct cv_Work_Type<CV_4U, CV_2U> : cv_Data_Type<CV_8U>{};
template<> struct cv_Work_Type<CV_4U, CV_4U> : cv_Data_Type<CV_16U>{};
template<> struct cv_Work_Type<CV_4U, CV_8U> : cv_Data_Type<CV_32U>{};
template<> struct cv_Work_Type<CV_4U,CV_16U> : cv_Data_Type<CV_64U>{};
template<> struct cv_Work_Type<CV_4U,CV_32U> : cv_Data_Type<CV_64U>{};
template<> struct cv_Work_Type<CV_4U,CV_64U> : cv_Data_Type<CV_64U>{};

template<> struct cv_Work_Type<CV_4U, CV_8S> : cv_Data_Type<CV_32S>{};
template<> struct cv_Work_Type<CV_4U,CV_16S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_4U,CV_32S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_4U,CV_64S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_4U,CV_32F> : cv_Data_Type<CV_64F>{};
template<> struct cv_Work_Type<CV_4U,CV_64F> : cv_Data_Type<CV_64F>{};

template<> struct cv_Work_Type<CV_8U, CV_2U> : cv_Data_Type<CV_16U>{};
template<> struct cv_Work_Type<CV_8U, CV_4U> : cv_Data_Type<CV_32U>{};
template<> struct cv_Work_Type<CV_8U, CV_8U> : cv_Data_Type<CV_16U>{};
template<> struct cv_Work_Type<CV_8U,CV_16U> : cv_Data_Type<CV_32U>{};
template<> struct cv_Work_Type<CV_8U,CV_32U> : cv_Data_Type<CV_64U>{};
template<> struct cv_Work_Type<CV_8U,CV_64U> : cv_Data_Type<CV_64U>{};

template<> struct cv_Work_Type<CV_8U, CV_8S> : cv_Data_Type<CV_16S>{};
template<> struct cv_Work_Type<CV_8U,CV_16S> : cv_Data_Type<CV_32S>{};
template<> struct cv_Work_Type<CV_8U,CV_32S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_8U,CV_64S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_8U,CV_32F> : cv_Data_Type<CV_64F>{};
template<> struct cv_Work_Type<CV_8U,CV_64F> : cv_Data_Type<CV_64F>{};

template<> struct cv_Work_Type<CV_8S, CV_2U> : cv_Data_Type<CV_16S>{};
template<> struct cv_Work_Type<CV_8S, CV_4U> : cv_Data_Type<CV_32S>{};
template<> struct cv_Work_Type<CV_8S, CV_8U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_8S,CV_16U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_8S,CV_32U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_8S,CV_64U> : cv_Data_Type<CV_64S>{};

template<> struct cv_Work_Type<CV_8S, CV_8S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_8S,CV_16S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_8S,CV_32S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_8S,CV_64S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_8S,CV_32F> : cv_Data_Type<CV_64F>{};
template<> struct cv_Work_Type<CV_8S,CV_64F> : cv_Data_Type<CV_64F>{};

template<> struct cv_Work_Type<CV_16U, CV_2U> : cv_Data_Type<CV_32U>{};
template<> struct cv_Work_Type<CV_16U, CV_4U> : cv_Data_Type<CV_64U>{};
template<> struct cv_Work_Type<CV_16U, CV_8U> : cv_Data_Type<CV_64U>{};
template<> struct cv_Work_Type<CV_16U,CV_16U> : cv_Data_Type<CV_64U>{};
template<> struct cv_Work_Type<CV_16U,CV_32U> : cv_Data_Type<CV_64U>{};
template<> struct cv_Work_Type<CV_16U,CV_64U> : cv_Data_Type<CV_64U>{};

template<> struct cv_Work_Type<CV_16U, CV_8S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_16U,CV_16S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_16U,CV_32S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_16U,CV_64S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_16U,CV_32F> : cv_Data_Type<CV_64F>{};
template<> struct cv_Work_Type<CV_16U,CV_64F> : cv_Data_Type<CV_64F>{};

template<> struct cv_Work_Type<CV_16S, CV_2U> : cv_Data_Type<CV_32S>{};
template<> struct cv_Work_Type<CV_16S, CV_4U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_16S, CV_8U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_16S,CV_16U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_16S,CV_32U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_16S,CV_64U> : cv_Data_Type<CV_64S>{};

template<> struct cv_Work_Type<CV_16S, CV_8S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_16S,CV_16S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_16S,CV_32S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_16S,CV_64S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_16S,CV_32F> : cv_Data_Type<CV_64F>{};
template<> struct cv_Work_Type<CV_16S,CV_64F> : cv_Data_Type<CV_64F>{};

template<> struct cv_Work_Type<CV_32U, CV_2U> : cv_Data_Type<CV_64U>{};
template<> struct cv_Work_Type<CV_32U, CV_4U> : cv_Data_Type<CV_64U>{};
template<> struct cv_Work_Type<CV_32U, CV_8U> : cv_Data_Type<CV_64U>{};
template<> struct cv_Work_Type<CV_32U,CV_16U> : cv_Data_Type<CV_64U>{};
template<> struct cv_Work_Type<CV_32U,CV_32U> : cv_Data_Type<CV_64U>{};
template<> struct cv_Work_Type<CV_32U,CV_64U> : cv_Data_Type<CV_64U>{};

template<> struct cv_Work_Type<CV_32U, CV_8S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_32U,CV_16S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_32U,CV_32S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_32U,CV_64S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_32U,CV_32F> : cv_Data_Type<CV_64F>{};
template<> struct cv_Work_Type<CV_32U,CV_64F> : cv_Data_Type<CV_64F>{};

template<> struct cv_Work_Type<CV_32S, CV_2U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_32S, CV_4U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_32S, CV_8U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_32S,CV_16U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_32S,CV_32U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_32S,CV_64U> : cv_Data_Type<CV_64S>{};

template<> struct cv_Work_Type<CV_32S, CV_8S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_32S,CV_16S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_32S,CV_32S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_32S,CV_64S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_32S,CV_32F> : cv_Data_Type<CV_64F>{};
template<> struct cv_Work_Type<CV_32S,CV_64F> : cv_Data_Type<CV_64F>{};

template<> struct cv_Work_Type<CV_64U, CV_2U> : cv_Data_Type<CV_64U>{};
template<> struct cv_Work_Type<CV_64U, CV_4U> : cv_Data_Type<CV_64U>{};
template<> struct cv_Work_Type<CV_64U, CV_8U> : cv_Data_Type<CV_64U>{};
template<> struct cv_Work_Type<CV_64U,CV_16U> : cv_Data_Type<CV_64U>{};
template<> struct cv_Work_Type<CV_64U,CV_32U> : cv_Data_Type<CV_64U>{};
template<> struct cv_Work_Type<CV_64U,CV_64U> : cv_Data_Type<CV_64U>{};

template<> struct cv_Work_Type<CV_64U, CV_8S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_64U,CV_16S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_64U,CV_32S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_64U,CV_64S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_64U,CV_32F> : cv_Data_Type<CV_64F>{};
template<> struct cv_Work_Type<CV_64U,CV_64F> : cv_Data_Type<CV_64F>{};

template<> struct cv_Work_Type<CV_64S, CV_2U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_64S, CV_4U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_64S, CV_8U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_64S,CV_16U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_64S,CV_32U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_64S,CV_64U> : cv_Data_Type<CV_64S>{};

template<> struct cv_Work_Type<CV_64S, CV_8S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_64S,CV_16S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_64S,CV_32S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_64S,CV_64S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Work_Type<CV_64S,CV_32F> : cv_Data_Type<CV_64F>{};
template<> struct cv_Work_Type<CV_64S,CV_64F> : cv_Data_Type<CV_64F>{};

template<> struct cv_Work_Type<CV_32F, CV_2U> : cv_Data_Type<CV_64F>{};
template<> struct cv_Work_Type<CV_32F, CV_4U> : cv_Data_Type<CV_64F>{};
template<> struct cv_Work_Type<CV_32F, CV_8U> : cv_Data_Type<CV_64F>{};
template<> struct cv_Work_Type<CV_32F,CV_16U> : cv_Data_Type<CV_64F>{};
template<> struct cv_Work_Type<CV_32F,CV_32U> : cv_Data_Type<CV_64F>{};
template<> struct cv_Work_Type<CV_32F,CV_64U> : cv_Data_Type<CV_64F>{};

template<> struct cv_Work_Type<CV_32F, CV_8S> : cv_Data_Type<CV_64F>{};
template<> struct cv_Work_Type<CV_32F,CV_16S> : cv_Data_Type<CV_64F>{};
template<> struct cv_Work_Type<CV_32F,CV_32S> : cv_Data_Type<CV_64F>{};
template<> struct cv_Work_Type<CV_32F,CV_64S> : cv_Data_Type<CV_64F>{};
template<> struct cv_Work_Type<CV_32F,CV_32F> : cv_Data_Type<CV_64F>{};
template<> struct cv_Work_Type<CV_32F,CV_64F> : cv_Data_Type<CV_64F>{};

template<> struct cv_Work_Type<CV_64F, CV_2U> : cv_Data_Type<CV_64F>{};
template<> struct cv_Work_Type<CV_64F, CV_4U> : cv_Data_Type<CV_64F>{};
template<> struct cv_Work_Type<CV_64F, CV_8U> : cv_Data_Type<CV_64F>{};
template<> struct cv_Work_Type<CV_64F,CV_16U> : cv_Data_Type<CV_64F>{};
template<> struct cv_Work_Type<CV_64F,CV_32U> : cv_Data_Type<CV_64F>{};
template<> struct cv_Work_Type<CV_64F,CV_64U> : cv_Data_Type<CV_64F>{};

template<> struct cv_Work_Type<CV_64F, CV_8S> : cv_Data_Type<CV_64F>{};
template<> struct cv_Work_Type<CV_64F,CV_16S> : cv_Data_Type<CV_64F>{};
template<> struct cv_Work_Type<CV_64F,CV_32S> : cv_Data_Type<CV_64F>{};
template<> struct cv_Work_Type<CV_64F,CV_64S> : cv_Data_Type<CV_64F>{};
template<> struct cv_Work_Type<CV_64F,CV_32F> : cv_Data_Type<CV_64F>{};
template<> struct cv_Work_Type<CV_64F,CV_64F> : cv_Data_Type<CV_64F>{};

//


template struct cv_Work_Type<CV_2U, CV_2U>;
template struct cv_Work_Type<CV_2U, CV_4U>;
template struct cv_Work_Type<CV_2U, CV_8U>;
template struct cv_Work_Type<CV_2U,CV_16U>;
template struct cv_Work_Type<CV_2U,CV_32U>;
template struct cv_Work_Type<CV_2U,CV_64U>;

template struct cv_Work_Type<CV_2U, CV_8S>;
template struct cv_Work_Type<CV_2U,CV_16S>;
template struct cv_Work_Type<CV_2U,CV_32S>;
template struct cv_Work_Type<CV_2U,CV_64S>;
template struct cv_Work_Type<CV_2U,CV_32F>;
template struct cv_Work_Type<CV_2U,CV_64F>;

template struct cv_Work_Type<CV_4U, CV_2U>;
template struct cv_Work_Type<CV_4U, CV_4U>;
template struct cv_Work_Type<CV_4U, CV_8U>;
template struct cv_Work_Type<CV_4U,CV_16U>;
template struct cv_Work_Type<CV_4U,CV_32U>;
template struct cv_Work_Type<CV_4U,CV_64U>;

template struct cv_Work_Type<CV_4U, CV_8S>;
template struct cv_Work_Type<CV_4U,CV_16S>;
template struct cv_Work_Type<CV_4U,CV_32S>;
template struct cv_Work_Type<CV_4U,CV_64S>;
template struct cv_Work_Type<CV_4U,CV_32F>;
template struct cv_Work_Type<CV_4U,CV_64F>;

template struct cv_Work_Type<CV_8U, CV_2U>;
template struct cv_Work_Type<CV_8U, CV_4U>;
template struct cv_Work_Type<CV_8U, CV_8U>;
template struct cv_Work_Type<CV_8U,CV_16U>;
template struct cv_Work_Type<CV_8U,CV_32U>;
template struct cv_Work_Type<CV_8U,CV_64U>;

template struct cv_Work_Type<CV_8U, CV_8S>;
template struct cv_Work_Type<CV_8U,CV_16S>;
template struct cv_Work_Type<CV_8U,CV_32S>;
template struct cv_Work_Type<CV_8U,CV_64S>;
template struct cv_Work_Type<CV_8U,CV_32F>;
template struct cv_Work_Type<CV_8U,CV_64F>;

template struct cv_Work_Type<CV_16U, CV_2U>;
template struct cv_Work_Type<CV_16U, CV_4U>;
template struct cv_Work_Type<CV_16U, CV_8U>;
template struct cv_Work_Type<CV_16U,CV_16U>;
template struct cv_Work_Type<CV_16U,CV_32U>;
template struct cv_Work_Type<CV_16U,CV_64U>;

template struct cv_Work_Type<CV_16U, CV_8S>;
template struct cv_Work_Type<CV_16U,CV_16S>;
template struct cv_Work_Type<CV_16U,CV_32S>;
template struct cv_Work_Type<CV_16U,CV_64S>;
template struct cv_Work_Type<CV_16U,CV_32F>;
template struct cv_Work_Type<CV_16U,CV_64F>;

template struct cv_Work_Type<CV_32U, CV_2U>;
template struct cv_Work_Type<CV_32U, CV_4U>;
template struct cv_Work_Type<CV_32U, CV_8U>;
template struct cv_Work_Type<CV_32U,CV_16U>;
template struct cv_Work_Type<CV_32U,CV_32U>;
template struct cv_Work_Type<CV_32U,CV_64U>;

template struct cv_Work_Type<CV_32U, CV_8S>;
template struct cv_Work_Type<CV_32U,CV_16S>;
template struct cv_Work_Type<CV_32U,CV_32S>;
template struct cv_Work_Type<CV_32U,CV_64S>;
template struct cv_Work_Type<CV_32U,CV_32F>;
template struct cv_Work_Type<CV_32U,CV_64F>;

template struct cv_Work_Type<CV_64U, CV_2U>;
template struct cv_Work_Type<CV_64U, CV_4U>;
template struct cv_Work_Type<CV_64U, CV_8U>;
template struct cv_Work_Type<CV_64U,CV_16U>;
template struct cv_Work_Type<CV_64U,CV_32U>;
template struct cv_Work_Type<CV_64U,CV_64U>;

template struct cv_Work_Type<CV_64U, CV_8S>;
template struct cv_Work_Type<CV_64U,CV_16S>;
template struct cv_Work_Type<CV_64U,CV_32S>;
template struct cv_Work_Type<CV_64U,CV_64S>;
template struct cv_Work_Type<CV_64U,CV_32F>;
template struct cv_Work_Type<CV_64U,CV_64F>;

template struct cv_Work_Type<CV_32F, CV_2U>;
template struct cv_Work_Type<CV_32F, CV_4U>;
template struct cv_Work_Type<CV_32F, CV_8U>;
template struct cv_Work_Type<CV_32F,CV_16U>;
template struct cv_Work_Type<CV_32F,CV_32U>;
template struct cv_Work_Type<CV_32F,CV_64U>;

template struct cv_Work_Type<CV_32F, CV_8S>;
template struct cv_Work_Type<CV_32F,CV_16S>;
template struct cv_Work_Type<CV_32F,CV_32S>;
template struct cv_Work_Type<CV_32F,CV_64S>;
template struct cv_Work_Type<CV_32F,CV_32F>;
template struct cv_Work_Type<CV_32F,CV_64F>;

template struct cv_Work_Type<CV_64F, CV_2U>;
template struct cv_Work_Type<CV_64F, CV_4U>;
template struct cv_Work_Type<CV_64F, CV_8U>;
template struct cv_Work_Type<CV_64F,CV_16U>;
template struct cv_Work_Type<CV_64F,CV_32U>;
template struct cv_Work_Type<CV_64F,CV_64U>;

template struct cv_Work_Type<CV_64F, CV_8S>;
template struct cv_Work_Type<CV_64F,CV_16S>;
template struct cv_Work_Type<CV_64F,CV_32S>;
template struct cv_Work_Type<CV_64F,CV_64S>;
template struct cv_Work_Type<CV_64F,CV_32F>;
template struct cv_Work_Type<CV_64F,CV_64F>;

// Signed Working types

template<int t1, int t2> struct cv_Signed_Work_Type;

template<> struct cv_Signed_Work_Type<CV_2U, CV_2U> : cv_Data_Type<CV_8S>{};
template<> struct cv_Signed_Work_Type<CV_2U, CV_4U> : cv_Data_Type<CV_8S>{};
template<> struct cv_Signed_Work_Type<CV_2U, CV_8U> : cv_Data_Type<CV_16S>{};
template<> struct cv_Signed_Work_Type<CV_2U,CV_16U> : cv_Data_Type<CV_32S>{};
template<> struct cv_Signed_Work_Type<CV_2U,CV_32U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_2U,CV_64U> : cv_Data_Type<CV_64S>{};

template<> struct cv_Signed_Work_Type<CV_2U, CV_8S> : cv_Data_Type<CV_16S>{};
template<> struct cv_Signed_Work_Type<CV_2U,CV_16S> : cv_Data_Type<CV_32S>{};
template<> struct cv_Signed_Work_Type<CV_2U,CV_32S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_2U,CV_64S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_2U,CV_32F> : cv_Data_Type<CV_64F>{};
template<> struct cv_Signed_Work_Type<CV_2U,CV_64F> : cv_Data_Type<CV_64F>{};

template<> struct cv_Signed_Work_Type<CV_4U, CV_2U> : cv_Data_Type<CV_8S>{};
template<> struct cv_Signed_Work_Type<CV_4U, CV_4U> : cv_Data_Type<CV_16S>{};
template<> struct cv_Signed_Work_Type<CV_4U, CV_8U> : cv_Data_Type<CV_32S>{};
template<> struct cv_Signed_Work_Type<CV_4U,CV_16U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_4U,CV_32U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_4U,CV_64U> : cv_Data_Type<CV_64S>{};

template<> struct cv_Signed_Work_Type<CV_4U, CV_8S> : cv_Data_Type<CV_32S>{};
template<> struct cv_Signed_Work_Type<CV_4U,CV_16S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_4U,CV_32S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_4U,CV_64S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_4U,CV_32F> : cv_Data_Type<CV_64F>{};
template<> struct cv_Signed_Work_Type<CV_4U,CV_64F> : cv_Data_Type<CV_64F>{};

template<> struct cv_Signed_Work_Type<CV_8U, CV_2U> : cv_Data_Type<CV_16S>{};
template<> struct cv_Signed_Work_Type<CV_8U, CV_4U> : cv_Data_Type<CV_32S>{};
template<> struct cv_Signed_Work_Type<CV_8U, CV_8U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_8U,CV_16U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_8U,CV_32U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_8U,CV_64U> : cv_Data_Type<CV_64S>{};

template<> struct cv_Signed_Work_Type<CV_8U, CV_8S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_8U,CV_16S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_8U,CV_32S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_8U,CV_64S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_8U,CV_32F> : cv_Data_Type<CV_64F>{};
template<> struct cv_Signed_Work_Type<CV_8U,CV_64F> : cv_Data_Type<CV_64F>{};

template<> struct cv_Signed_Work_Type<CV_8S, CV_2U> : cv_Data_Type<CV_16S>{};
template<> struct cv_Signed_Work_Type<CV_8S, CV_4U> : cv_Data_Type<CV_32S>{};
template<> struct cv_Signed_Work_Type<CV_8S, CV_8U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_8S,CV_16U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_8S,CV_32U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_8S,CV_64U> : cv_Data_Type<CV_64S>{};

template<> struct cv_Signed_Work_Type<CV_8S, CV_8S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_8S,CV_16S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_8S,CV_32S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_8S,CV_64S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_8S,CV_32F> : cv_Data_Type<CV_64F>{};
template<> struct cv_Signed_Work_Type<CV_8S,CV_64F> : cv_Data_Type<CV_64F>{};

template<> struct cv_Signed_Work_Type<CV_16U, CV_2U> : cv_Data_Type<CV_32S>{};
template<> struct cv_Signed_Work_Type<CV_16U, CV_4U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_16U, CV_8U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_16U,CV_16U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_16U,CV_32U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_16U,CV_64U> : cv_Data_Type<CV_64S>{};

template<> struct cv_Signed_Work_Type<CV_16U, CV_8S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_16U,CV_16S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_16U,CV_32S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_16U,CV_64S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_16U,CV_32F> : cv_Data_Type<CV_64F>{};
template<> struct cv_Signed_Work_Type<CV_16U,CV_64F> : cv_Data_Type<CV_64F>{};

template<> struct cv_Signed_Work_Type<CV_16S, CV_2U> : cv_Data_Type<CV_32S>{};
template<> struct cv_Signed_Work_Type<CV_16S, CV_4U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_16S, CV_8U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_16S,CV_16U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_16S,CV_32U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_16S,CV_64U> : cv_Data_Type<CV_64S>{};

template<> struct cv_Signed_Work_Type<CV_16S, CV_8S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_16S,CV_16S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_16S,CV_32S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_16S,CV_64S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_16S,CV_32F> : cv_Data_Type<CV_64F>{};
template<> struct cv_Signed_Work_Type<CV_16S,CV_64F> : cv_Data_Type<CV_64F>{};

template<> struct cv_Signed_Work_Type<CV_32U, CV_2U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_32U, CV_4U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_32U, CV_8U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_32U,CV_16U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_32U,CV_32U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_32U,CV_64U> : cv_Data_Type<CV_64S>{};

template<> struct cv_Signed_Work_Type<CV_32U, CV_8S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_32U,CV_16S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_32U,CV_32S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_32U,CV_64S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_32U,CV_32F> : cv_Data_Type<CV_64F>{};
template<> struct cv_Signed_Work_Type<CV_32U,CV_64F> : cv_Data_Type<CV_64F>{};

template<> struct cv_Signed_Work_Type<CV_32S, CV_2U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_32S, CV_4U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_32S, CV_8U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_32S,CV_16U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_32S,CV_32U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_32S,CV_64U> : cv_Data_Type<CV_64S>{};

template<> struct cv_Signed_Work_Type<CV_32S, CV_8S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_32S,CV_16S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_32S,CV_32S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_32S,CV_64S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_32S,CV_32F> : cv_Data_Type<CV_64F>{};
template<> struct cv_Signed_Work_Type<CV_32S,CV_64F> : cv_Data_Type<CV_64F>{};

template<> struct cv_Signed_Work_Type<CV_64U, CV_2U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_64U, CV_4U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_64U, CV_8U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_64U,CV_16U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_64U,CV_32U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_64U,CV_64U> : cv_Data_Type<CV_64S>{};

template<> struct cv_Signed_Work_Type<CV_64U, CV_8S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_64U,CV_16S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_64U,CV_32S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_64U,CV_64S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_64U,CV_32F> : cv_Data_Type<CV_64F>{};
template<> struct cv_Signed_Work_Type<CV_64U,CV_64F> : cv_Data_Type<CV_64F>{};

template<> struct cv_Signed_Work_Type<CV_64S, CV_2U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_64S, CV_4U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_64S, CV_8U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_64S,CV_16U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_64S,CV_32U> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_64S,CV_64U> : cv_Data_Type<CV_64S>{};

template<> struct cv_Signed_Work_Type<CV_64S, CV_8S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_64S,CV_16S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_64S,CV_32S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_64S,CV_64S> : cv_Data_Type<CV_64S>{};
template<> struct cv_Signed_Work_Type<CV_64S,CV_32F> : cv_Data_Type<CV_64F>{};
template<> struct cv_Signed_Work_Type<CV_64S,CV_64F> : cv_Data_Type<CV_64F>{};

template<> struct cv_Signed_Work_Type<CV_32F, CV_2U> : cv_Data_Type<CV_64F>{};
template<> struct cv_Signed_Work_Type<CV_32F, CV_4U> : cv_Data_Type<CV_64F>{};
template<> struct cv_Signed_Work_Type<CV_32F, CV_8U> : cv_Data_Type<CV_64F>{};
template<> struct cv_Signed_Work_Type<CV_32F,CV_16U> : cv_Data_Type<CV_64F>{};
template<> struct cv_Signed_Work_Type<CV_32F,CV_32U> : cv_Data_Type<CV_64F>{};
template<> struct cv_Signed_Work_Type<CV_32F,CV_64U> : cv_Data_Type<CV_64F>{};

template<> struct cv_Signed_Work_Type<CV_32F, CV_8S> : cv_Data_Type<CV_64F>{};
template<> struct cv_Signed_Work_Type<CV_32F,CV_16S> : cv_Data_Type<CV_64F>{};
template<> struct cv_Signed_Work_Type<CV_32F,CV_32S> : cv_Data_Type<CV_64F>{};
template<> struct cv_Signed_Work_Type<CV_32F,CV_64S> : cv_Data_Type<CV_64F>{};
template<> struct cv_Signed_Work_Type<CV_32F,CV_32F> : cv_Data_Type<CV_64F>{};
template<> struct cv_Signed_Work_Type<CV_32F,CV_64F> : cv_Data_Type<CV_64F>{};

template<> struct cv_Signed_Work_Type<CV_64F, CV_2U> : cv_Data_Type<CV_64F>{};
template<> struct cv_Signed_Work_Type<CV_64F, CV_4U> : cv_Data_Type<CV_64F>{};
template<> struct cv_Signed_Work_Type<CV_64F, CV_8U> : cv_Data_Type<CV_64F>{};
template<> struct cv_Signed_Work_Type<CV_64F,CV_16U> : cv_Data_Type<CV_64F>{};
template<> struct cv_Signed_Work_Type<CV_64F,CV_32U> : cv_Data_Type<CV_64F>{};
template<> struct cv_Signed_Work_Type<CV_64F,CV_64U> : cv_Data_Type<CV_64F>{};

template<> struct cv_Signed_Work_Type<CV_64F, CV_8S> : cv_Data_Type<CV_64F>{};
template<> struct cv_Signed_Work_Type<CV_64F,CV_16S> : cv_Data_Type<CV_64F>{};
template<> struct cv_Signed_Work_Type<CV_64F,CV_32S> : cv_Data_Type<CV_64F>{};
template<> struct cv_Signed_Work_Type<CV_64F,CV_64S> : cv_Data_Type<CV_64F>{};
template<> struct cv_Signed_Work_Type<CV_64F,CV_32F> : cv_Data_Type<CV_64F>{};
template<> struct cv_Signed_Work_Type<CV_64F,CV_64F> : cv_Data_Type<CV_64F>{};

// Instantiate templates

template struct cv_Signed_Work_Type<CV_2U, CV_2U>;
template struct cv_Signed_Work_Type<CV_2U, CV_4U>;
template struct cv_Signed_Work_Type<CV_2U, CV_8U>;
template struct cv_Signed_Work_Type<CV_2U,CV_16U>;
template struct cv_Signed_Work_Type<CV_2U,CV_32U>;
template struct cv_Signed_Work_Type<CV_2U,CV_64U>;

template struct cv_Signed_Work_Type<CV_2U, CV_8S>;
template struct cv_Signed_Work_Type<CV_2U,CV_16S>;
template struct cv_Signed_Work_Type<CV_2U,CV_32S>;
template struct cv_Signed_Work_Type<CV_2U,CV_64S>;
template struct cv_Signed_Work_Type<CV_2U,CV_32F>;
template struct cv_Signed_Work_Type<CV_2U,CV_64F>;

template struct cv_Signed_Work_Type<CV_4U, CV_2U>;
template struct cv_Signed_Work_Type<CV_4U, CV_4U>;
template struct cv_Signed_Work_Type<CV_4U, CV_8U>;
template struct cv_Signed_Work_Type<CV_4U,CV_16U>;
template struct cv_Signed_Work_Type<CV_4U,CV_32U>;
template struct cv_Signed_Work_Type<CV_4U,CV_64U>;

template struct cv_Signed_Work_Type<CV_4U, CV_8S>;
template struct cv_Signed_Work_Type<CV_4U,CV_16S>;
template struct cv_Signed_Work_Type<CV_4U,CV_32S>;
template struct cv_Signed_Work_Type<CV_4U,CV_64S>;
template struct cv_Signed_Work_Type<CV_4U,CV_32F>;
template struct cv_Signed_Work_Type<CV_4U,CV_64F>;

template struct cv_Signed_Work_Type<CV_8U, CV_2U>;
template struct cv_Signed_Work_Type<CV_8U, CV_4U>;
template struct cv_Signed_Work_Type<CV_8U, CV_8U>;
template struct cv_Signed_Work_Type<CV_8U,CV_16U>;
template struct cv_Signed_Work_Type<CV_8U,CV_32U>;
template struct cv_Signed_Work_Type<CV_8U,CV_64U>;

template struct cv_Signed_Work_Type<CV_8U, CV_8S>;
template struct cv_Signed_Work_Type<CV_8U,CV_16S>;
template struct cv_Signed_Work_Type<CV_8U,CV_32S>;
template struct cv_Signed_Work_Type<CV_8U,CV_64S>;
template struct cv_Signed_Work_Type<CV_8U,CV_32F>;
template struct cv_Signed_Work_Type<CV_8U,CV_64F>;

template struct cv_Signed_Work_Type<CV_16U, CV_2U>;
template struct cv_Signed_Work_Type<CV_16U, CV_4U>;
template struct cv_Signed_Work_Type<CV_16U, CV_8U>;
template struct cv_Signed_Work_Type<CV_16U,CV_16U>;
template struct cv_Signed_Work_Type<CV_16U,CV_32U>;
template struct cv_Signed_Work_Type<CV_16U,CV_64U>;

template struct cv_Signed_Work_Type<CV_16U, CV_8S>;
template struct cv_Signed_Work_Type<CV_16U,CV_16S>;
template struct cv_Signed_Work_Type<CV_16U,CV_32S>;
template struct cv_Signed_Work_Type<CV_16U,CV_64S>;
template struct cv_Signed_Work_Type<CV_16U,CV_32F>;
template struct cv_Signed_Work_Type<CV_16U,CV_64F>;

template struct cv_Signed_Work_Type<CV_32U, CV_2U>;
template struct cv_Signed_Work_Type<CV_32U, CV_4U>;
template struct cv_Signed_Work_Type<CV_32U, CV_8U>;
template struct cv_Signed_Work_Type<CV_32U,CV_16U>;
template struct cv_Signed_Work_Type<CV_32U,CV_32U>;
template struct cv_Signed_Work_Type<CV_32U,CV_64U>;

template struct cv_Signed_Work_Type<CV_32U, CV_8S>;
template struct cv_Signed_Work_Type<CV_32U,CV_16S>;
template struct cv_Signed_Work_Type<CV_32U,CV_32S>;
template struct cv_Signed_Work_Type<CV_32U,CV_64S>;
template struct cv_Signed_Work_Type<CV_32U,CV_32F>;
template struct cv_Signed_Work_Type<CV_32U,CV_64F>;

template struct cv_Signed_Work_Type<CV_64U, CV_2U>;
template struct cv_Signed_Work_Type<CV_64U, CV_4U>;
template struct cv_Signed_Work_Type<CV_64U, CV_8U>;
template struct cv_Signed_Work_Type<CV_64U,CV_16U>;
template struct cv_Signed_Work_Type<CV_64U,CV_32U>;
template struct cv_Signed_Work_Type<CV_64U,CV_64U>;

template struct cv_Signed_Work_Type<CV_64U, CV_8S>;
template struct cv_Signed_Work_Type<CV_64U,CV_16S>;
template struct cv_Signed_Work_Type<CV_64U,CV_32S>;
template struct cv_Signed_Work_Type<CV_64U,CV_64S>;
template struct cv_Signed_Work_Type<CV_64U,CV_32F>;
template struct cv_Signed_Work_Type<CV_64U,CV_64F>;

template struct cv_Signed_Work_Type<CV_32F, CV_2U>;
template struct cv_Signed_Work_Type<CV_32F, CV_4U>;
template struct cv_Signed_Work_Type<CV_32F, CV_8U>;
template struct cv_Signed_Work_Type<CV_32F,CV_16U>;
template struct cv_Signed_Work_Type<CV_32F,CV_32U>;
template struct cv_Signed_Work_Type<CV_32F,CV_64U>;

template struct cv_Signed_Work_Type<CV_32F, CV_8S>;
template struct cv_Signed_Work_Type<CV_32F,CV_16S>;
template struct cv_Signed_Work_Type<CV_32F,CV_32S>;
template struct cv_Signed_Work_Type<CV_32F,CV_64S>;
template struct cv_Signed_Work_Type<CV_32F,CV_32F>;
template struct cv_Signed_Work_Type<CV_32F,CV_64F>;

template struct cv_Signed_Work_Type<CV_64F, CV_2U>;
template struct cv_Signed_Work_Type<CV_64F, CV_4U>;
template struct cv_Signed_Work_Type<CV_64F, CV_8U>;
template struct cv_Signed_Work_Type<CV_64F,CV_16U>;
template struct cv_Signed_Work_Type<CV_64F,CV_32U>;
template struct cv_Signed_Work_Type<CV_64F,CV_64U>;

template struct cv_Signed_Work_Type<CV_64F, CV_8S>;
template struct cv_Signed_Work_Type<CV_64F,CV_16S>;
template struct cv_Signed_Work_Type<CV_64F,CV_32S>;
template struct cv_Signed_Work_Type<CV_64F,CV_64S>;
template struct cv_Signed_Work_Type<CV_64F,CV_32F>;
template struct cv_Signed_Work_Type<CV_64F,CV_64F>;

template<int cv_data_type> using cv_Type = typename cv_Data_Type<cv_data_type>::type;

// Don't use cv_Data_Type directly; use Data_Type which works for both types and types with channels.

namespace cv {
    template<int t> struct Data_Type : cv_Data_Type<CV_MAT_DEPTH(t)>{
        constexpr static int channels  = CV_MAT_CN(t);
    };
    template<int t1,int t2> struct Work_Type : cv_Work_Type<CV_MAT_DEPTH(t1),CV_MAT_DEPTH(t2)>{
        constexpr static int channels  = CV_MAT_CN(t1) + CV_MAT_CN(t2);
    };
    template<int t1,int t2> struct Signed_Work_Type : cv_Signed_Work_Type<CV_MAT_DEPTH(t1),CV_MAT_DEPTH(t2)>{
        constexpr static int channels  = CV_MAT_CN(t1) + CV_MAT_CN(t2);
    };


}

///////////////////////////// Bitwise and discrete math operations ///////////////////////////

template<typename _Tp> _Tp gComDivisor(_Tp u, _Tp v) {
    if (v)
        return gComDivisor<_Tp>(v, u % v);
    else
        return u < 0 ? -u : u; /* abs(u) */
};

template<typename _Tp> _Tp gComDivisor(_Tp a, _Tp b, _Tp c){
    return gComDivisor<_Tp>(gComDivisor<_Tp>(a, b), c);
};


template<typename _Tp> _Tp gComDivisor(_Tp a, _Tp* b, unsigned int size_b){
    if (size_b >= 2){
        gComDivisor<_Tp>(a, b[0]);
        return gComDivisor<_Tp>(gComDivisor<_Tp>(a, b[0]), b++, size_b-1);
    }
    else if(size_b == 1) {
        return gComDivisor<_Tp>(a, b[0]);
    }
    else {
        return a;
    }
};

template<typename _Tp> _Tp gComDivisor(_Tp* b, unsigned int size_b){
    switch (size_b) {
        case 0:
            return _Tp();
            break;
        case 1:
            return b[0];
            break;
        case 2:
            return gComDivisor<_Tp>(b[0],b[1]);
            break;
        case 3:
            return gComDivisor<_Tp>(gComDivisor<_Tp>(b[0],b[1]),b[2]);
            break;
        case 4:
            return gComDivisor<_Tp>(gComDivisor<_Tp>(b[0],b[1]), gComDivisor<_Tp>(b[2],b[3]));
            break;
        default:
            return gComDivisor<_Tp>(gComDivisor<_Tp>(b,size_b/2), gComDivisor<_Tp>(b+(size_b)/2,(size_b+1)/2));
            break;
    }
};

unsigned int CV_INLINE mostSignificantBit(uint64_t x)
{
    static const unsigned int bval[] = {0,1,2,2,3,3,3,3,4,4,4,4,4,4,4,4};
    unsigned int r = 0;
    if (x & 0xFFFFFFFF00000000) { r += 32/1; x >>= 32/1; }
    if (x & 0x00000000FFFF0000) { r += 32/2; x >>= 32/2; }
    if (x & 0x000000000000FF00) { r += 32/4; x >>= 32/4; }
    if (x & 0x00000000000000F0) { r += 32/8; x >>= 32/8; }
    return r + bval[x];
}
unsigned int CV_INLINE  mostSignificantBit(uint32_t x)
{
    static const unsigned int bval[] = {0,1,2,2,3,3,3,3,4,4,4,4,4,4,4,4};
    unsigned int r = 0;
    if (x & 0xFFFF0000) { r += 16/1; x >>= 16/1; }
    if (x & 0x0000FF00) { r += 16/2; x >>= 16/2; }
    if (x & 0x000000F0) { r += 16/4; x >>= 16/4; }
    return r + bval[x];
}

unsigned int CV_INLINE  mostSignificantBit(uint16_t x)
{
    static const unsigned int bval[] = {0,1,2,2,3,3,3,3,4,4,4,4,4,4,4,4};
    unsigned int r = 0;
    if (x & 0xFF00) { r += 8/1; x >>= 8/1; }
    if (x & 0x00F0) { r += 8/2; x >>= 8/2; }
    return r + bval[x];
}

unsigned int CV_INLINE  mostSignificantBit(uint8_t x)
{
    static const unsigned int bval[] = {0,1,2,2,3,3,3,3,4,4,4,4,4,4,4,4};
    unsigned int r = 0;
    if (x & 0xF0) { r += 4/1; x >>= 4/1; }
    return r + bval[x];
}

unsigned int CV_INLINE  mostSignificantBit(int64_t x)
{
    static const unsigned int bval[] = {0,1,2,2,3,3,3,3,4,4,4,4,4,4,4,4};
    unsigned int r = 0;
    if (x & 0x7FFFFFFF00000000) { r += 32/1; x >>= 32/1; }
    if (x & 0x00000000FFFF0000) { r += 32/2; x >>= 32/2; }
    if (x & 0x000000000000FF00) { r += 32/4; x >>= 32/4; }
    if (x & 0x00000000000000F0) { r += 32/8; x >>= 32/8; }
    return r + bval[x];
}
unsigned int CV_INLINE  mostSignificantBit(int32_t x)
{
    static const unsigned int bval[] = {0,1,2,2,3,3,3,3,4,4,4,4,4,4,4,4};
    unsigned int r = 0;
    if (x & 0x7FFF0000) { r += 16/1; x >>= 16/1; }
    if (x & 0x0000FF00) { r += 16/2; x >>= 16/2; }
    if (x & 0x000000F0) { r += 16/4; x >>= 16/4; }
    return r + bval[x];
}

unsigned int CV_INLINE  mostSignificantBit(int16_t x)
{
    static const unsigned int bval[] = {0,1,2,2,3,3,3,3,4,4,4,4,4,4,4,4};
    unsigned int r = 0;
    if (x & 0x7F00) { r += 8/1; x >>= 8/1; }
    if (x & 0x00F0) { r += 8/2; x >>= 8/2; }
    return r + bval[x];
}

unsigned int CV_INLINE  mostSignificantBit(int8_t x)
{
    static const unsigned int bval[] = {0,1,2,2,3,3,3,3,4,4,4,4,4,4,4,4};
    unsigned int r = 0;
    if (x & 0x70) { r += 4/1; x >>= 4/1; }
    return r + bval[x];
}


/* f : number to convert.
 * num, denom: returned parts of the rational.
 * max_denom: max denominator value.  Note that machine floating point number
 *     has a finite resolution (10e-16 ish for 64 bit double), so specifying
 *     a "best match with minimal error" is often wrong, because one can
 *     always just retrieve the significand and return that divided by
 *     2**52, which is in a sense accurate, but generally not very useful:
 *     1.0/7.0 would be "2573485501354569/18014398509481984", for example.
 */
void CV_INLINE rat_approx(double f, int64_t max_denom, int64_t *num, int64_t *denom)
{
    /*  a: continued fraction coefficients. */
    int64_t a, h[3] = { 0, 1, 0 }, k[3] = { 1, 0, 0 };
    int64_t x, d, n = 1;
    int i, neg = 0;

    if (max_denom <= 1) { *denom = 1; *num = (int64_t) f; return; }

    if (f < 0) { neg = 1; f = -f; }

    while (f != floor(f)) { n <<= 1; f *= 2; }
    d = f;

    /* continued fraction and check denominator each step */
    for (i = 0; i < 64; i++) {
        a = n ? d / n : 0;
        if (i && !a) break;

        x = d; d = n; n = x % n;

        x = a;
        if (k[1] * a + k[0] >= max_denom) {
            x = (max_denom - k[0]) / k[1];
            if (x * 2 >= a || k[1] >= max_denom)
                i = 65;
            else
                break;
        }

        h[2] = x * h[1] + h[0]; h[0] = h[1]; h[1] = h[2];
        k[2] = x * k[1] + k[0]; k[0] = k[1]; k[1] = k[2];
    }
    *denom = k[1];
    *num = neg ? -h[1] : h[1];
}


#  endif


#endif
