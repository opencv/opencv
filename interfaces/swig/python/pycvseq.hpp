/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

#ifndef PY_CV_SEQ_H
#define PY_CV_SEQ_H

#include "cxcore.h"
#include "cvtypes.h"


/** class to make sequence iteration nicer */
template<class T>
class CvTypedSeqReader : public CvSeqReader {
    int pos;
public:
    CvTypedSeqReader( const CvSeq * seq ){
        cvStartReadSeq( seq, this );
        pos = 0;
    }
    T * current(){
        return (T*) this->ptr;
    }
    void next(){
        CV_NEXT_SEQ_ELEM( this->seq->elem_size, *this );
        pos++;
    }
    bool valid(){
        return pos<this->seq->total;
    }
};

template<class T>
class CvTypedSeq : public CvSeq {
public:
	static CvTypedSeq<T> * cast(CvSeq * seq){
		return (CvTypedSeq<T> *) seq;
	}
	T * __getitem__ (int i){
		return (T *) cvGetSeqElem(this, i);
	}
	void __setitem__ (int i, T * obj){
		T * ptr = this->__getitem__(i);
		memcpy(ptr, obj, sizeof(T));
	}
	void append(T * obj){
		cvSeqPush( this, obj );
	}
	T * pop(){
		T * obj = new T;
		cvSeqPop( this, obj );
		return obj;
	}
};

template<class T, int size=2>
struct CvTuple {
	T val[2];
	void __setitem__(int i, T * obj){
		val[i] = *obj;
	}
	const T & __getitem__(int i){
		return val[i];
	}
};

#endif  //PY_CV_SEQ_H
