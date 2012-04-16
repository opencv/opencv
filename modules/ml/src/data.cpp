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

#include "precomp.hpp"
#include <ctype.h>

#define MISS_VAL    FLT_MAX 
#define CV_VAR_MISS    0

CvTrainTestSplit::CvTrainTestSplit()
{
    train_sample_part_mode = CV_COUNT;
    train_sample_part.count = -1;
    mix = false;
}

CvTrainTestSplit::CvTrainTestSplit( int _train_sample_count, bool _mix )
{
    train_sample_part_mode = CV_COUNT;
    train_sample_part.count = _train_sample_count;
    mix = _mix;
}
    
CvTrainTestSplit::CvTrainTestSplit( float _train_sample_portion, bool _mix )
{
    train_sample_part_mode = CV_PORTION;
    train_sample_part.portion = _train_sample_portion;
    mix = _mix;
}

////////////////

CvMLData::CvMLData()
{
    values = missing = var_types = var_idx_mask = response_out = var_idx_out = var_types_out = 0;
    train_sample_idx = test_sample_idx = 0;
    sample_idx = 0;
    response_idx = -1;

    train_sample_count = -1;

    delimiter = ',';
    miss_ch = '?';
    //flt_separator = '.';

    rng = &cv::theRNG();
}

CvMLData::~CvMLData()
{
    clear();
}

void CvMLData::free_train_test_idx()
{
    cvReleaseMat( &train_sample_idx );
    cvReleaseMat( &test_sample_idx );
    sample_idx = 0;
}

void CvMLData::clear()
{
    class_map.clear();

    cvReleaseMat( &values );
    cvReleaseMat( &missing );
    cvReleaseMat( &var_types );
    cvReleaseMat( &var_idx_mask );

    cvReleaseMat( &response_out );
    cvReleaseMat( &var_idx_out );
    cvReleaseMat( &var_types_out );

    free_train_test_idx();
    
    total_class_count = 0;

    response_idx = -1;

    train_sample_count = -1;
}

static char *fgets_chomp(char *str, int n, FILE *stream)
{
	char *head = fgets(str, n, stream);
	if( head )
	{
		for(char *tail = head + strlen(head) - 1; tail >= head; --tail)
		{
			if( *tail != '\r'  && *tail != '\n' )
				break;
			*tail = '\0';
		}
	}
	return head;
}


int CvMLData::read_csv(const char* filename)
{
    const int M = 1000000;
    const char str_delimiter[3] = { ' ', delimiter, '\0' };
    FILE* file = 0;
    CvMemStorage* storage;
    CvSeq* seq;
    char *ptr;
    float* el_ptr;
    CvSeqReader reader;
    int cols_count = 0;    
    uchar *var_types_ptr = 0;

    clear();

    file = fopen( filename, "rt" );
    
    if( !file )
        return -1;

    // read the first line and determine the number of variables
    std::vector<char> _buf(M);
    char* buf = &_buf[0];
    if( !fgets_chomp( buf, M, file ))
    {
        fclose(file);
        return -1;
    }

    ptr = buf;
    while( *ptr == ' ' )
        ptr++;
    for( ; *ptr != '\0'; )
    {
        if(*ptr == delimiter || *ptr == ' ')
        {
            cols_count++;
            ptr++;
            while( *ptr == ' ' ) ptr++;
        }
        else
            ptr++;
    }

    if ( cols_count == 0)
    {
        fclose(file);
        return -1;
    }
    cols_count++;

    // create temporary memory storage to store the whole database
    el_ptr = new float[cols_count];
    storage = cvCreateMemStorage();
    seq = cvCreateSeq( 0, sizeof(*seq), cols_count*sizeof(float), storage );

    var_types = cvCreateMat( 1, cols_count, CV_8U );
    cvZero( var_types );
    var_types_ptr = var_types->data.ptr;

    for(;;)
    {
        char *token = NULL;
        int type;
        token = strtok(buf, str_delimiter);
        if (!token) 
        {
             fclose(file);
             return -1;
        }
        for (int i = 0; i < cols_count-1; i++)
        {
            str_to_flt_elem( token, el_ptr[i], type);
            var_types_ptr[i] |= type;
            token = strtok(NULL, str_delimiter);
            if (!token)
            {
                fclose(file);
                return -1;
            }
        }
        str_to_flt_elem( token, el_ptr[cols_count-1], type);
        var_types_ptr[cols_count-1] |= type;
        cvSeqPush( seq, el_ptr );
        if( !fgets_chomp( buf, M, file ) || !strchr( buf, delimiter ) )
            break;
    }
    fclose(file);

    values = cvCreateMat( seq->total, cols_count, CV_32FC1 );
    missing = cvCreateMat( seq->total, cols_count, CV_8U );
    var_idx_mask = cvCreateMat( 1, values->cols, CV_8UC1 );
    cvSet( var_idx_mask, cvRealScalar(1) );
    train_sample_count = seq->total;

    cvStartReadSeq( seq, &reader );
    for(int i = 0; i < seq->total; i++ )
    {
        const float* sdata = (float*)reader.ptr;
        float* ddata = values->data.fl + cols_count*i;
        uchar* dm = missing->data.ptr + cols_count*i;

        for( int j = 0; j < cols_count; j++ )
        {
            ddata[j] = sdata[j];
            dm[j] = ( fabs( MISS_VAL - sdata[j] ) <= FLT_EPSILON );
        }
        CV_NEXT_SEQ_ELEM( seq->elem_size, reader );
    }

    if ( cvNorm( missing, 0, CV_L1 ) <= FLT_EPSILON )
        cvReleaseMat( &missing );

    cvReleaseMemStorage( &storage );
    delete []el_ptr;
    return 0;
}

const CvMat* CvMLData::get_values() const
{
    return values;
}

const CvMat* CvMLData::get_missing() const
{
    CV_FUNCNAME( "CvMLData::get_missing" );
    __BEGIN__;

    if ( !values )
        CV_ERROR( CV_StsInternal, "data is empty" );

    __END__;

    return missing;
}

const std::map<std::string, int>& CvMLData::get_class_labels_map() const
{
    return class_map;
}

void CvMLData::str_to_flt_elem( const char* token, float& flt_elem, int& type)
{
    
    char* stopstring = NULL;
    flt_elem = (float)strtod( token, &stopstring );
    assert( stopstring );
    type = CV_VAR_ORDERED;
    if ( *stopstring == miss_ch && strlen(stopstring) == 1 ) // missed value
    {
        flt_elem = MISS_VAL;
        type = CV_VAR_MISS;
    }
    else
    {
        if ( (*stopstring != 0) && (*stopstring != '\n') && (strcmp(stopstring, "\r\n") != 0) ) // class label
        {
            int idx = class_map[token];
            if ( idx == 0)
            {
                total_class_count++;
                idx = total_class_count;
                class_map[token] = idx;
            }
            flt_elem = (float)idx;
            type = CV_VAR_CATEGORICAL;
        }
    }
}

void CvMLData::set_delimiter(char ch)
{
    CV_FUNCNAME( "CvMLData::set_delimited" );
    __BEGIN__;

    if (ch == miss_ch /*|| ch == flt_separator*/)
        CV_ERROR(CV_StsBadArg, "delimited, miss_character and flt_separator must be different");
    
    delimiter = ch;

    __END__;
}

char CvMLData::get_delimiter() const
{
    return delimiter;
}

void CvMLData::set_miss_ch(char ch)
{
    CV_FUNCNAME( "CvMLData::set_miss_ch" );
    __BEGIN__;

    if (ch == delimiter/* || ch == flt_separator*/)
        CV_ERROR(CV_StsBadArg, "delimited, miss_character and flt_separator must be different");
   
    miss_ch = ch;

    __END__;
}

char CvMLData::get_miss_ch() const
{
    return miss_ch;
}

void CvMLData::set_response_idx( int idx )
{
    CV_FUNCNAME( "CvMLData::set_response_idx" );
    __BEGIN__;

    if ( !values )
        CV_ERROR( CV_StsInternal, "data is empty" );

    if ( idx >= values->cols)
        CV_ERROR( CV_StsBadArg, "idx value is not correct" );

    if ( response_idx >= 0 )
        chahge_var_idx( response_idx, true );
    if ( idx >= 0 )
        chahge_var_idx( idx, false );
    response_idx = idx;

    __END__;    
}

int CvMLData::get_response_idx() const
{
    CV_FUNCNAME( "CvMLData::get_response_idx" );
    __BEGIN__;

    if ( !values )
        CV_ERROR( CV_StsInternal, "data is empty" );
     __END__;
    return response_idx;
}

void CvMLData::change_var_type( int var_idx, int type )
{
    CV_FUNCNAME( "CvMLData::change_var_type" );
    __BEGIN__;
    
    int var_count = 0;

    if ( !values )
        CV_ERROR( CV_StsInternal, "data is empty" );
    
     var_count = values->cols;

    if ( var_idx < 0 || var_idx >= var_count)
        CV_ERROR( CV_StsBadArg, "var_idx is not correct" );

    if ( type != CV_VAR_ORDERED && type != CV_VAR_CATEGORICAL)
         CV_ERROR( CV_StsBadArg, "type is not correct" );

    assert( var_types );    
    if ( var_types->data.ptr[var_idx] == CV_VAR_CATEGORICAL && type == CV_VAR_ORDERED)
        CV_ERROR( CV_StsBadArg, "it`s impossible to assign CV_VAR_ORDERED type to categorical variable" );
    var_types->data.ptr[var_idx] = (uchar)type;

    __END__;

    return;
}

void CvMLData::set_var_types( const char* str )
{
    CV_FUNCNAME( "CvMLData::set_var_types" );
    __BEGIN__;

    const char* ord = 0, *cat = 0;
    int var_count = 0, set_var_type_count = 0;
    if ( !values )
        CV_ERROR( CV_StsInternal, "data is empty" );

    var_count = values->cols;

    assert( var_types );
 
    ord = strstr( str, "ord" );
    cat = strstr( str, "cat" );    
    if ( !ord && !cat )
        CV_ERROR( CV_StsBadArg, "types string is not correct" );
    
    if ( !ord && strlen(cat) == 3 ) // str == "cat"
    {
        cvSet( var_types, cvScalarAll(CV_VAR_CATEGORICAL) );
        return;
    }

    if ( !cat && strlen(ord) == 3 ) // str == "ord"
    {
        cvSet( var_types, cvScalarAll(CV_VAR_ORDERED) );
        return;
    }

    if ( ord ) // parse ord str
    {
        char* stopstring = NULL;            
        if ( ord[3] != '[')
            CV_ERROR( CV_StsBadArg, "types string is not correct" );
        
        ord += 4; // pass "ord["
        do
        {
            int b1 = (int)strtod( ord, &stopstring );
            if ( *stopstring == 0 || (*stopstring != ',' && *stopstring != ']' && *stopstring != '-') )
                CV_ERROR( CV_StsBadArg, "types string is not correct" );
            ord = stopstring + 1;
            if ( (stopstring[0] == ',') || (stopstring[0] == ']'))
            {
                if ( var_types->data.ptr[b1] == CV_VAR_CATEGORICAL)
                    CV_ERROR( CV_StsBadArg, "it`s impossible to assign CV_VAR_ORDERED type to categorical variable" );
                var_types->data.ptr[b1] = CV_VAR_ORDERED;
                set_var_type_count++;
            }
            else 
            {
                if ( stopstring[0] == '-') 
                {
                    int b2 = (int)strtod( ord, &stopstring);
                    if ( (*stopstring == 0) || (*stopstring != ',' && *stopstring != ']') )
                        CV_ERROR( CV_StsBadArg, "types string is not correct" );           
                    ord = stopstring + 1;
                    for (int i = b1; i <= b2; i++)
                    {
                        if ( var_types->data.ptr[i] == CV_VAR_CATEGORICAL)
                            CV_ERROR( CV_StsBadArg, "it`s impossible to assign CV_VAR_ORDERED type to categorical variable" );                
                        var_types->data.ptr[i] = CV_VAR_ORDERED;
                    }
                    set_var_type_count += b2 - b1 + 1;
                }
                else
                    CV_ERROR( CV_StsBadArg, "types string is not correct" );

            }
        }
        while (*stopstring != ']');

        if ( stopstring[1] != '\0' && stopstring[1] != ',')
            CV_ERROR( CV_StsBadArg, "types string is not correct" );
    }    

    if ( cat ) // parse cat str
    {
        char* stopstring = NULL;            
        if ( cat[3] != '[')
            CV_ERROR( CV_StsBadArg, "types string is not correct" );
        
        cat += 4; // pass "cat["
        do
        {
            int b1 = (int)strtod( cat, &stopstring );
            if ( *stopstring == 0 || (*stopstring != ',' && *stopstring != ']' && *stopstring != '-') )
                CV_ERROR( CV_StsBadArg, "types string is not correct" );
            cat = stopstring + 1;
            if ( (stopstring[0] == ',') || (stopstring[0] == ']'))
            {
                var_types->data.ptr[b1] = CV_VAR_CATEGORICAL;
                set_var_type_count++;
            }
            else 
            {
                if ( stopstring[0] == '-') 
                {
                    int b2 = (int)strtod( cat, &stopstring);
                    if ( (*stopstring == 0) || (*stopstring != ',' && *stopstring != ']') )
                        CV_ERROR( CV_StsBadArg, "types string is not correct" );           
                    cat = stopstring + 1;
                    for (int i = b1; i <= b2; i++)
                        var_types->data.ptr[i] = CV_VAR_CATEGORICAL;
                    set_var_type_count += b2 - b1 + 1;
                }
                else
                    CV_ERROR( CV_StsBadArg, "types string is not correct" );

            }
        }
        while (*stopstring != ']');

        if ( stopstring[1] != '\0' && stopstring[1] != ',')
            CV_ERROR( CV_StsBadArg, "types string is not correct" );
    }    

    if (set_var_type_count != var_count)
        CV_ERROR( CV_StsBadArg, "types string is not correct" );

     __END__;
}

const CvMat* CvMLData::get_var_types()
{
    CV_FUNCNAME( "CvMLData::get_var_types" );
    __BEGIN__;

    uchar *var_types_out_ptr = 0;
    int avcount, vt_size;
    if ( !values )
        CV_ERROR( CV_StsInternal, "data is empty" );

    assert( var_idx_mask );

    avcount = cvFloor( cvNorm( var_idx_mask, 0, CV_L1 ) );
    vt_size = avcount + (response_idx >= 0);

    if ( avcount == values->cols || (avcount == values->cols-1 && response_idx == values->cols-1) )
        return var_types;

    if ( !var_types_out || ( var_types_out && var_types_out->cols != vt_size ) ) 
    {
        cvReleaseMat( &var_types_out );
        var_types_out = cvCreateMat( 1, vt_size, CV_8UC1 );
    }

    var_types_out_ptr = var_types_out->data.ptr;
    for( int i = 0; i < var_types->cols; i++)
    {
        if (i == response_idx || !var_idx_mask->data.ptr[i]) continue;
        *var_types_out_ptr = var_types->data.ptr[i];
        var_types_out_ptr++;
    }
    if ( response_idx >= 0 )
        *var_types_out_ptr = var_types->data.ptr[response_idx];

    __END__;

    return var_types_out;
}

int CvMLData::get_var_type( int var_idx ) const
{
    return var_types->data.ptr[var_idx];
}

const CvMat* CvMLData::get_responses()
{
    CV_FUNCNAME( "CvMLData::get_responses_ptr" );
    __BEGIN__;

    int var_count = 0;

    if ( !values )
        CV_ERROR( CV_StsInternal, "data is empty" );
    var_count = values->cols;
    
    if ( response_idx < 0 || response_idx >= var_count )
       return 0;
    if ( !response_out )
        response_out = cvCreateMatHeader( values->rows, 1, CV_32FC1 );
    else
        cvInitMatHeader( response_out, values->rows, 1, CV_32FC1);
    cvGetCol( values, response_out, response_idx );

    __END__;

    return response_out;
}

void CvMLData::set_train_test_split( const CvTrainTestSplit * spl)
{
    CV_FUNCNAME( "CvMLData::set_division" );
    __BEGIN__;

    int sample_count = 0;

    if ( !values )
        CV_ERROR( CV_StsInternal, "data is empty" );

    sample_count = values->rows;
    
    float train_sample_portion;

    if (spl->train_sample_part_mode == CV_COUNT)
    {
        train_sample_count = spl->train_sample_part.count;
        if (train_sample_count > sample_count)
            CV_ERROR( CV_StsBadArg, "train samples count is not correct" );
        train_sample_count = train_sample_count<=0 ? sample_count : train_sample_count;
    }
    else // dtype.train_sample_part_mode == CV_PORTION
    {
        train_sample_portion = spl->train_sample_part.portion;
        if ( train_sample_portion > 1)
            CV_ERROR( CV_StsBadArg, "train samples count is not correct" );
        train_sample_portion = train_sample_portion <= FLT_EPSILON || 
            1 - train_sample_portion <= FLT_EPSILON ? 1 : train_sample_portion;
        train_sample_count = std::max(1, cvFloor( train_sample_portion * sample_count ));
    }

    if ( train_sample_count == sample_count )
    {
        free_train_test_idx();
        return;
    }

    if ( train_sample_idx && train_sample_idx->cols != train_sample_count )
        free_train_test_idx();

    if ( !sample_idx)
    {
        int test_sample_count = sample_count- train_sample_count;
        sample_idx = (int*)cvAlloc( sample_count * sizeof(sample_idx[0]) );
        for (int i = 0; i < sample_count; i++ )
            sample_idx[i] = i;
        train_sample_idx = cvCreateMatHeader( 1, train_sample_count, CV_32SC1 );
        *train_sample_idx = cvMat( 1, train_sample_count, CV_32SC1, &sample_idx[0] );

        CV_Assert(test_sample_count > 0);
        test_sample_idx = cvCreateMatHeader( 1, test_sample_count, CV_32SC1 );
        *test_sample_idx = cvMat( 1, test_sample_count, CV_32SC1, &sample_idx[train_sample_count] );
    }
    
    mix = spl->mix;
    if ( mix )
        mix_train_and_test_idx();
    
    __END__;
}

const CvMat* CvMLData::get_train_sample_idx() const
{
    CV_FUNCNAME( "CvMLData::get_train_sample_idx" );
    __BEGIN__;

    if ( !values )
        CV_ERROR( CV_StsInternal, "data is empty" );
    __END__;

    return train_sample_idx;
}

const CvMat* CvMLData::get_test_sample_idx() const
{
    CV_FUNCNAME( "CvMLData::get_test_sample_idx" );
    __BEGIN__;

    if ( !values )
        CV_ERROR( CV_StsInternal, "data is empty" );
    __END__;

    return test_sample_idx;
}

void CvMLData::mix_train_and_test_idx()
{
    CV_FUNCNAME( "CvMLData::mix_train_and_test_idx" );
    __BEGIN__;

    if ( !values )
        CV_ERROR( CV_StsInternal, "data is empty" );
    __END__;

    if ( !sample_idx)
        return;

    if ( train_sample_count > 0 && train_sample_count < values->rows )
    {
        int n = values->rows;
        for (int i = 0; i < n; i++)
        {
            int a = (*rng)(n);
            int b = (*rng)(n);
            int t;
            CV_SWAP( sample_idx[a], sample_idx[b], t );
        }
    }
}

const CvMat* CvMLData::get_var_idx()
{
     CV_FUNCNAME( "CvMLData::get_var_idx" );
    __BEGIN__;

    int avcount = 0;

    if ( !values )
        CV_ERROR( CV_StsInternal, "data is empty" );

    assert( var_idx_mask );
    
    avcount = cvFloor( cvNorm( var_idx_mask, 0, CV_L1 ) );
    int* vidx;

    if ( avcount == values->cols )
        return 0;
     
    if ( !var_idx_out || ( var_idx_out && var_idx_out->cols != avcount ) ) 
    {
        cvReleaseMat( &var_idx_out );
        var_idx_out = cvCreateMat( 1, avcount, CV_32SC1);
        if ( response_idx >=0 )
            var_idx_mask->data.ptr[response_idx] = 0;
    }

    vidx = var_idx_out->data.i;
    
    for(int i = 0; i < var_idx_mask->cols; i++)
        if ( var_idx_mask->data.ptr[i] )
        {            
            *vidx = i;
            vidx++;
        }

    __END__;

    return var_idx_out;
}

void CvMLData::chahge_var_idx( int vi, bool state )
{
    change_var_idx( vi, state );
}

void CvMLData::change_var_idx( int vi, bool state )
{
     CV_FUNCNAME( "CvMLData::change_var_idx" );
    __BEGIN__;

    int var_count = 0;

    if ( !values )
        CV_ERROR( CV_StsInternal, "data is empty" );

    var_count = values->cols;

    if ( vi < 0 || vi >= var_count)
        CV_ERROR( CV_StsBadArg, "variable index is not correct" );

    assert( var_idx_mask );    
    var_idx_mask->data.ptr[vi] = state;

    __END__;
}

/* End of file. */
