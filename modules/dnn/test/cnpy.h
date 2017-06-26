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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

//Copyright (C) 2011  Carl Rogers
//Released under MIT License
//license available in LICENSE file, or at http://www.opensource.org/licenses/mit-license.php

#ifndef LIBCNPY_H_
#define LIBCNPY_H_

#include<string>
#include<stdexcept>
#include<sstream>
#include<vector>
#include<cstdio>
#include<typeinfo>
#include<iostream>
#include<cassert>
#include<map>
#if defined(HAVE_ZLIB) && HAVE_ZLIB
#include<zlib.h>
#endif

#ifndef NDEBUG
#define cnpy_assert(expression) assert(expression)
#else
#define cnpy_assert(expression) ((void)(expression))
#endif

namespace cnpy {

    struct NpyArray {
        char* data;
        std::vector<unsigned int> shape;
        unsigned int word_size;
        bool fortran_order;
        void destruct() {delete[] data;}
    };

    struct npz_t : public std::map<std::string, NpyArray>
    {
        void destruct()
        {
            npz_t::iterator it = this->begin();
            for(; it != this->end(); ++it) (*it).second.destruct();
        }
    };

    char BigEndianTest();
    char map_type(const std::type_info& t);
    template<typename T> std::vector<char> create_npy_header(const T* data, const unsigned int* shape, const unsigned int ndims);
    void parse_npy_header(FILE* fp,unsigned int& word_size, unsigned int*& shape, unsigned int& ndims, bool& fortran_order);
    void parse_zip_footer(FILE* fp, unsigned short& nrecs, unsigned int& global_header_size, unsigned int& global_header_offset);
    npz_t npz_load(std::string fname);
    NpyArray npz_load(std::string fname, std::string varname);
    NpyArray npy_load(std::string fname);

    template<typename T> std::vector<char>& operator+=(std::vector<char>& lhs, const T rhs) {
        //write in little endian
        for(char byte = 0; (size_t)byte < sizeof(T); byte++) {
            char val = *((char*)&rhs+byte);
            lhs.push_back(val);
        }
        return lhs;
    }

    template<> std::vector<char>& operator+=(std::vector<char>& lhs, const std::string rhs);
    template<> std::vector<char>& operator+=(std::vector<char>& lhs, const char* rhs);


    template<typename T> std::string tostring(T i, int = 0, char = ' ') {
        std::stringstream s;
        s << i;
        return s.str();
    }

    template<typename T> void npy_save(std::string fname, const T* data, const unsigned int* shape, const unsigned int ndims, std::string mode = "w") {
        FILE* fp = NULL;

        if(mode == "a") fp = fopen(fname.c_str(),"r+b");

        if(fp) {
            //file exists. we need to append to it. read the header, modify the array size
            unsigned int word_size, tmp_dims;
            unsigned int* tmp_shape = 0;
            bool fortran_order;
            parse_npy_header(fp,word_size,tmp_shape,tmp_dims,fortran_order);
            cnpy_assert(!fortran_order);

            if(word_size != sizeof(T)) {
                std::cout<<"libnpy error: "<<fname<<" has word size "<<word_size<<" but npy_save appending data sized "<<sizeof(T)<<"\n";
                cnpy_assert( word_size == sizeof(T) );
            }
            if(tmp_dims != ndims) {
                std::cout<<"libnpy error: npy_save attempting to append misdimensioned data to "<<fname<<"\n";
                cnpy_assert(tmp_dims == ndims);
            }

            for(unsigned i = 1; i < ndims; i++) {
                if(shape[i] != tmp_shape[i]) {
                    std::cout<<"libnpy error: npy_save attempting to append misshaped data to "<<fname<<"\n";
                    cnpy_assert(shape[i] == tmp_shape[i]);
                }
            }
            tmp_shape[0] += shape[0];

            fseek(fp,0,SEEK_SET);
            std::vector<char> header = create_npy_header(data,tmp_shape,ndims);
            fwrite(&header[0],sizeof(char),header.size(),fp);
            fseek(fp,0,SEEK_END);

            delete[] tmp_shape;
        }
        else {
            fp = fopen(fname.c_str(),"wb");
            std::vector<char> header = create_npy_header(data,shape,ndims);
            fwrite(&header[0],sizeof(char),header.size(),fp);
        }

        unsigned int nels = 1;
        for(unsigned i = 0;i < ndims;i++) nels *= shape[i];

        fwrite(data,sizeof(T),nels,fp);
        fclose(fp);
    }

    template<typename T> void npz_save(std::string zipname, std::string fname, const T* data, const unsigned int* shape, const unsigned int ndims, std::string mode = "w")
    {
        //first, append a .npy to the fname
        fname += ".npy";

        //now, on with the show
        FILE* fp = NULL;
        unsigned short nrecs = 0;
        unsigned int global_header_offset = 0;
        std::vector<char> global_header;

        if(mode == "a") fp = fopen(zipname.c_str(),"r+b");

        if(fp) {
            //zip file exists. we need to add a new npy file to it.
            //first read the footer. this gives us the offset and size of the global header
            //then read and store the global header.
            //below, we will write the the new data at the start of the global header then append the global header and footer below it
            unsigned int global_header_size;
            parse_zip_footer(fp,nrecs,global_header_size,global_header_offset);
            fseek(fp,global_header_offset,SEEK_SET);
            global_header.resize(global_header_size);
            size_t res = fread(&global_header[0],sizeof(char),global_header_size,fp);
            if(res != global_header_size){
                throw std::runtime_error("npz_save: header read error while adding to existing zip");
            }
            fseek(fp,global_header_offset,SEEK_SET);
        }
        else {
            fp = fopen(zipname.c_str(),"wb");
        }

        std::vector<char> npy_header = create_npy_header(data,shape,ndims);

        unsigned long nels = 1;
        for (unsigned m=0; m<ndims; m++ ) nels *= shape[m];
        int nbytes = nels*sizeof(T) + npy_header.size();

        //get the CRC of the data to be added
        #if defined(HAVE_ZLIB) && HAVE_ZLIB
        unsigned int crc = crc32(0L,(unsigned char*)&npy_header[0],npy_header.size());
        crc = crc32(crc,(unsigned char*)data,nels*sizeof(T));
        #else
        unsigned int crc = 0;
        #endif

        //build the local header
        std::vector<char> local_header;
        local_header += "PK"; //first part of sig
        local_header += (unsigned short) 0x0403; //second part of sig
        local_header += (unsigned short) 20; //min version to extract
        local_header += (unsigned short) 0; //general purpose bit flag
        local_header += (unsigned short) 0; //compression method
        local_header += (unsigned short) 0; //file last mod time
        local_header += (unsigned short) 0;     //file last mod date
        local_header += (unsigned int) crc; //crc
        local_header += (unsigned int) nbytes; //compressed size
        local_header += (unsigned int) nbytes; //uncompressed size
        local_header += (unsigned short) fname.size(); //fname length
        local_header += (unsigned short) 0; //extra field length
        local_header += fname;

        //build global header
        global_header += "PK"; //first part of sig
        global_header += (unsigned short) 0x0201; //second part of sig
        global_header += (unsigned short) 20; //version made by
        global_header.insert(global_header.end(),local_header.begin()+4,local_header.begin()+30);
        global_header += (unsigned short) 0; //file comment length
        global_header += (unsigned short) 0; //disk number where file starts
        global_header += (unsigned short) 0; //internal file attributes
        global_header += (unsigned int) 0; //external file attributes
        global_header += (unsigned int) global_header_offset; //relative offset of local file header, since it begins where the global header used to begin
        global_header += fname;

        //build footer
        std::vector<char> footer;
        footer += "PK"; //first part of sig
        footer += (unsigned short) 0x0605; //second part of sig
        footer += (unsigned short) 0; //number of this disk
        footer += (unsigned short) 0; //disk where footer starts
        footer += (unsigned short) (nrecs+1); //number of records on this disk
        footer += (unsigned short) (nrecs+1); //total number of records
        footer += (unsigned int) global_header.size(); //nbytes of global headers
        footer += (unsigned int) (global_header_offset + nbytes + local_header.size()); //offset of start of global headers, since global header now starts after newly written array
        footer += (unsigned short) 0; //zip file comment length

        //write everything
        fwrite(&local_header[0],sizeof(char),local_header.size(),fp);
        fwrite(&npy_header[0],sizeof(char),npy_header.size(),fp);
        fwrite(data,sizeof(T),nels,fp);
        fwrite(&global_header[0],sizeof(char),global_header.size(),fp);
        fwrite(&footer[0],sizeof(char),footer.size(),fp);
        fclose(fp);
    }

    template<typename T> std::vector<char> create_npy_header(const T*, const unsigned int* shape, const unsigned int ndims) {

        std::vector<char> dict;
        dict += "{'descr': '";
        dict += BigEndianTest();
        dict += map_type(typeid(T));
        dict += tostring(sizeof(T));
        dict += "', 'fortran_order': False, 'shape': (";
        dict += tostring(shape[0]);
        for(unsigned i = 1;i < ndims;i++) {
            dict += ", ";
            dict += tostring(shape[i]);
        }
        if(ndims == 1) dict += ",";
        dict += "), }";
        //pad with spaces so that preamble+dict is modulo 16 bytes. preamble is 10 bytes. dict needs to end with \n
        int remainder = 16 - (10 + dict.size()) % 16;
        dict.insert(dict.end(),remainder,' ');
        dict.back() = '\n';

        std::vector<char> header;
        header += (unsigned char) 0x93;
        header += "NUMPY";
        header += (char) 0x01; //major version of numpy format
        header += (char) 0x00; //minor version of numpy format
        header += (unsigned short) dict.size();
        header.insert(header.end(),dict.begin(),dict.end());

        return header;
    }


}

#endif
