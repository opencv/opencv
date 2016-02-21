// Copyright (c) 2016 Andrés Solís Montero <http://www.solism.ca>, All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
// OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
// OF THE POSSIBILITY OF SUCH DAMAGE.
//
#ifndef __OPENCV_CORE_BPERSISTENCE_HPP__
#define __OPENCV_CORE_BPERSISTENCE_HPP__

/**

 Binary file storage.
 Writing and Reading OpenCV data structures to and from binary files.
 It's also possible to include your own data/classes. It uses a similar approach to FileStorage(xml/yml)
 for easy integration with OpenCV.
 The reason behind of a binary format to save OpenCV data is given when file size is more important than file
 readability.

 Use the following procedure to write something to a binary file:
 -# Create new BFileStorage and open it for writing.
 -# Write all the data you want using the streaming operator `<<`, just like in the case of STL
 streams.
 -# Close the file using BFileStorage::release. BFileStorage destructor also closes the file.

 Use the following procedure to read something from a binary file:
 -#  Open the file storage using BFileStorage constructor.
 -#  Read the data you are interested in sequentially in the same order it was written.
 Use the operator `>>`, just like in the case of STL streams
 -#  Close the storage using BFileStorage::release. BFileStorage destructor also closes the file.

 To write/read your own data structures you need to implement two functions:

 static void writeB(std::ostream &f, const YOURCLASS &example)
 {
     ...
 }
 static void readB(std::istream &f, YOURCLASS &example)
 {
     ...
 }


 An example for reading and writing data structures to binary files.
*/

/**
#include <vector>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

class B
{
private:
    vector<float> _vec;
    float _float;
    int _int;
    Mat _mat;

public:
    void write(ostream &f) const
    {
        cv::writeB(f, _vec);
        cv::writeB(f, _float);
        cv::writeB(f, _int);
        cv::writeB(f, _mat);
    }
    void read(istream &f)
    {
        cv::readB(f, _vec);
        cv::readB(f, _float);
        cv::readB(f, _int);
        cv::readB(f, _mat);
    }
};

static void writeB(std::ostream &f, const B &example)
{
    example.write(f);
}
static void readB(std::istream &f, B &example)
{
    example.read(f);
}


int main(int , char**)
{
    string filename = "file.bin";
    B writeExample;

    BFileStorage bfs(filename, BFileStorage::WRITE);
    bfs << writeExample;
    bfs.release();

    B readExample;
    BFileStorage bfs2(filename, BFileStorage::READ);
    bfs2 >> readExample;
    bfs2.release();
}
**/


#ifndef __cplusplus
#  error bpersistence.hpp header must be compiled as C++
#endif

#include <map>
#include <fstream>
#include "opencv2/core/types.hpp"
#include "opencv2/core/mat.hpp"


namespace cv
{
    class CV_EXPORTS_W BFileStorage
    {

    public:
        enum Mode { READ, WRITE };
        enum { UNDEFINED = 0, OPENED = 1};

        BFileStorage(BFileStorage&)   = delete;
        void operator=(BFileStorage&) = delete;
        virtual ~BFileStorage();
        BFileStorage(const std::string &filename, int flags);
        bool isOpened();
        void release();

        template<typename _Tp>
        friend BFileStorage& operator << (BFileStorage& bfs, const _Tp& value);
        template<typename _Tp>
        friend BFileStorage& operator >> (BFileStorage& bfs, _Tp& value);

    private:
        std::ifstream _fin;
        std::ofstream _fout;
        int _mode;
        int _status;

    };

    template<typename T>
    static inline void writeScalar(std::ostream& out, const T& data)
    {
        out.write(reinterpret_cast<const char*>(&data), sizeof(T));
    }

    template<typename T>
    static inline void readScalar(std::istream& in, T& data)
    {
        in.read(reinterpret_cast<char*>(&data), sizeof(T));
    }

    static inline void writeB(std::ostream& out, const size_t &_data)
    {
        writeScalar(out, _data);
    }
    static inline void writeB(std::ostream& out, const int &_data)
    {
        writeScalar(out, _data);
    }
    static inline void writeB(std::ostream& out, const float &_data)
    {
        writeScalar(out, _data);
    }
    static inline void writeB(std::ostream& out, const double &_data)
    {
        writeScalar(out, _data);
    }
    static inline void writeB(std::ostream& out, const std::string &_data)
    {
        writeScalar(out, _data);
    }

    static inline void readB(std::istream& in, size_t &_data)
    {
        readScalar(in, _data);
    }
    static inline void readB(std::istream& in, int &_data)
    {
        readScalar(in, _data);
    }
    static inline void readB(std::istream& in, float &_data)
    {
        readScalar(in, _data);
    }
    static inline void readB(std::istream& in, double &_data)
    {
        readScalar(in, _data);
    }
    static inline void readB(std::istream& in, std::string &_data)
    {
        readScalar(in, _data);
    }

    static inline void writeB(std::ostream& out, const Mat &_data)
    {
        writeScalar(out, _data.rows);
        writeScalar(out, _data.cols);
        writeScalar(out, _data.type());
        const size_t bytes = _data.cols * _data.elemSize();
        for (int i = 0; i < _data.rows; i++)
            out.write(reinterpret_cast<const char*>(_data.ptr(i, 0)), bytes);
    }

    static inline void readB(std::istream& in, Mat &_data)
    {
        int rows, cols, type;
        readScalar(in, rows);
        readScalar(in, cols);
        readScalar(in, type);
        _data.create(rows, cols, type);
        const size_t bytes = _data.cols * _data.elemSize();
        for (int i = 0; i < _data.rows; i++)
            in.read(reinterpret_cast<char*>(_data.ptr(i, 0)), bytes);
    }
    template<typename _Tp>
    static inline void writeB(std::ostream& out, const Point_<_Tp>& _data)
    {
        writeScalar(out, _data.x);
        writeScalar(out, _data.y);
    }

    template<typename _Tp>
    static inline void readB(std::istream& in, Point_<_Tp>& _data)
    {
        readScalar(in, _data.x);
        readScalar(in, _data.y);
    }

    template<typename _Tp>
    static inline void writeB(std::ostream& out, const Point3_<_Tp>& _data)
    {
        writeScalar(out, _data.x);
        writeScalar(out, _data.y);
        writeScalar(out, _data.z);
    }

    template<typename _Tp>
    static inline void readB(std::istream& in, Point3_<_Tp>& _data)
    {
        readScalar(in, _data.x);
        readScalar(in, _data.y);
        readScalar(in, _data.z);
    }

    template<typename _Tp>
    static inline void writeB(std::ostream& out, const Size_<_Tp>& _data)
    {
        writeScalar(out, _data.width);
        writeScalar(out, _data.height);
    }

    template<typename _Tp>
    static inline void readB(std::istream& in, Size_<_Tp>& _data)
    {
        readScalar(in, _data.width);
        readScalar(in, _data.height);
    }

    template<typename _Tp>
    static inline void writeB(std::ostream& out, const Complex<_Tp>& _data)
    {
        writeScalar(out, _data.r);
        writeScalar(out, _data.i);
    }
    template<typename _Tp>
    static inline void readB(std::istream& in, Complex<_Tp>& _data)
    {
        readScalar(in, _data.r);
        readScalar(in, _data.i);
    }

    template<typename _Tp>
    static inline void writeB(std::ostream& out, const Rect_<_Tp>& _data)
    {
        writeScalar(out, _data.x);
        writeScalar(out, _data.y);
        writeScalar(out, _data.width);
        writeScalar(out, _data.height);
    }


    template<typename _Tp>
    static inline void readB(std::istream& in, Rect_<_Tp>& _data)
    {
        readScalar(in, _data.x);
        readScalar(in, _data.y);
        readScalar(in, _data.width);
        readScalar(in, _data.height);
    }

    template<typename _Tp>
    static inline void writeB(std::ostream& out,const Scalar_<_Tp>& _data)
    {

        writeScalar(out, _data.val[0]);
        writeScalar(out, _data.val[1]);
        writeScalar(out, _data.val[2]);
        writeScalar(out, _data.val[3]);

    }

    template<typename _Tp>
    static inline void readB(std::istream& in, Scalar_<_Tp>& _data)
    {
        _Tp v0,v1,v2,v3;
        readScalar(in, v0);
        readScalar(in, v1);
        readScalar(in, v2);
        readScalar(in, v3);
        _data = Scalar_<_Tp>(saturate_cast<_Tp>(v0),
                             saturate_cast<_Tp>(v1),
                             saturate_cast<_Tp>(v2),
                             saturate_cast<_Tp>(v3));
    }

    template<typename _Tp>
    static inline void writeB(std::ostream& out, const Range& _data)
    {
        writeScalar(out, _data.start);
        writeScalar(out, _data.end);
    }

    template<typename _Tp>
    static inline void readB(std::istream& in, Range& _data)
    {
        readScalar(in, _data.start);
        readScalar(in, _data.end);
    }

    template<typename T>
    static inline void writeB(std::ostream& out, const std::vector<T>& vec) {
        writeScalar(out, vec.size());
        for (typename  std::vector<T>::const_iterator it = vec.begin();
                                                 it != vec.end(); it++)
            writeB(out, *it);
    }

    template<typename K, typename V>
    static inline void writeB(std::ostream& out, const std::map<K,V> &dict)
    {
        writeScalar(out, dict.size());
        for (typename std::map<K,V>::const_iterator it = dict.begin();
                                               it != dict.end(); ++it)
        {
            writeB(out, it->first);
            writeB(out, it->second);
        }
    }
    template<typename T>
    static inline void writeB(std::ostream& out, const std::vector<std::vector<T> > &vec)
    {
        writeScalar(out, vec.size());
        for (typename std::vector<std::vector<T> >::const_iterator it = vec.begin();
                                                        it!= vec.end(); it++)
        {
            writeScalar(out, it->size());
        }
        for (typename std::vector<std::vector<T> >::const_iterator it = vec.begin();
                                                        it != vec.end(); it++)
        {
            for (typename std::vector<T>::const_iterator it2 = it->begin();
                                                    it2 != it->end(); it2++)
            {
                writeB(out, *it2);
            }
        }
    }

    template<typename T>
    static inline void readB(std::istream& in, std::vector<T>& vec) {
        size_t size;
        readScalar(in, size);
        vec.resize(size);
        for (size_t i = 0; i < vec.size(); i++)
            readB(in, vec[i]);
    }

    template<typename K, typename V>
    static inline void readB(std::istream& in, std::map<K,V> &dict)
    {
        size_t size;
        readScalar(in, size);
        for (size_t i = 0; i < size; i++)
        {
            K key;
            V value;
            readB(in, key);
            readB(in, value);
            dict.insert(std::pair<K,V>(key,value));
        }
    }

    template<typename T>
    static inline void readB(std::istream& in, std::vector<std::vector<T> > &vec)
    {
        std::vector<size_t> _sizes;
        readB(in, _sizes);
        vec.clear();
        for (typename std::vector<size_t>::iterator it = _sizes.begin();
                                            it != _sizes.end(); it++)
        {
            std::vector<T> _tmp;
            _tmp.resize(*it);
            for (size_t i = 0; i < *it; i++)
                readB(in, _tmp[i]);

            vec.push_back(_tmp);
        }
    }

    static inline void writeB(std::ostream& out, const KeyPoint &_kp)
    {
        writeB(out, _kp.pt);
        writeScalar(out, _kp.size);
        writeScalar(out, _kp.angle);
        writeScalar(out, _kp.response);
        writeScalar(out, _kp.octave);
        writeScalar(out, _kp.class_id);
    }
    static inline void readB(std::istream &in, KeyPoint &_kp)
    {
        readB(in, _kp.pt);
        readScalar(in, _kp.size);
        readScalar(in, _kp.angle);
        readScalar(in, _kp.response);
        readScalar(in, _kp.octave);
        readScalar(in, _kp.class_id);
    }

    static inline void writeB(std::ostream &out, const DMatch &_kp)
    {
        writeScalar(out, _kp.queryIdx);
        writeScalar(out, _kp.trainIdx);
        writeScalar(out, _kp.imgIdx);
        writeScalar(out, _kp.distance);
    }
    static inline void readB(std::istream &in, DMatch &_kp)
    {
        readScalar(in, _kp.queryIdx);
        readScalar(in, _kp.trainIdx);
        readScalar(in, _kp.imgIdx);
        readScalar(in, _kp.distance);
    }

    template<typename _Tp>
    BFileStorage& operator << (BFileStorage& fs, const _Tp& value)
    {
        if (fs.isOpened() && fs._mode == cv::BFileStorage::WRITE)
            writeB(fs._fout, value);
        return fs;
    }

    template<typename _Tp>
    BFileStorage& operator >> (BFileStorage& fs, _Tp& value)
    {
        if (fs.isOpened() && fs._mode == cv::BFileStorage::READ)
            readB(fs._fin, value);
        return fs;
    }
}
#endif
