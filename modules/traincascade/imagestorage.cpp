#include "cv.h"
#include "imagestorage.h"
#include <stdio.h>
#include <iostream>
#include <fstream>

bool CvCascadeImageReader::create( const String _posFilename, const String _negFilename, Size _winSize )
{
    return posReader.create(_posFilename) && negReader.create(_negFilename, _winSize);
}

CvCascadeImageReader::NegReader::NegReader()
{
    src.create( 0, 0 , CV_8UC1 );
    img.create( 0, 0, CV_8UC1 );
    point = offset = Point( 0, 0 );
    scale       = 1.0F;
    scaleFactor = 1.4142135623730950488016887242097F;
    stepFactor  = 0.5F;
}

bool CvCascadeImageReader::NegReader::create( const String _filename, Size _winSize )
{
    String dirname, str;
    std::ifstream file(_filename.c_str());
    if ( !file.is_open() )
        return false;

    size_t pos = _filename.rfind('\\');
    char dlmrt = '\\';
    if (pos == String::npos)
    {
        pos = _filename.rfind('/');
        dlmrt = '/';
    }
    dirname = pos == String::npos ? "" : _filename.substr(0, pos) + dlmrt;
    while( !file.eof() )
    {
        std::getline(file, str);
        if (str.empty()) break;
        if (str.at(0) == '#' ) continue; /* comment */
        imgFilenames.push_back(dirname + str);
    }
    file.close();

    winSize = _winSize;
    last = round = 0;
    return true;
}

bool CvCascadeImageReader::NegReader::nextImg()
{
    Point _offset = Point(0,0);
    size_t count = imgFilenames.size();
    for( size_t i = 0; i < count; i++ )
    {
        src = imread( imgFilenames[last++], 0 );
        if( src.empty() ) 
            continue;
        round += last / count;
        round = round % (winSize.width * winSize.height);
        last %= count;

        _offset.x = min( (int)round % winSize.width, src.cols - winSize.width );
        _offset.y = min( (int)round / winSize.width, src.rows - winSize.height );
        if( !src.empty() && src.type() == CV_8UC1 
                && offset.x >= 0 && offset.y >= 0 )
            break;
    }

    if( src.empty() )
        return false; // no appropriate image
    point = offset = _offset;
    scale = max( ((float)winSize.width + point.x) / ((float)src.cols),
                 ((float)winSize.height + point.y) / ((float)src.rows) );
    
    Size sz( (int)(scale*src.cols + 0.5F), (int)(scale*src.rows + 0.5F) );
    resize( src, img, sz );
    return true;
}

bool CvCascadeImageReader::NegReader::get( Mat& _img )
{
    CV_Assert( !_img.empty() );
    CV_Assert( _img.type() == CV_8UC1 );
    CV_Assert( _img.cols == winSize.width );
    CV_Assert( _img.rows == winSize.height );

    if( img.empty() )
        if ( !nextImg() ) 
            return false;

    Mat mat( winSize.height, winSize.width, CV_8UC1,
        (void*)(img.data + point.y * img.step + point.x * img.elemSize()), img.step );
    mat.copyTo(_img);

    if( (int)( point.x + (1.0F + stepFactor ) * winSize.width ) < img.cols )
        point.x += (int)(stepFactor * winSize.width);
    else
    {
        point.x = offset.x;
        if( (int)( point.y + (1.0F + stepFactor ) * winSize.height ) < img.rows )
            point.y += (int)(stepFactor * winSize.height);
        else
        {
            point.y = offset.y;
            scale *= scaleFactor;
            if( scale <= 1.0F )
                resize( src, img, Size( (int)(scale*src.cols), (int)(scale*src.rows) ) );
            else
            {
                if ( !nextImg() ) 
                    return false;
            }
        }
    }
    return true;
}

CvCascadeImageReader::PosReader::PosReader()
{
    file = 0;
    vec = 0;
}

bool CvCascadeImageReader::PosReader::create( const String _filename )
{
    if ( file )
        fclose( file );
    file = fopen( _filename.c_str(), "rb" );

    if( !file )
        return false;
    short tmp = 0;  
    if( fread( &count, sizeof( count ), 1, file ) != 1 ||
        fread( &vecSize, sizeof( vecSize ), 1, file ) != 1 ||
        fread( &tmp, sizeof( tmp ), 1, file ) != 1 ||
        fread( &tmp, sizeof( tmp ), 1, file ) != 1 )
        CV_Error_( CV_StsParseError, ("wrong file format for %s\n", _filename.c_str()) );
    base = sizeof( count ) + sizeof( vecSize ) + 2*sizeof( tmp );
    if( feof( file ) )
        return false;
    last = 0;
    vec = (short*) cvAlloc( sizeof( *vec ) * vecSize );
    CV_Assert( vec );
    return true;
}

bool CvCascadeImageReader::PosReader::get( Mat &_img )
{
    CV_Assert( _img.rows * _img.cols == vecSize );
    uchar tmp = 0;
    fread( &tmp, sizeof( tmp ), 1, file );
    fread( vec, sizeof( vec[0] ), vecSize, file );

    if( feof( file ) || last++ >= count )
        return false;

    for( int r = 0; r < _img.rows; r++ )
    {
        for( int c = 0; c < _img.cols; c++ )
            _img.ptr(r)[c] = (uchar)vec[r * _img.cols + c];
    }
    return true;
}

void CvCascadeImageReader::PosReader::restart()
{
    CV_Assert( file );
    last = 0;
    fseek( file, base, SEEK_SET );
}

CvCascadeImageReader::PosReader::~PosReader()
{
    if (file)
        fclose( file );
    cvFree( &vec );
}
