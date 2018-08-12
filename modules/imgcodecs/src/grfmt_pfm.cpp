// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "utils.hpp"
#include "grfmt_pfm.hpp"
#include <iostream>

#ifdef HAVE_IMGCODEC_PFM

namespace {

static_assert(sizeof(float) == 4, "float must be 32 bit.");

#if    ( defined(_M_IX86) || defined(__i386__) || defined(__i386) || defined(i386) ) \
    || ( defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__ ) \
    || ( defined(__ANDROID_API__) && defined(__LITTLE_ENDIAN_BITFIELD)) \
    || ( defined (_WIN32) )
# define OPENCV_PLATFORM_LITTLE_ENDIAN
#elif  ( ( defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__ ) && defined(__GNUC__) && ( __GNUC__>4  || (__GNUC__==4 && __GNUC_MINOR__>=3) ) ) \
    || ( defined(__ANDROID_API__) && defined(__BIG_ENDIAN_BITFIELS)) 
#  ifdef OPENCV_PLATFORM_LITTLE_ENDIAN
#    error Cannot determine endianess of platform.
#  else
#    define OPENCV_PLATFORM_BIG_ENDIAN
#  endif
#else
#  error Cannot determine endianess of platform.
#endif

bool is_byte_order_swapped(float scale)
{
  // ".pfm" format file specifies that:
  // positive scale means big endianess;
  // negative scale means little endianess.

  #ifdef OPENCV_PLATFORM_LITTLE_ENDIAN
    return scale >= 0.0;
  #elif OPENCV_PLATFORM_BIG_ENDIAN
    return scale < 0.0;
  #endif
}

void swap_endianess(uint32_t& ui)
{
  static const uint32_t A(0x000000ffU);
  static const uint32_t B(0x0000ff00U);
  static const uint32_t C(0x00ff0000U);
  static const uint32_t D(0xff000000U);

  ui = ( (ui & A) << 24 )
     | ( (ui & B) <<  8 )
     | ( (ui & C) >>  8 )
     | ( (ui & D) >> 24 );
}

template<typename T> T atoT(const std::string& s);
template<> int atoT<int>(const std::string& s) { return std::atoi(s.c_str()); }
template<> float atoT<float>(const std::string& s) { return std::atof(s.c_str()); }

template<typename T>
T read_number(cv::RLByteStream& strm)
{
  // should be enogh to take string representation of any number
  const size_t buffer_size = 2048;

  std::vector<char> buffer(buffer_size, 0);
  for (size_t i = 0; i < buffer_size; ++i) {
    char c = strm.getByte();
    if (std::isspace(c)) {
      break;
    }
    buffer[i] = c;
  }
  const std::string str(buffer.begin(), buffer.end());
  return atoT<T>(str);
}

template<typename T> void write_anything(cv::WLByteStream& strm, const T& t)
{
  const std::string str = std::to_string(t);
  strm.putBytes(str.c_str(), str.size());
}

}

namespace cv {

PFMDecoder::~PFMDecoder()
{
}

PFMDecoder::PFMDecoder()
{
  m_strm.close();
}

bool PFMDecoder::readHeader()
{
  if (m_buf.empty()) {
    if (!m_strm.open(m_filename)) {
      return false;
    }
  } else {
    if (!m_strm.open(m_buf)) {
      return false;
    }
  }

  if (m_strm.getByte() != 'P') {
    CV_Error(Error::StsError, "Unexpected file type (expected P)");
    return false;
  }

  switch (m_strm.getByte()) {
  case 'f':
    m_type = CV_32FC1;
    break;
  case 'F':
    m_type = CV_32FC3;
    break;
  default:
    CV_Error(Error::StsError, "Unexpected file type (expected `f` or `F`)");
    return false;
  }

  if ('\n' != m_strm.getByte()) {
    CV_Error(Error::StsError, "Unexpected header format (expected line break)");
    return false;
  }


  m_width = read_number<int>(m_strm);
  m_height = read_number<int>(m_strm);
  m_scale_factor = read_number<float>(m_strm);
  m_swap_byte_order = is_byte_order_swapped(m_scale_factor);

  return true;
}

bool PFMDecoder::readData(Mat& mat)
{
  if (!m_strm.isOpened()) {
    CV_Error(Error::StsError, "Unexpected status in data stream");
    return false;
  }

  Mat buffer(mat.size(), m_type);
  for (int y = m_height - 1; y >= 0; --y) {
    m_strm.getBytes(buffer.ptr(y), m_width * buffer.elemSize());
    if (is_byte_order_swapped(m_scale_factor)) {
      for (int i = 0; i < m_width * buffer.channels(); ++i) {
        static_assert( sizeof(uint32_t) == sizeof(float),
                       "uint32_t and float must have same size." );
        swap_endianess(buffer.ptr<uint32_t>(y)[i]);
      }
    }
  }

  if (buffer.channels() == 3) {
    cv::cvtColor(buffer, buffer, cv::COLOR_BGR2RGB);
  }

  CV_Assert(fabs(m_scale_factor) > 0.0f);
  buffer *= 1.f / fabs(m_scale_factor);

  buffer.convertTo(mat, mat.type());

  return true;
}

size_t PFMDecoder::signatureLength() const
{
    return 3;
}

bool PFMDecoder::checkSignature( const String& signature ) const
{
    return signature.size() >= 3
        && signature[0] == 'P'
        && ( signature[1] == 'f' || signature[1] == 'F' )
        && isspace(signature[2]);
}

void PFMDecoder::close()
{
  // noop
}

//////////////////////////////////////////////////////////////////////////////////////////

PFMEncoder::PFMEncoder()
{
  m_description = "Portable image format - float (*.pfm)";
}

PFMEncoder::~PFMEncoder()
{
}

bool PFMEncoder::isFormatSupported(int depth) const
{
  return CV_MAT_DEPTH(depth) == CV_32F || CV_MAT_DEPTH(depth) == CV_8U;
}

bool PFMEncoder::write(const Mat& img, const std::vector<int>& params)
{
  (void) params;

  WLByteStream strm;
  if (m_buf) {
    if (!strm.open(*m_buf)) {
      return false;
    } else {
      m_buf->reserve(alignSize(256 + sizeof(float) * img.channels() * img.total(), 256));
    }
  } else if (!strm.open(m_filename)) {
    return false;
  }

  Mat float_img;
  strm.putByte('P');
  switch (img.channels()) {
  case 1:
    strm.putByte('f');
    img.convertTo(float_img, CV_32FC1);
    break;
  case 3:
    strm.putByte('F');
    img.convertTo(float_img, CV_32FC3);
    break;
  default:
    CV_Error(Error::StsBadArg, "Expected 1 or 3 channel image.");
  }
  strm.putByte('\n');


  write_anything(strm, float_img.cols);
  strm.putByte(' ');
  write_anything(strm, float_img.rows);
  strm.putByte('\n');
#ifdef OPENCV_PLATFORM_LITTLE_ENDIAN
  write_anything(strm, -1.0);
#elif OPENCV_PLATFORM_BIG_ENDIAN
  write_anything(strm, 1.0);
#endif

  strm.putByte('\n');

  // Comments are not officially supported in this file format.
  // write_anything(strm, "# Generated by OpenCV " CV_VERSION "\n");

  for (int y = float_img.rows - 1; y >= 0; --y)
  {
    if (float_img.channels() == 3) {
      const float* bgr_row = float_img.ptr<float>(y);
      size_t row_size = float_img.cols * float_img.channels();
      std::vector<float> rgb_row(row_size);
      for (int x = 0; x < float_img.cols; ++x) {
        rgb_row[x*3+0] = bgr_row[x*3+2];
        rgb_row[x*3+1] = bgr_row[x*3+1];
        rgb_row[x*3+2] = bgr_row[x*3+0];
      }
      strm.putBytes(reinterpret_cast<const uchar*>(rgb_row.data()), sizeof(float) * row_size);
    } else if (float_img.channels() == 1) {
      strm.putBytes(float_img.ptr(y), sizeof(float) * float_img.cols);
    }
  }
  return true;
}


}


#endif // HAVE_IMGCODEC_PFM
