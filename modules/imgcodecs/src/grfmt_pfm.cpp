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

#if (defined(_M_IX86) || defined(__i386__) || defined(__i386) || defined(i386)) \
    || (defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
# define OPENCV_PLATFORM_LITTLE_ENDIAN
#elif (defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
    && defined(__GNUC__) && (__GNUC__>4 || (__GNUC__==4 && __GNUC_MINOR__>=3))
# define OPENCV_PLATFORM_BIG_ENDIAN
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
  #else
    #error Cannot determine endianess of platform.
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
    m_channels = 1;
    break;
  case 'F':
    m_channels = 3;
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

  if (m_channels == 3) {
    mat = Mat3f(m_height, m_width);
  } else if (m_channels == 1) {
    mat = Mat1f(m_height, m_width);
  } else {
    CV_Error(Error::StsError, "Expected 1 or 3 channels.");
    return false;
  }

  CV_Assert(abs(m_scale_factor) == 1.0);

  for (int y = m_height - 1; y >= 0; --y) {
    m_strm.getBytes(mat.ptr(y), m_width * mat.elemSize());
    if (is_byte_order_swapped(m_scale_factor)) {
      for (size_t i = 0; i < m_width * m_channels; ++i) {
        swap_endianess(mat.ptr<float>(y)[i]);
      }
    }
  }

  if (mat.channels() == 3) {
    cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);
  }

  mat /= abs(m_scale_factor);


  return true;
}

size_t PFMDecoder::signatureLength() const
{
    return 2;
}

bool PFMDecoder::checkSignature( const String& signature ) const
{
    return signature.size() >= 2 && signature[0] == 'P'
                                 && ( signature[1] == 'f' || signature[1] == 'F' );
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
    return depth == CV_32FC1 || depth == CV_32FC3;
}

bool PFMEncoder::write(const Mat& img, const std::vector<int>& params)
{
  (void) params;

  WLByteStream strm;
  if (m_buf) {
    if (!strm.open(*m_buf)) {
      return false;
    } else {
      m_buf->reserve(alignSize(256 + img.elemSize() * img.total(), 256));
    }
  } else if (!strm.open(m_filename)) {
    return false;
  }

  strm.putByte('P');
  switch (img.channels()) {
  case 1:
    strm.putByte('f');
    break;
  case 3:
    strm.putByte('F');
    break;
  default:
    CV_Error(Error::StsBadArg, "Expected 1 or 3 channel image.");
  }
  strm.putByte('\n');


  write_anything(strm, img.cols);
  strm.putByte(' ');
  write_anything(strm, img.rows);
  strm.putByte('\n');

#ifdef OPENCV_PLATFORM_LITTLE_ENDIAN
  write_anything(strm, -1.0);
#elif OPENCV_PLATFORM_BIG_ENDIAN
  write_anything(strm, 1.0);
#else
  #error Cannot determine endianess of platform.
#endif

  strm.putByte('\n');

  // Comments are not officially supported in this file format.
  // write_anything(strm, "# Generated by OpenCV " CV_VERSION "\n");


  for (int y = img.rows - 1; y >= 0; --y)
  {
    if (img.channels() == 3) {
      const float* bgr_row = img.ptr<float>(y);
      size_t row_size = img.cols * img.channels();
      std::vector<float> rgb_row(row_size);
      for (int x = 0; x < img.cols; ++x) {
        rgb_row[x*3+0] = bgr_row[x*3+2];
        rgb_row[x*3+1] = bgr_row[x*3+1];
        rgb_row[x*3+2] = bgr_row[x*3+0];
      }
      strm.putBytes(reinterpret_cast<const uchar*>(rgb_row.data()), sizeof(float) * row_size);
    } else if (img.channels() == 1) {
      strm.putBytes(img.ptr(y), sizeof(float) * img.cols);
    }
  }
  return true;
}


}


#endif // HAVE_IMGCODEC_PFM
