// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html

#include <string>
#include <vector>

#include "test_precomp.hpp"

namespace opencv_test { namespace {

/**
 * Test to check whether the EXIF orientation tag was processed successfully or not.
 * The test uses a set of 8 images named testExifOrientation_{1 to 8}.(extension).
 * Each test image is a 10x10 square, divided into four smaller sub-squares:
 * (R corresponds to Red, G to Green, B to Blue, W to White)
 * ---------             ---------
 * | R | G |             | G | R |
 * |-------| - (tag 1)   |-------| - (tag 2)
 * | B | W |             | W | B |
 * ---------             ---------
 *
 * ---------             ---------
 * | W | B |             | B | W |
 * |-------| - (tag 3)   |-------| - (tag 4)
 * | G | R |             | R | G |
 * ---------             ---------
 *
 * ---------             ---------
 * | R | B |             | G | W |
 * |-------| - (tag 5)   |-------| - (tag 6)
 * | G | W |             | R | B |
 * ---------             ---------
 *
 * ---------             ---------
 * | W | G |             | B | R |
 * |-------| - (tag 7)   |-------| - (tag 8)
 * | B | R |             | W | G |
 * ---------             ---------
 *
 *
 * Each image contains an EXIF field with an orientation tag (0x112).
 * After reading each image and applying the orientation tag,
 * the resulting image should be:
 * ---------
 * | R | G |
 * |-------|
 * | B | W |
 * ---------
 *
 * Note:
 * The flags parameter of the imread function is set as IMREAD_COLOR | IMREAD_ANYCOLOR | IMREAD_ANYDEPTH.
 * Using this combination is an undocumented trick to load images similarly to the IMREAD_UNCHANGED flag,
 * preserving the alpha channel (if present) while also applying the orientation.
 */

typedef testing::TestWithParam<string> Exif;

TEST_P(Exif, exif_orientation)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    const string filename = root + GetParam();
    const int colorThresholdHigh = 250;
    const int colorThresholdLow = 5;

    // Refer to the note in the explanation above.
    Mat m_img = imread(filename, IMREAD_COLOR | IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
    ASSERT_FALSE(m_img.empty());

    if (m_img.channels() == 3)
    {
        Vec3b vec;

        //Checking the first quadrant (with supposed red)
        vec = m_img.at<Vec3b>(2, 2); //some point inside the square
        EXPECT_LE(vec.val[0], colorThresholdLow);
        EXPECT_LE(vec.val[1], colorThresholdLow);
        EXPECT_GE(vec.val[2], colorThresholdHigh);

        //Checking the second quadrant (with supposed green)
        vec = m_img.at<Vec3b>(2, 7);  //some point inside the square
        EXPECT_LE(vec.val[0], colorThresholdLow);
        EXPECT_GE(vec.val[1], colorThresholdHigh);
        EXPECT_LE(vec.val[2], colorThresholdLow);

        //Checking the third quadrant (with supposed blue)
        vec = m_img.at<Vec3b>(7, 2);  //some point inside the square
        EXPECT_GE(vec.val[0], colorThresholdHigh);
        EXPECT_LE(vec.val[1], colorThresholdLow);
        EXPECT_LE(vec.val[2], colorThresholdLow);
    }
    else
    {
        Vec4b vec;

        //Checking the first quadrant (with supposed red)
        vec = m_img.at<Vec4b>(2, 2); //some point inside the square
        EXPECT_LE(vec.val[0], colorThresholdLow);
        EXPECT_LE(vec.val[1], colorThresholdLow);
        EXPECT_GE(vec.val[2], colorThresholdHigh);

        //Checking the second quadrant (with supposed green)
        vec = m_img.at<Vec4b>(2, 7);  //some point inside the square
        EXPECT_LE(vec.val[0], colorThresholdLow);
        EXPECT_GE(vec.val[1], colorThresholdHigh);
        EXPECT_LE(vec.val[2], colorThresholdLow);

        //Checking the third quadrant (with supposed blue)
        vec = m_img.at<Vec4b>(7, 2);  //some point inside the square
        EXPECT_GE(vec.val[0], colorThresholdHigh);
        EXPECT_LE(vec.val[1], colorThresholdLow);
        EXPECT_LE(vec.val[2], colorThresholdLow);
    }
}

const std::vector<std::string> exif_files
{
#ifdef HAVE_JPEG
    "readwrite/testExifOrientation_1.jpg",
    "readwrite/testExifOrientation_2.jpg",
    "readwrite/testExifOrientation_3.jpg",
    "readwrite/testExifOrientation_4.jpg",
    "readwrite/testExifOrientation_5.jpg",
    "readwrite/testExifOrientation_6.jpg",
    "readwrite/testExifOrientation_7.jpg",
    "readwrite/testExifOrientation_8.jpg",
#endif
#ifdef OPENCV_IMGCODECS_PNG_WITH_EXIF
    "readwrite/testExifOrientation_1.png",
    "readwrite/testExifOrientation_2.png",
    "readwrite/testExifOrientation_3.png",
    "readwrite/testExifOrientation_4.png",
    "readwrite/testExifOrientation_5.png",
    "readwrite/testExifOrientation_6.png",
    "readwrite/testExifOrientation_7.png",
    "readwrite/testExifOrientation_8.png",
#endif
#ifdef HAVE_AVIF
    "readwrite/testExifOrientation_1.avif",
    "readwrite/testExifOrientation_2.avif",
    "readwrite/testExifOrientation_3.avif",
    "readwrite/testExifOrientation_4.avif",
    "readwrite/testExifOrientation_5.avif",
    "readwrite/testExifOrientation_6.avif",
    "readwrite/testExifOrientation_7.avif",
    "readwrite/testExifOrientation_8.avif",
#endif
};

INSTANTIATE_TEST_CASE_P(Imgcodecs, Exif,
    testing::ValuesIn(exif_files));

enum ExifTagId
{
    TAG_EMPTY = 0,
    TAG_SUB_FILETYPE = 254,
    TAG_IMAGE_WIDTH = 256,
    TAG_IMAGE_LENGTH = 257,
    TAG_BITS_PER_SAMPLE = 258,
    TAG_COMPRESSION = 259,
    TAG_PHOTOMETRIC = 262,
    TAG_IMAGE_DESCRIPTION = 270,
    TAG_MAKE = 271,
    TAG_MODEL = 272,
    TAG_STRIP_OFFSET = 273,
    TAG_SAMPLES_PER_PIXEL = 277,
    TAG_ROWS_PER_STRIP = 278,
    TAG_STRIP_BYTE_COUNTS = 279,
    TAG_PLANAR_CONFIG = 284,
    TAG_ORIENTATION = 274,

    TAG_XRESOLUTION = 282,
    TAG_YRESOLUTION = 283,
    TAG_RESOLUTION_UNIT = 296,

    TAG_SOFTWARE = 305,
    TAG_MODIFY_DATE = 306,

    TAG_SAMPLE_FORMAT = 339,

    // DNG extension
    TAG_CFA_REPEAT_PATTERN_DIM = 33421,
    TAG_CFA_PATTERN = 33422,

    TAG_COPYRIGHT = 33432,
    TAG_EXPOSURE_TIME = 33434,
    TAG_FNUMBER = 33437,

    TAG_EXIF_TAGS = 34665,
    TAG_ISOSPEED = 34855,

    TAG_EXIF_VERSION = 36864,
    TAG_DATETIME_ORIGINAL = 36867,
    TAG_DATETIME_CREATE = 36868,

    TAG_SHUTTER_SPEED = 37377,
    TAG_APERTURE_VALUE = 37378,
    TAG_FLASH = 37385,
    TAG_FOCALLENGTH = 37386,
    TAG_EP_STANDARD_ID = 37398,

    TAG_SUBSECTIME = 37520,
    TAG_SUBSECTIME_ORIGINAL = 37521,
    TAG_SUBSECTIME_DIGITIZED = 37522,

    TAG_EXIF_IMAGE_WIDTH = 40962,
    TAG_EXIF_IMAGE_HEIGHT = 40963,
    TAG_WHITE_BALANCE = 41987,
};

enum ExifTagType
{
    TAG_TYPE_NOTYPE = 0,
    TAG_TYPE_BYTE = 1,
    TAG_TYPE_ASCII = 2,  // null-terminated string
    TAG_TYPE_SHORT = 3,
    TAG_TYPE_LONG = 4,
    TAG_TYPE_RATIONAL = 5,  // 64-bit unsigned fraction
    TAG_TYPE_SBYTE = 6,
    TAG_TYPE_UNDEFINED = 7,  // 8-bit untyped data */
    TAG_TYPE_SSHORT = 8,
    TAG_TYPE_SLONG = 9,
    TAG_TYPE_SRATIONAL = 10,  // 64-bit signed fraction
    TAG_TYPE_FLOAT = 11,
    TAG_TYPE_DOUBLE = 12,
    TAG_TYPE_IFD = 13,     // 32-bit unsigned integer (offset)
    TAG_TYPE_LONG8 = 16,   // BigTIFF 64-bit unsigned
    TAG_TYPE_SLONG8 = 17,  // BigTIFF 64-bit signed
    TAG_TYPE_IFD8 = 18     // BigTIFF 64-bit unsigned integer (offset)
};

struct rational64_t
{
    int64_t num, denom;
};

struct ExifTag
{
    int id=0;
    ExifTagType type=TAG_TYPE_NOTYPE;
    std::string str;
    rational64_t n={0, 1};
    std::vector<rational64_t> v;
    
    bool empty() const { return id == 0; }
    size_t nvalues() const;
};

constexpr size_t EXIF_HDR_SIZE = 8; // ('II' or 'MM'), (0x2A 0x00), (IFD0 offset: 4 bytes)
constexpr size_t IFD_ENTRY_SIZE = 12;
constexpr size_t IFD_MAX_INLINE_SIZE = 4;
constexpr size_t IFD_HDR_SIZE = 6;

size_t tagTypeSize(ExifTagType type)
{
    return
    type == TAG_TYPE_NOTYPE ? 0 :
    type == TAG_TYPE_BYTE ? 1 :
    type == TAG_TYPE_ASCII ? 1 :
    type == TAG_TYPE_SHORT ? 2 :
    type == TAG_TYPE_LONG ? 4 :
    type == TAG_TYPE_RATIONAL ? 8 :
    type == TAG_TYPE_SBYTE ? 1 :
    type == TAG_TYPE_UNDEFINED ? 1 :
    type == TAG_TYPE_SSHORT ? 2 :
    type == TAG_TYPE_SLONG ? 4 :
    type == TAG_TYPE_SRATIONAL ? 8 :
    type == TAG_TYPE_FLOAT ? 4 :
    type == TAG_TYPE_DOUBLE ? 8 :
    type == TAG_TYPE_IFD ? 0 :
    type == TAG_TYPE_LONG8 ? 8 :
    type == TAG_TYPE_SLONG8 ? 8 :
    type == TAG_TYPE_IFD8 ? 0 : 0;
}

size_t ExifTag::nvalues() const
{
    return empty() ? 0u :
        type == TAG_TYPE_ASCII || type == TAG_TYPE_UNDEFINED ? str.size() + (type == TAG_TYPE_ASCII) :
        !v.empty() ? v.size() : 1u;
}

size_t tagValueSize(ExifTagType type, size_t nvalues)
{
    size_t size = tagTypeSize(type)*nvalues;
    return (size + 1u) & ~1u;
}

static void pack1(std::vector<uchar>& data, size_t& offset, uint8_t value)
{
    data.resize(std::max(data.size(), offset+1));
    data[offset++] = (char)value;
}

static void pack2(std::vector<uchar>& data, size_t& offset,
                  uint16_t value, bool bigendian_)
{
    size_t ofs = offset, bigendian = (size_t)bigendian_;
    data.resize(std::max(data.size(), ofs+sizeof(uint16_t)));
    uchar* ptr = data.data();
    ptr[ofs + bigendian] = (uchar)value;
    ptr[ofs + 1 - bigendian] = (uchar)(value >> 8);
    offset = ofs + sizeof(uint16_t);
}

static void pack4(std::vector<uchar>& data, size_t& offset,
                  uint32_t value, bool bigendian_)
{
    size_t ofs = offset, bigendian = (size_t)bigendian_;
    data.resize(std::max(data.size(), ofs+sizeof(uint32_t)));
    uchar* ptr = data.data();
    ptr[ofs+bigendian*3] = (uchar)value;
    ptr[ofs+1+bigendian] = (uchar)(value >> 8);
    ptr[ofs+2-bigendian] = (uchar)(value >> 16);
    ptr[ofs+3-bigendian*3] = (uchar)(value >> 24);
    offset = ofs + sizeof(uint32_t);
}

static size_t computeIFDSize(const std::vector<ExifTag>* ifds,
                             size_t nifds, size_t idx, size_t& values_size)
{
    CV_Assert(idx < nifds);
    const std::vector<ExifTag>& ifd = ifds[idx];
    size_t i, ntags = ifd.size(), size = IFD_HDR_SIZE + IFD_ENTRY_SIZE*ntags;
    for (i = 0; i < ntags; i++) {
        const ExifTag& tag = ifd[i];
        if (tag.type == TAG_TYPE_IFD) {
            int64_t subifd_idx = tag.n.num;
            CV_Assert_N(0 <= subifd_idx, (size_t)subifd_idx < nifds);
            size += computeIFDSize(ifds, nifds, (size_t)subifd_idx, values_size);
        } else {
            size_t tag_values_size = tagValueSize(tag.type, tag.nvalues());
            if (tag_values_size > IFD_MAX_INLINE_SIZE)
                values_size += tag_values_size;
        }
    }
    return size;
}

static void packIFD(const std::vector<ExifTag>* ifds, size_t nifds, size_t idx,
                    std::vector<uchar>& data, size_t& offset,
                    size_t& values_offset, bool bigendian)
{
    CV_Assert(idx < nifds);
    const std::vector<ExifTag>& ifd = ifds[idx];
    std::vector<std::pair<size_t, size_t> > subifds;
    size_t ntags = ifd.size();
    
    size_t subifd_offset0 = offset + IFD_HDR_SIZE + ntags*IFD_ENTRY_SIZE;
    size_t subifd_offset = subifd_offset0;
    pack2(data, offset, (uint16_t)ntags, bigendian);
    
    // first, pack the specified (by idx) IFD without subdirectories
    for (const ExifTag& tag: ifd) {
        pack2(data, offset, (uint16_t)tag.id, bigendian);
        
        ExifTagType type = tag.type == TAG_TYPE_IFD ? TAG_TYPE_LONG : tag.type;
        pack2(data, offset, (uint16_t)type, bigendian);
        size_t nvalues = tag.nvalues();
        
        pack4(data, offset, (uint32_t)nvalues, bigendian);
        if (tag.type == TAG_TYPE_IFD) {
            int64_t sub_idx = tag.n.num;
            CV_Assert_N(sub_idx >= 0, (size_t)sub_idx < nifds);
            subifds.push_back({(size_t)sub_idx, subifd_offset});
            pack4(data, offset, (uint32_t)subifd_offset, bigendian);
            const std::vector<ExifTag>& subifd = ifds[sub_idx];
            size_t subifd_ntags = subifd.size();
            subifd_offset += IFD_HDR_SIZE + subifd_ntags*IFD_ENTRY_SIZE;
            continue;
        }
        size_t tag_values_size = tagValueSize(type, nvalues);
        int inline_values = tag_values_size <= 4u;
        size_t tag_values_offset = inline_values ? offset : values_offset;
        if (!inline_values) {
            pack4(data, offset, (uint32_t)values_offset, bigendian);
            data.resize(std::max(data.size(), tag_values_offset + tag_values_size));
        } else {
            pack4(data, offset, 0u, bigendian);
        }
        
        if (type == TAG_TYPE_ASCII || type == TAG_TYPE_UNDEFINED) {
            size_t v_size = tag.str.size();
            memcpy(&data[tag_values_offset], tag.str.c_str(), v_size);
            if (type == TAG_TYPE_ASCII) {
                data[tag_values_offset + v_size] = '\0';
                v_size++;
            }
            if ((v_size & 1u) != 0) {
                data[tag_values_offset + v_size] = '\0';
                v_size++;
            }
            tag_values_offset += v_size;
        } else if (type == TAG_TYPE_RATIONAL || type == TAG_TYPE_SRATIONAL ||
                   type == TAG_TYPE_BYTE || type == TAG_TYPE_SBYTE ||
                   type == TAG_TYPE_SHORT || type == TAG_TYPE_SSHORT ||
                   type == TAG_TYPE_LONG || type == TAG_TYPE_SLONG) {
            const rational64_t* nptr = tag.v.empty() ? &tag.n : tag.v.data();
            int64_t minval =
                type == TAG_TYPE_SBYTE ? INT8_MIN :
                type == TAG_TYPE_SSHORT ? INT16_MIN :
                type == TAG_TYPE_SLONG || type == TAG_TYPE_SRATIONAL ? INT32_MIN : 0;
            int64_t maxval =
                type == TAG_TYPE_BYTE || type == TAG_TYPE_UNDEFINED ? UINT8_MAX :
                type == TAG_TYPE_SBYTE ? INT8_MAX :
                type == TAG_TYPE_SHORT ? UINT16_MAX :
                type == TAG_TYPE_SSHORT ? INT16_MAX :
                type == TAG_TYPE_LONG ? UINT32_MAX :
                type == TAG_TYPE_SLONG || type == TAG_TYPE_SRATIONAL ? INT32_MAX : INT64_MAX;
            for (size_t i = 0; i < nvalues; i++) {
                int64_t n = std::min(std::max(nptr[i].num, minval), maxval);
                if (type == TAG_TYPE_RATIONAL || type == TAG_TYPE_SRATIONAL) {
                    int64_t d = std::min(std::max(nptr[i].denom, minval), maxval);
                    pack4(data, tag_values_offset, (uint32_t)n, bigendian);
                    pack4(data, tag_values_offset, (uint32_t)d, bigendian);
                }
                else if (type == TAG_TYPE_LONG || type == TAG_TYPE_SLONG)
                    pack4(data, tag_values_offset, (uint32_t)n, bigendian);
                else if (type == TAG_TYPE_SHORT || type == TAG_TYPE_SSHORT)
                    pack2(data, tag_values_offset, (uint16_t)n, bigendian);
                else
                    pack1(data, tag_values_offset, (uint8_t)n);
            }
            if ((type == TAG_TYPE_BYTE || type == TAG_TYPE_SBYTE) && (nvalues & 1) != 0)
                pack1(data, tag_values_offset, (uint8_t)0);
        } else {
            CV_Error_(Error::StsBadArg, ("unsupported tag type %d", tag.type));
        }
        
        if (!inline_values)
            values_offset = tag_values_offset;
    }
    
    pack4(data, offset, 0u, bigendian);
    
    // now pack all sub-IFDs and the next one, if any
    for (auto sub: subifds) {
        size_t subofs = sub.second;
        packIFD(ifds, nifds, sub.first, data, subofs, values_offset, bigendian);
    }
}

static bool packExif(const std::vector<std::vector<ExifTag> >& exif,
                     std::vector<uchar>& data, bool bigendian)
{
    data.clear();
    size_t values_size = 0;
    size_t ifd_size = computeIFDSize(exif.data(), exif.size(), 0u, values_size) + EXIF_HDR_SIZE;
    data.resize(ifd_size + values_size);
    
    char signature = bigendian ? 'M' : 'I';
    size_t offset = 0;
    pack1(data, offset, (uint8_t)signature);
    pack1(data, offset, (uint8_t)signature);
    pack2(data, offset, 42u, bigendian);
    pack4(data, offset, 8u, bigendian);
    
    packIFD(exif.data(), exif.size(), 0u, data, offset, ifd_size, bigendian);
    return true;
}

static ExifTag exifInt(int id, ExifTagType type, int64_t v)
{
    ExifTag tag;
    tag.id = id;
    tag.type = type;
    CV_Assert(type == TAG_TYPE_LONG || type == TAG_TYPE_SLONG ||
              type == TAG_TYPE_SHORT || type == TAG_TYPE_SSHORT ||
              type == TAG_TYPE_BYTE || type == TAG_TYPE_SBYTE);
    tag.n.num = v;
    tag.n.denom = 1;
    return tag;
}

static ExifTag exifStr(int id, ExifTagType type, const std::string& str)
{
    ExifTag tag;
    tag.id = id;
    CV_Assert(type == TAG_TYPE_ASCII || type == TAG_TYPE_UNDEFINED);
    tag.type = type;
    tag.str = str;
    return tag;
}

static rational64_t doubleToRational(double v, int maxbits)
{
    rational64_t r = {1, 0};
    if (std::isfinite(v)) {
        int e = 0;
        frexp(v, &e);
        if (e >= maxbits)
            return r;

        double iv = round(v);
        if (iv == v) {
            r.denom = 1;
            r.num = (int64_t)iv;
        } else {
            r.denom = (int64_t)1 << (maxbits - std::max(e, 0));
            r.num = (int64_t)round(v*r.denom);
            while ((r.denom & 1) == 0 && (r.num & 1) == 0) {
                r.num >>= 1;
                r.denom >>= 1;
            }
        }
    }
    return r;
}

static ExifTag exifRatio(int id, ExifTagType type, double v)
{
    ExifTag tag;
    tag.id = id;
    CV_Assert(type == TAG_TYPE_RATIONAL || type == TAG_TYPE_SRATIONAL);
    tag.type = type;
    tag.n = doubleToRational(v, (type == TAG_TYPE_RATIONAL ? 31 : 30));
    return tag;
}

static ExifTag exifIDF(int id, int idx)
{
    ExifTag tag;
    tag.id = id;
    tag.type = TAG_TYPE_IFD;
    tag.n.num = idx;
    tag.n.denom = 1;
    return tag;
}

static Mat makeCirclesImage(Size size, int type, int nbits)
{
    Mat img(size, type);
    img.setTo(Scalar::all(0));
    RNG& rng = theRNG();
    int maxval = (int)(1 << nbits);
    for (int i = 0; i < 100; i++) {
        int x = rng.uniform(0, img.cols);
        int y = rng.uniform(0, img.rows);
        int radius = rng.uniform(5, std::min(img.cols, img.rows)/5);
        int b = rng.uniform(0, maxval);
        int g = rng.uniform(0, maxval);
        int r = rng.uniform(0, maxval);
        circle(img, Point(x, y), radius, Scalar(b, g, r), -1, LINE_AA);
    }
    return img;
}

static std::vector<std::vector<ExifTag> > makeTestExif(Size imgsize, int nbits, int orientation=1)
{
    std::vector<std::vector<ExifTag> > exif =
    {
        {
            exifInt(TAG_IMAGE_WIDTH, TAG_TYPE_LONG, imgsize.width),
            exifInt(TAG_IMAGE_LENGTH, TAG_TYPE_LONG, imgsize.height),
            exifInt(TAG_BITS_PER_SAMPLE, TAG_TYPE_SHORT, nbits),
            exifInt(TAG_ORIENTATION, TAG_TYPE_SHORT, orientation),
            exifStr(TAG_IMAGE_DESCRIPTION, TAG_TYPE_ASCII, format("Sample %d-bit image with metadata", nbits)),
            exifStr(TAG_SOFTWARE, TAG_TYPE_ASCII, "OpenCV"),
            exifRatio(TAG_XRESOLUTION, TAG_TYPE_RATIONAL, 72.),
            exifRatio(TAG_YRESOLUTION, TAG_TYPE_RATIONAL, 72.),
            exifInt(TAG_RESOLUTION_UNIT, TAG_TYPE_SHORT, 2),
            exifIDF(TAG_EXIF_TAGS, 1)
        },
        {
            exifStr(TAG_EXIF_VERSION, TAG_TYPE_UNDEFINED, "0221"),
            exifInt(TAG_EXIF_IMAGE_WIDTH, TAG_TYPE_LONG, imgsize.width),
            exifInt(TAG_EXIF_IMAGE_HEIGHT, TAG_TYPE_LONG, imgsize.height)
        }
    };
    return exif;
}

TEST(Imgcodecs_Avif, ReadWriteWithExif)
{
    int avif_nbits = 10;
    int avif_speed = 10;
    int avif_quality = 85;
    int imgdepth = avif_nbits > 8 ? CV_16U : CV_8U;
    int imgtype = CV_MAKETYPE(imgdepth, 3);
    const string outputname = cv::tempfile(".avif");
    Mat img = makeCirclesImage(Size(1280, 720), imgtype, avif_nbits);
    
    std::vector<int> metadata_types = {IMAGE_METADATA_EXIF};
    std::vector<std::vector<uchar> > metadata(1);
    std::vector<std::vector<ExifTag> > exif = makeTestExif(img.size(), avif_nbits);
    packExif(exif, metadata[0], true);
    
    std::vector<int> write_params = {
        IMWRITE_AVIF_DEPTH, avif_nbits,
        IMWRITE_AVIF_SPEED, avif_speed,
        IMWRITE_AVIF_QUALITY, avif_quality
    };
    
    imwriteWithMetadata(outputname, img, metadata_types, metadata, write_params);
    std::vector<uchar> compressed;
    imencodeWithMetadata(outputname, img, metadata_types, metadata, compressed, write_params);
    
    std::vector<int> read_metadata_types, read_metadata_types2;
    std::vector<std::vector<uchar> > read_metadata, read_metadata2;
    Mat img2 = imreadWithMetadata(outputname, read_metadata_types, read_metadata, IMREAD_UNCHANGED);
    Mat img3 = imdecodeWithMetadata(compressed, read_metadata_types2, read_metadata2, IMREAD_UNCHANGED);
    EXPECT_EQ(img2.cols, img.cols);
    EXPECT_EQ(img2.rows, img.rows);
    EXPECT_EQ(img2.type(), imgtype);
    EXPECT_EQ(read_metadata_types, read_metadata_types2);
    EXPECT_GE(read_metadata_types.size(), 1u);
    EXPECT_EQ(read_metadata, read_metadata2);
    EXPECT_EQ(read_metadata_types[0], IMAGE_METADATA_EXIF);
    EXPECT_EQ(read_metadata_types.size(), read_metadata.size());
    EXPECT_EQ(read_metadata[0], metadata[0]);
    EXPECT_EQ(cv::norm(img2, img3, NORM_INF), 0.);
    double mse = cv::norm(img, img2, NORM_L2SQR)/(img.rows*img.cols);
    EXPECT_LT(mse, 1500);
    remove(outputname.c_str());
}

TEST(Imgcodecs_Jpeg, ReadWriteWithExif)
{
    int jpeg_quality = 95;
    int imgtype = CV_MAKETYPE(CV_8U, 3);
    const string outputname = cv::tempfile(".jpeg");
    Mat img = makeCirclesImage(Size(1280, 720), imgtype, 8);
    
    std::vector<int> metadata_types = {IMAGE_METADATA_EXIF};
    std::vector<std::vector<uchar> > metadata(1);
    std::vector<std::vector<ExifTag> > exif = makeTestExif(img.size(), 8);
    packExif(exif, metadata[0], true);
    
    std::vector<int> write_params = {
        IMWRITE_JPEG_QUALITY, jpeg_quality
    };
    
    imwriteWithMetadata(outputname, img, metadata_types, metadata, write_params);
    std::vector<uchar> compressed;
    imencodeWithMetadata(outputname, img, metadata_types, metadata, compressed, write_params);

    std::vector<int> read_metadata_types, read_metadata_types2;
    std::vector<std::vector<uchar> > read_metadata, read_metadata2;
    Mat img2 = imreadWithMetadata(outputname, read_metadata_types, read_metadata, IMREAD_UNCHANGED);
    Mat img3 = imdecodeWithMetadata(compressed, read_metadata_types2, read_metadata2, IMREAD_UNCHANGED);
    EXPECT_EQ(img2.cols, img.cols);
    EXPECT_EQ(img2.rows, img.rows);
    EXPECT_EQ(img2.type(), imgtype);
    EXPECT_EQ(read_metadata_types, read_metadata_types2);
    EXPECT_GE(read_metadata_types.size(), 1u);
    EXPECT_EQ(read_metadata, read_metadata2);
    EXPECT_EQ(read_metadata_types[0], IMAGE_METADATA_EXIF);
    EXPECT_EQ(read_metadata_types.size(), read_metadata.size());
    EXPECT_EQ(read_metadata[0], metadata[0]);
    EXPECT_EQ(cv::norm(img2, img3, NORM_INF), 0.);
    double mse = cv::norm(img, img2, NORM_L2SQR)/(img.rows*img.cols);
    EXPECT_LT(mse, 80);
    remove(outputname.c_str());
}

}
}
