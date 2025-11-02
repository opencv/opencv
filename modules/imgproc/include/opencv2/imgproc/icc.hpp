#ifndef OPENCV_IMGPROC_ICC_HPP
#define OPENCV_IMGPROC_ICC_HPP

#include "opencv2/core.hpp"

namespace cv {

enum IccProfileVersion {
    ICC_VERSION_2 = 2,
    ICC_VERSION_4 = 4, 
    ICC_VERSION_5 = 5,
    ICC_MAX_VERSION = 255
};

// Constants for implementation compatibility  
#define ICC_PROFILE_V2 ICC_VERSION_2
#define ICC_PROFILE_V4 ICC_VERSION_4
#define ICC_PROFILE_V5 ICC_VERSION_5

enum ColorAppearanceModel {
    CAM_NONE = 0,
    CAM_CAM02 = 1,
    CAM_CAM16 = 2,
    CAM_ZCAM = 3
};

enum IccRenderingIntent {
    INTENT_PERCEPTUAL = 0,
    INTENT_RELATIVE_COLORIMETRIC = 1,
    INTENT_SATURATION = 2,
    INTENT_ABSOLUTE_COLORIMETRIC = 3
};

class CV_EXPORTS IccProfile {
public:
    IccProfile();
    explicit IccProfile(const String& filename);
    IccProfile(const void* data, size_t size);
    IccProfile(const std::vector<uchar>& data);
    ~IccProfile();
    
    bool load(const String& filename);
    bool load(const std::vector<uchar>& data);
    bool isValid() const;
    IccProfileVersion getVersion() const;
    String getColorSpace() const;
    String getPCS() const;
    int getInputChannels() const;
    int getOutputChannels() const;
    bool supportsFloat() const;
    bool supportsHDR() const;
    String getDescription() const;
    
private:
    struct Impl;
    std::shared_ptr<Impl> p;
};

struct CV_EXPORTS ViewingConditions {
    Vec3f whitePoint;
    float adaptingLuminance;
    float backgroundLuminance;
    int surround;
    bool discountIlluminant;
    ViewingConditions();
};

// Function declarations moved to main imgproc.hpp for Python bindings
void colorProfileTransform(InputArray src, OutputArray dst,
                          const String& srcProfilePath,
                          const String& dstProfilePath,
                          int renderingIntent);

String createStandardProfilePath(const String& colorSpace);

bool isIccSupported();

// Constants for backward compatibility with tests  
const int ICC_PERCEPTUAL = INTENT_PERCEPTUAL;
const int ICC_RELATIVE_COLORIMETRIC = INTENT_RELATIVE_COLORIMETRIC;
const int ICC_SATURATION = INTENT_SATURATION;
const int ICC_ABSOLUTE_COLORIMETRIC = INTENT_ABSOLUTE_COLORIMETRIC;
const int CAM02 = CAM_CAM02;
const int CAM16 = CAM_CAM16;

// Additional functions for comprehensive ICC API (always declared, implementation conditional)
IccProfile createStandardProfile(const String& colorSpace);
ViewingConditions getStandardViewingConditions(const String& environment);
void colorProfileTransformSingle(InputArray src, OutputArray dst, 
                                const IccProfile& srcProfile, const IccProfile& dstProfile);

// Overloaded versions for different parameter combinations
void colorProfileTransform(InputArray src, OutputArray dst,
                          const IccProfile& srcProfile, const IccProfile& dstProfile);
void colorProfileTransform(InputArray src, OutputArray dst,
                          const IccProfile& srcProfile, const IccProfile& dstProfile,
                          int renderingIntent, int cam = CAM_NONE,
                          const ViewingConditions& vc = ViewingConditions());

} // namespace cv

#endif
