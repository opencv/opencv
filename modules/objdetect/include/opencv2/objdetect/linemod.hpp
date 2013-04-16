/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#ifndef __OPENCV_OBJDETECT_LINEMOD_HPP__
#define __OPENCV_OBJDETECT_LINEMOD_HPP__

#include "opencv2/core.hpp"
#include <map>

/****************************************************************************************\
*                                 LINE-MOD                                               *
\****************************************************************************************/

namespace cv {
namespace linemod {

/// @todo Convert doxy comments to rst

/**
 * \brief Discriminant feature described by its location and label.
 */
struct CV_EXPORTS Feature
{
  int x; ///< x offset
  int y; ///< y offset
  int label; ///< Quantization

  Feature() : x(0), y(0), label(0) {}
  Feature(int x, int y, int label);

  void read(const FileNode& fn);
  void write(FileStorage& fs) const;
};

inline Feature::Feature(int _x, int _y, int _label) : x(_x), y(_y), label(_label) {}

struct CV_EXPORTS Template
{
  int width;
  int height;
  int pyramid_level;
  std::vector<Feature> features;

  void read(const FileNode& fn);
  void write(FileStorage& fs) const;
};

/**
 * \brief Represents a modality operating over an image pyramid.
 */
class QuantizedPyramid
{
public:
  // Virtual destructor
  virtual ~QuantizedPyramid() {}

  /**
   * \brief Compute quantized image at current pyramid level for online detection.
   *
   * \param[out] dst The destination 8-bit image. For each pixel at most one bit is set,
   *                 representing its classification.
   */
  virtual void quantize(Mat& dst) const =0;

  /**
   * \brief Extract most discriminant features at current pyramid level to form a new template.
   *
   * \param[out] templ The new template.
   */
  virtual bool extractTemplate(Template& templ) const =0;

  /**
   * \brief Go to the next pyramid level.
   *
   * \todo Allow pyramid scale factor other than 2
   */
  virtual void pyrDown() =0;

protected:
  /// Candidate feature with a score
  struct Candidate
  {
    Candidate(int x, int y, int label, float score);

    /// Sort candidates with high score to the front
    bool operator<(const Candidate& rhs) const
    {
      return score > rhs.score;
    }

    Feature f;
    float score;
  };

  /**
   * \brief Choose candidate features so that they are not bunched together.
   *
   * \param[in]  candidates   Candidate features sorted by score.
   * \param[out] features     Destination vector of selected features.
   * \param[in]  num_features Number of candidates to select.
   * \param[in]  distance     Hint for desired distance between features.
   */
  static void selectScatteredFeatures(const std::vector<Candidate>& candidates,
                                      std::vector<Feature>& features,
                                      size_t num_features, float distance);
};

inline QuantizedPyramid::Candidate::Candidate(int x, int y, int label, float _score) : f(x, y, label), score(_score) {}

/**
 * \brief Interface for modalities that plug into the LINE template matching representation.
 *
 * \todo Max response, to allow optimization of summing (255/MAX) features as uint8
 */
class CV_EXPORTS Modality
{
public:
  // Virtual destructor
  virtual ~Modality() {}

  /**
   * \brief Form a quantized image pyramid from a source image.
   *
   * \param[in] src  The source image. Type depends on the modality.
   * \param[in] mask Optional mask. If not empty, unmasked pixels are set to zero
   *                 in quantized image and cannot be extracted as features.
   */
  Ptr<QuantizedPyramid> process(const Mat& src,
                    const Mat& mask = Mat()) const
  {
    return processImpl(src, mask);
  }

  virtual String name() const =0;

  virtual void read(const FileNode& fn) =0;
  virtual void write(FileStorage& fs) const =0;

  /**
   * \brief Create modality by name.
   *
   * The following modality types are supported:
   * - "ColorGradient"
   * - "DepthNormal"
   */
  static Ptr<Modality> create(const String& modality_type);

  /**
   * \brief Load a modality from file.
   */
  static Ptr<Modality> create(const FileNode& fn);

protected:
  // Indirection is because process() has a default parameter.
  virtual Ptr<QuantizedPyramid> processImpl(const Mat& src,
                        const Mat& mask) const =0;
};

/**
 * \brief Modality that computes quantized gradient orientations from a color image.
 */
class CV_EXPORTS ColorGradient : public Modality
{
public:
  /**
   * \brief Default constructor. Uses reasonable default parameter values.
   */
  ColorGradient();

  /**
   * \brief Constructor.
   *
   * \param weak_threshold   When quantizing, discard gradients with magnitude less than this.
   * \param num_features     How many features a template must contain.
   * \param strong_threshold Consider as candidate features only gradients whose norms are
   *                         larger than this.
   */
  ColorGradient(float weak_threshold, size_t num_features, float strong_threshold);

  virtual String name() const;

  virtual void read(const FileNode& fn);
  virtual void write(FileStorage& fs) const;

  float weak_threshold;
  size_t num_features;
  float strong_threshold;

protected:
  virtual Ptr<QuantizedPyramid> processImpl(const Mat& src,
                        const Mat& mask) const;
};

/**
 * \brief Modality that computes quantized surface normals from a dense depth map.
 */
class CV_EXPORTS DepthNormal : public Modality
{
public:
  /**
   * \brief Default constructor. Uses reasonable default parameter values.
   */
  DepthNormal();

  /**
   * \brief Constructor.
   *
   * \param distance_threshold   Ignore pixels beyond this distance.
   * \param difference_threshold When computing normals, ignore contributions of pixels whose
   *                             depth difference with the central pixel is above this threshold.
   * \param num_features         How many features a template must contain.
   * \param extract_threshold    Consider as candidate feature only if there are no differing
   *                             orientations within a distance of extract_threshold.
   */
  DepthNormal(int distance_threshold, int difference_threshold, size_t num_features,
              int extract_threshold);

  virtual String name() const;

  virtual void read(const FileNode& fn);
  virtual void write(FileStorage& fs) const;

  int distance_threshold;
  int difference_threshold;
  size_t num_features;
  int extract_threshold;

protected:
  virtual Ptr<QuantizedPyramid> processImpl(const Mat& src,
                        const Mat& mask) const;
};

/**
 * \brief Debug function to colormap a quantized image for viewing.
 */
void colormap(const Mat& quantized, Mat& dst);

/**
 * \brief Represents a successful template match.
 */
struct CV_EXPORTS Match
{
  Match()
  {
  }

  Match(int x, int y, float similarity, const String& class_id, int template_id);

  /// Sort matches with high similarity to the front
  bool operator<(const Match& rhs) const
  {
    // Secondarily sort on template_id for the sake of duplicate removal
    if (similarity != rhs.similarity)
      return similarity > rhs.similarity;
    else
      return template_id < rhs.template_id;
  }

  bool operator==(const Match& rhs) const
  {
    return x == rhs.x && y == rhs.y && similarity == rhs.similarity && class_id == rhs.class_id;
  }

  int x;
  int y;
  float similarity;
  String class_id;
  int template_id;
};

inline
Match::Match(int _x, int _y, float _similarity, const String& _class_id, int _template_id)
    : x(_x), y(_y), similarity(_similarity), class_id(_class_id), template_id(_template_id)
{}

/**
 * \brief Object detector using the LINE template matching algorithm with any set of
 * modalities.
 */
class CV_EXPORTS Detector
{
public:
  /**
   * \brief Empty constructor, initialize with read().
   */
  Detector();

  /**
   * \brief Constructor.
   *
   * \param modalities       Modalities to use (color gradients, depth normals, ...).
   * \param T_pyramid        Value of the sampling step T at each pyramid level. The
   *                         number of pyramid levels is T_pyramid.size().
   */
  Detector(const std::vector< Ptr<Modality> >& modalities, const std::vector<int>& T_pyramid);

  /**
   * \brief Detect objects by template matching.
   *
   * Matches globally at the lowest pyramid level, then refines locally stepping up the pyramid.
   *
   * \param      sources   Source images, one for each modality.
   * \param      threshold Similarity threshold, a percentage between 0 and 100.
   * \param[out] matches   Template matches, sorted by similarity score.
   * \param      class_ids If non-empty, only search for the desired object classes.
   * \param[out] quantized_images Optionally return vector<Mat> of quantized images.
   * \param      masks     The masks for consideration during matching. The masks should be CV_8UC1
   *                       where 255 represents a valid pixel.  If non-empty, the vector must be
   *                       the same size as sources.  Each element must be
   *                       empty or the same size as its corresponding source.
   */
  void match(const std::vector<Mat>& sources, float threshold, std::vector<Match>& matches,
             const std::vector<String>& class_ids = std::vector<String>(),
             OutputArrayOfArrays quantized_images = noArray(),
             const std::vector<Mat>& masks = std::vector<Mat>()) const;

  /**
   * \brief Add new object template.
   *
   * \param      sources      Source images, one for each modality.
   * \param      class_id     Object class ID.
   * \param      object_mask  Mask separating object from background.
   * \param[out] bounding_box Optionally return bounding box of the extracted features.
   *
   * \return Template ID, or -1 if failed to extract a valid template.
   */
  int addTemplate(const std::vector<Mat>& sources, const String& class_id,
          const Mat& object_mask, Rect* bounding_box = NULL);

  /**
   * \brief Add a new object template computed by external means.
   */
  int addSyntheticTemplate(const std::vector<Template>& templates, const String& class_id);

  /**
   * \brief Get the modalities used by this detector.
   *
   * You are not permitted to add/remove modalities, but you may dynamic_cast them to
   * tweak parameters.
   */
  const std::vector< Ptr<Modality> >& getModalities() const { return modalities; }

  /**
   * \brief Get sampling step T at pyramid_level.
   */
  int getT(int pyramid_level) const { return T_at_level[pyramid_level]; }

  /**
   * \brief Get number of pyramid levels used by this detector.
   */
  int pyramidLevels() const { return pyramid_levels; }

  /**
   * \brief Get the template pyramid identified by template_id.
   *
   * For example, with 2 modalities (Gradient, Normal) and two pyramid levels
   * (L0, L1), the order is (GradientL0, NormalL0, GradientL1, NormalL1).
   */
  const std::vector<Template>& getTemplates(const String& class_id, int template_id) const;

  int numTemplates() const;
  int numTemplates(const String& class_id) const;
  int numClasses() const { return static_cast<int>(class_templates.size()); }

  std::vector<String> classIds() const;

  void read(const FileNode& fn);
  void write(FileStorage& fs) const;

  String readClass(const FileNode& fn, const String &class_id_override = "");
  void writeClass(const String& class_id, FileStorage& fs) const;

  void readClasses(const std::vector<String>& class_ids,
                   const String& format = "templates_%s.yml.gz");
  void writeClasses(const String& format = "templates_%s.yml.gz") const;

protected:
  std::vector< Ptr<Modality> > modalities;
  int pyramid_levels;
  std::vector<int> T_at_level;

  typedef std::vector<Template> TemplatePyramid;
  typedef std::map<String, std::vector<TemplatePyramid> > TemplatesMap;
  TemplatesMap class_templates;

  typedef std::vector<Mat> LinearMemories;
  // Indexed as [pyramid level][modality][quantized label]
  typedef std::vector< std::vector<LinearMemories> > LinearMemoryPyramid;

  void matchClass(const LinearMemoryPyramid& lm_pyramid,
                  const std::vector<Size>& sizes,
                  float threshold, std::vector<Match>& matches,
                  const String& class_id,
                  const std::vector<TemplatePyramid>& template_pyramids) const;
};

/**
 * \brief Factory function for detector using LINE algorithm with color gradients.
 *
 * Default parameter settings suitable for VGA images.
 */
CV_EXPORTS Ptr<Detector> getDefaultLINE();

/**
 * \brief Factory function for detector using LINE-MOD algorithm with color gradients
 * and depth normals.
 *
 * Default parameter settings suitable for VGA images.
 */
CV_EXPORTS Ptr<Detector> getDefaultLINEMOD();

} // namespace linemod
} // namespace cv

#endif // __OPENCV_OBJDETECT_LINEMOD_HPP__
