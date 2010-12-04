#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/ml/ml.hpp"

#include <fstream>
#include <iostream>
#include <memory>

#if defined WIN32 || defined _WIN32
#include "sys/types.h"
#endif
#include <sys/stat.h>

#define DEBUG_DESC_PROGRESS

using namespace cv;
using namespace std;

const string paramsFile = "params.xml";
const string vocabularyFile = "vocabulary.xml.gz";
const string bowImageDescriptorsDir = "/bowImageDescriptors";
const string svmsDir = "/svms";
const string plotsDir = "/plots";

void help(char** argv)
{
	cout << "\nThis program shows how to read in, train on and produce test results for the PASCAL VOC (Visual Object Challenge) data. \n"
	 << "It shows how to use detectors, descriptors and recognition methods \n"
		"Using OpenCV version %s\n" << CV_VERSION << "\n"
	 << "Call: \n"
    << "Format:\n ./" << argv[0] << " [VOC path] [result directory]  \n"
    << "       or:  \n"
    << " ./" << argv[0] << " [VOC path] [result directory] [feature detector] [descriptor extractor] [descriptor matcher] \n"
    << "\n"
    << "Input parameters: \n"
    << "[VOC path]             Path to Pascal VOC data (e.g. /home/my/VOCdevkit/VOC2010). Note: VOC2007-VOC2010 are supported. \n"
    << "[result directory]     Path to result diractory. Following folders will be created in [result directory]: \n"
    << "                         bowImageDescriptors - to store image descriptors, \n"
    << "                         svms - to store trained svms, \n"
    << "                         plots - to store files for plots creating. \n"
    << "[feature detector]     Feature detector name (e.g. SURF, FAST...) - see createFeatureDetector() function in detectors.cpp \n"
    << "                         Currently 12/2010, this is FAST, STAR, SIFT, SURF, MSER, GFTT, HARRIS \n"
    << "[descriptor extractor] Descriptor extractor name (e.g. SURF, SIFT) - see createDescriptorExtractor() function in descriptors.cpp \n"
    << "                         Currently 12/2010, this is SURF, OpponentSIFT, SIFT, OpponentSURF, BRIEF \n"
    << "[descriptor matcher]   Descriptor matcher name (e.g. BruteForce) - see createDescriptorMatcher() function in matchers.cpp \n"
    << "                         Currently 12/2010, this is BruteForce, BruteForce-L1, FlannBased, BruteForce-Hamming, BruteForce-HammingLUT \n"
    << "\n";
}





void makeDir( const string& dir )
{
#if defined WIN32 || defined _WIN32
    CreateDirectory( dir.c_str(), 0 );
#else
    mkdir( dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH );
#endif
}

void makeUsedDirs( const string& rootPath )
{
    makeDir(rootPath + bowImageDescriptorsDir);
    makeDir(rootPath + svmsDir);
    makeDir(rootPath + plotsDir);
}

/****************************************************************************************\
*                    Classes to work with PASCAL VOC dataset                             *
\****************************************************************************************/
//
// TODO: refactor this part of the code
//


//used to specify the (sub-)dataset over which operations are performed
enum ObdDatasetType {CV_OBD_TRAIN, CV_OBD_TEST};

class ObdObject
{
public:
    string object_class;
    Rect boundingBox;
};

//extended object data specific to VOC
enum VocPose {CV_VOC_POSE_UNSPECIFIED, CV_VOC_POSE_FRONTAL, CV_VOC_POSE_REAR, CV_VOC_POSE_LEFT, CV_VOC_POSE_RIGHT};
class VocObjectData
{
public:
    bool difficult;
    bool occluded;
    bool truncated;
    VocPose pose;
};
//enum VocDataset {CV_VOC2007, CV_VOC2008, CV_VOC2009, CV_VOC2010};
enum VocPlotType {CV_VOC_PLOT_SCREEN, CV_VOC_PLOT_PNG};
enum VocGT {CV_VOC_GT_NONE, CV_VOC_GT_DIFFICULT, CV_VOC_GT_PRESENT};
enum VocConfCond {CV_VOC_CCOND_RECALL, CV_VOC_CCOND_SCORETHRESH};
enum VocTask {CV_VOC_TASK_CLASSIFICATION, CV_VOC_TASK_DETECTION};

class ObdImage
{
public:
    ObdImage(string p_id, string p_path) : id(p_id), path(p_path) {}
    string id;
    string path;
};

//used by getDetectorGroundTruth to sort a two dimensional list of floats in descending order
class ObdScoreIndexSorter
{
public:
    float score;
    int image_idx;
    int obj_idx;
    bool operator < (const ObdScoreIndexSorter& compare) const {return (score < compare.score);}
};

class VocData
{
public:
    VocData( const string& vocPath, bool useTestDataset )
        { initVoc( vocPath, useTestDataset ); }
    ~VocData(){}
    /* functions for returning classification/object data for multiple images given an object class */
    void getClassImages(const string& obj_class, const ObdDatasetType dataset, vector<ObdImage>& images, vector<char>& object_present);
    void getClassObjects(const string& obj_class, const ObdDatasetType dataset, vector<ObdImage>& images, vector<vector<ObdObject> >& objects);
    void getClassObjects(const string& obj_class, const ObdDatasetType dataset, vector<ObdImage>& images, vector<vector<ObdObject> >& objects, vector<vector<VocObjectData> >& object_data, vector<VocGT>& ground_truth);
    /* functions for returning object data for a single image given an image id */
    ObdImage getObjects(const string& id, vector<ObdObject>& objects);
    ObdImage getObjects(const string& id, vector<ObdObject>& objects, vector<VocObjectData>& object_data);
    ObdImage getObjects(const string& obj_class, const string& id, vector<ObdObject>& objects, vector<VocObjectData>& object_data, VocGT& ground_truth);
    /* functions for returning the ground truth (present/absent) for groups of images */
    void getClassifierGroundTruth(const string& obj_class, const vector<ObdImage>& images, vector<char>& ground_truth);
    void getClassifierGroundTruth(const string& obj_class, const vector<string>& images, vector<char>& ground_truth);
    int getDetectorGroundTruth(const string& obj_class, const ObdDatasetType dataset, const vector<ObdImage>& images, const vector<vector<Rect> >& bounding_boxes, const vector<vector<float> >& scores, vector<vector<char> >& ground_truth, vector<vector<char> >& detection_difficult, bool ignore_difficult = true);
    /* functions for writing VOC-compatible results files */
    void writeClassifierResultsFile(const string& out_dir, const string& obj_class, const ObdDatasetType dataset, const vector<ObdImage>& images, const vector<float>& scores, const int competition = 1, const bool overwrite_ifexists = false);
    /* functions for calculating metrics from a set of classification/detection results */
    string getResultsFilename(const string& obj_class, const VocTask task, const ObdDatasetType dataset, const int competition = -1, const int number = -1);
    void calcClassifierPrecRecall(const string& obj_class, const vector<ObdImage>& images, const vector<float>& scores, vector<float>& precision, vector<float>& recall, float& ap, vector<size_t>& ranking);
    void calcClassifierPrecRecall(const string& obj_class, const vector<ObdImage>& images, const vector<float>& scores, vector<float>& precision, vector<float>& recall, float& ap);
    void calcClassifierPrecRecall(const string& input_file, vector<float>& precision, vector<float>& recall, float& ap, bool outputRankingFile = false);
    /* functions for calculating confusion matrices */
    void calcClassifierConfMatRow(const string& obj_class, const vector<ObdImage>& images, const vector<float>& scores, const VocConfCond cond, const float threshold, vector<string>& output_headers, vector<float>& output_values);
    void calcDetectorConfMatRow(const string& obj_class, const ObdDatasetType dataset, const vector<ObdImage>& images, const vector<vector<float> >& scores, const vector<vector<Rect> >& bounding_boxes, const VocConfCond cond, const float threshold, vector<string>& output_headers, vector<float>& output_values, bool ignore_difficult = true);
    /* functions for outputting gnuplot output files */
    void savePrecRecallToGnuplot(const string& output_file, const vector<float>& precision, const vector<float>& recall, const float ap, const string title = string(), const VocPlotType plot_type = CV_VOC_PLOT_SCREEN);
    /* functions for reading in result/ground truth files */
    void readClassifierGroundTruth(const string& obj_class, const ObdDatasetType dataset, vector<ObdImage>& images, vector<char>& object_present);
    void readClassifierResultsFile(const std:: string& input_file, vector<ObdImage>& images, vector<float>& scores);
    void readDetectorResultsFile(const string& input_file, vector<ObdImage>& images, vector<vector<float> >& scores, vector<vector<Rect> >& bounding_boxes);
    /* functions for getting dataset info */
    const vector<string>& getObjectClasses();
    string getResultsDirectory();
protected:
    void initVoc( const string& vocPath, const bool useTestDataset );
    void initVoc2007to2010( const string& vocPath, const bool useTestDataset);
    void readClassifierGroundTruth(const string& filename, vector<string>& image_codes, vector<char>& object_present);
    void readClassifierResultsFile(const string& input_file, vector<string>& image_codes, vector<float>& scores);
    void readDetectorResultsFile(const string& input_file, vector<string>& image_codes, vector<vector<float> >& scores, vector<vector<Rect> >& bounding_boxes);
    void extractVocObjects(const string filename, vector<ObdObject>& objects, vector<VocObjectData>& object_data);
    string getImagePath(const string& input_str);

    void getClassImages_impl(const string& obj_class, const string& dataset_str, vector<ObdImage>& images, vector<char>& object_present);
    void calcPrecRecall_impl(const vector<char>& ground_truth, const vector<float>& scores, vector<float>& precision, vector<float>& recall, float& ap, vector<size_t>& ranking, int recall_normalization = -1);

    //test two bounding boxes to see if they meet the overlap criteria defined in the VOC documentation
    float testBoundingBoxesForOverlap(const Rect detection, const Rect ground_truth);
    //extract class and dataset name from a VOC-standard classification/detection results filename
    void extractDataFromResultsFilename(const string& input_file, string& class_name, string& dataset_name);
    //get classifier ground truth for a single image
    bool getClassifierGroundTruthImage(const string& obj_class, const string& id);

    //utility functions
    void getSortOrder(const vector<float>& values, vector<size_t>& order, bool descending = true);
    int stringToInteger(const string input_str);
    void readFileToString(const string filename, string& file_contents);
    string integerToString(const int input_int);
    string checkFilenamePathsep(const string filename, bool add_trailing_slash = false);
    void convertImageCodesToObdImages(const vector<string>& image_codes, vector<ObdImage>& images);
    int extractXMLBlock(const string src, const string tag, const int searchpos, string& tag_contents);
    //utility sorter
    struct orderingSorter
    {
        bool operator ()(std::pair<size_t, vector<float>::const_iterator> const& a, std::pair<size_t, vector<float>::const_iterator> const& b)
        {
            return (*a.second) > (*b.second);
        }
    };
    //data members
    string m_vocPath;
    string m_vocName;
    //string m_resPath;

    string m_annotation_path;
    string m_image_path;
    string m_imageset_path;
    string m_class_imageset_path;

    vector<string> m_classifier_gt_all_ids;
    vector<char> m_classifier_gt_all_present;
    string m_classifier_gt_class;

    //data members
    string m_train_set;
    string m_test_set;

    vector<string> m_object_classes;


    float m_min_overlap;
    bool m_sampled_ap;
};


//Return the classification ground truth data for all images of a given VOC object class
//--------------------------------------------------------------------------------------
//INPUTS:
// - obj_class          The VOC object class identifier string
// - dataset            Specifies whether to extract images from the training or test set
//OUTPUTS:
// - images             An array of ObdImage containing info of all images extracted from the ground truth file
// - object_present     An array of bools specifying whether the object defined by 'obj_class' is present in each image or not
//NOTES:
// This function is primarily useful for the classification task, where only
// whether a given object is present or not in an image is required, and not each object instance's
// position etc.
void VocData::getClassImages(const string& obj_class, const ObdDatasetType dataset, vector<ObdImage>& images, vector<char>& object_present)
{
    string dataset_str;
    //generate the filename of the classification ground-truth textfile for the object class
    if (dataset == CV_OBD_TRAIN)
    {
        dataset_str = m_train_set;
    } else {
        dataset_str = m_test_set;
    }

    getClassImages_impl(obj_class, dataset_str, images, object_present);
}

void VocData::getClassImages_impl(const string& obj_class, const string& dataset_str, vector<ObdImage>& images, vector<char>& object_present)
{
    //generate the filename of the classification ground-truth textfile for the object class
    string gtFilename = m_class_imageset_path;
    gtFilename.replace(gtFilename.find("%s"),2,obj_class);
    gtFilename.replace(gtFilename.find("%s"),2,dataset_str);

    //parse the ground truth file, storing in two separate vectors
    //for the image code and the ground truth value
    vector<string> image_codes;
    readClassifierGroundTruth(gtFilename, image_codes, object_present);

    //prepare output arrays
    images.clear();

    convertImageCodesToObdImages(image_codes, images);
}

//Return the object data for all images of a given VOC object class
//-----------------------------------------------------------------
//INPUTS:
// - obj_class          The VOC object class identifier string
// - dataset            Specifies whether to extract images from the training or test set
//OUTPUTS:
// - images             An array of ObdImage containing info of all images in chosen dataset (tag, path etc.)
// - objects            Contains the extended object info (bounding box etc.) for each object instance in each image
// - object_data        Contains VOC-specific extended object info (marked difficult etc.)
// - ground_truth       Specifies whether there are any difficult/non-difficult instances of the current
//                          object class within each image
//NOTES:
// This function returns extended object information in addition to the absent/present
// classification data returned by getClassImages. The objects returned for each image in the 'objects'
// array are of all object classes present in the image, and not just the class defined by 'obj_class'.
// 'ground_truth' can be used to determine quickly whether an object instance of the given class is present
// in an image or not.
void VocData::getClassObjects(const string& obj_class, const ObdDatasetType dataset, vector<ObdImage>& images, vector<vector<ObdObject> >& objects)
{
    vector<vector<VocObjectData> > object_data;
    vector<VocGT> ground_truth;

    getClassObjects(obj_class,dataset,images,objects,object_data,ground_truth);
}

void VocData::getClassObjects(const string& obj_class, const ObdDatasetType dataset, vector<ObdImage>& images, vector<vector<ObdObject> >& objects, vector<vector<VocObjectData> >& object_data, vector<VocGT>& ground_truth)
{
    //generate the filename of the classification ground-truth textfile for the object class
    string gtFilename = m_class_imageset_path;
    gtFilename.replace(gtFilename.find("%s"),2,obj_class);
    if (dataset == CV_OBD_TRAIN)
    {
        gtFilename.replace(gtFilename.find("%s"),2,m_train_set);
    } else {
        gtFilename.replace(gtFilename.find("%s"),2,m_test_set);
    }

    //parse the ground truth file, storing in two separate vectors
    //for the image code and the ground truth value
    vector<string> image_codes;
    vector<char> object_present;
    readClassifierGroundTruth(gtFilename, image_codes, object_present);

    //prepare output arrays
    images.clear();
    objects.clear();
    object_data.clear();
    ground_truth.clear();

    string annotationFilename;
    vector<ObdObject> image_objects;
    vector<VocObjectData> image_object_data;
    VocGT image_gt;

    //transfer to output arrays and read in object data for each image
    for (size_t i = 0; i < image_codes.size(); ++i)
    {
        ObdImage image = getObjects(obj_class, image_codes[i], image_objects, image_object_data, image_gt);

        images.push_back(image);
        objects.push_back(image_objects);
        object_data.push_back(image_object_data);
        ground_truth.push_back(image_gt);
    }
}

//Return ground truth data for the objects present in an image with a given UID
//-----------------------------------------------------------------------------
//INPUTS:
// - id                 VOC Dataset unique identifier (string code in form YYYY_XXXXXX where YYYY is the year)
//OUTPUTS:
// - obj_class (*3)     Specifies the object class to use to resolve 'ground_truth'
// - objects            Contains the extended object info (bounding box etc.) for each object in the image
// - object_data (*2,3) Contains VOC-specific extended object info (marked difficult etc.)
// - ground_truth (*3)  Specifies whether there are any difficult/non-difficult instances of the current
//                          object class within the image
//RETURN VALUE:
// ObdImage containing path and other details of image file with given code
//NOTES:
// There are three versions of this function
//  * One returns a simple array of objects given an id [1]
//  * One returns the same as (1) plus VOC specific object data [2]
//  * One returns the same as (2) plus the ground_truth flag. This also requires an extra input obj_class [3]
ObdImage VocData::getObjects(const string& id, vector<ObdObject>& objects)
{
    vector<VocObjectData> object_data;
    ObdImage image = getObjects(id, objects, object_data);

    return image;
}

ObdImage VocData::getObjects(const string& id, vector<ObdObject>& objects, vector<VocObjectData>& object_data)
{
    //first generate the filename of the annotation file
    string annotationFilename = m_annotation_path;

    annotationFilename.replace(annotationFilename.find("%s"),2,id);

    //extract objects contained in the current image from the xml
    extractVocObjects(annotationFilename,objects,object_data);

    //generate image path from extracted string code
    string path = getImagePath(id);

    ObdImage image(id, path);
    return image;
}

ObdImage VocData::getObjects(const string& obj_class, const string& id, vector<ObdObject>& objects, vector<VocObjectData>& object_data, VocGT& ground_truth)
{

    //extract object data (except for ground truth flag)
    ObdImage image = getObjects(id,objects,object_data);

    //pregenerate a flag to indicate whether the current class is present or not in the image
    ground_truth = CV_VOC_GT_NONE;
    //iterate through all objects in current image
    for (size_t j = 0; j < objects.size(); ++j)
    {
        if (objects[j].object_class == obj_class)
        {
            if (object_data[j].difficult == false)
            {
                //if at least one non-difficult example is present, this flag is always set to CV_VOC_GT_PRESENT
                ground_truth = CV_VOC_GT_PRESENT;
                break;
            } else {
                //set if at least one object instance is present, but it is marked difficult
                ground_truth = CV_VOC_GT_DIFFICULT;
            }
        }
    }

    return image;
}

//Return ground truth data for the presence/absence of a given object class in an arbitrary array of images
//---------------------------------------------------------------------------------------------------------
//INPUTS:
// - obj_class          The VOC object class identifier string
// - images             An array of ObdImage OR strings containing the images for which ground truth
//                          will be computed
//OUTPUTS:
// - ground_truth       An output array indicating the presence/absence of obj_class within each image
void VocData::getClassifierGroundTruth(const string& obj_class, const vector<ObdImage>& images, vector<char>& ground_truth)
{
    vector<char>(images.size()).swap(ground_truth);

    vector<ObdObject> objects;
    vector<VocObjectData> object_data;
    vector<char>::iterator gt_it = ground_truth.begin();
    for (vector<ObdImage>::const_iterator it = images.begin(); it != images.end(); ++it, ++gt_it)
    {
        //getObjects(obj_class, it->id, objects, object_data, voc_ground_truth);
        (*gt_it) = (getClassifierGroundTruthImage(obj_class, it->id));
    }
}

void VocData::getClassifierGroundTruth(const string& obj_class, const vector<string>& images, vector<char>& ground_truth)
{
    vector<char>(images.size()).swap(ground_truth);

    vector<ObdObject> objects;
    vector<VocObjectData> object_data;
    vector<char>::iterator gt_it = ground_truth.begin();
    for (vector<string>::const_iterator it = images.begin(); it != images.end(); ++it, ++gt_it)
    {
        //getObjects(obj_class, (*it), objects, object_data, voc_ground_truth);
        (*gt_it) = (getClassifierGroundTruthImage(obj_class, (*it)));
    }
}

//Return ground truth data for the accuracy of detection results
//--------------------------------------------------------------
//INPUTS:
// - obj_class          The VOC object class identifier string
// - images             An array of ObdImage containing the images for which ground truth
//                          will be computed
// - bounding_boxes     A 2D input array containing the bounding box rects of the objects of
//                          obj_class which were detected in each image
//OUTPUTS:
// - ground_truth       A 2D output array indicating whether each object detection was accurate
//                          or not
// - detection_difficult A 2D output array indicating whether the detection fired on an object
//                          marked as 'difficult'. This allows it to be ignored if necessary
//                          (the voc documentation specifies objects marked as difficult
//                          have no effects on the results and are effectively ignored)
// - (ignore_difficult) If set to true, objects marked as difficult will be ignored when returning
//                          the number of hits for p-r normalization (default = true)
//RETURN VALUE:
//                      Returns the number of object hits in total in the gt to allow proper normalization
//                          of a p-r curve
//NOTES:
// As stated in the VOC documentation, multiple detections of the same object in an image are
// considered FALSE detections e.g. 5 detections of a single object is counted as 1 correct
// detection and 4 false detections - it is the responsibility of the participant's system
// to filter multiple detections from its output
int VocData::getDetectorGroundTruth(const string& obj_class, const ObdDatasetType dataset, const vector<ObdImage>& images, const vector<vector<Rect> >& bounding_boxes, const vector<vector<float> >& scores, vector<vector<char> >& ground_truth, vector<vector<char> >& detection_difficult, bool ignore_difficult)
{
    int recall_normalization = 0;

    /* first create a list of indices referring to the elements of bounding_boxes and scores in
     * descending order of scores */
    vector<ObdScoreIndexSorter> sorted_ids;
    {
        /* first count how many objects to allow preallocation */
        size_t obj_count = 0;
        CV_Assert(images.size() == bounding_boxes.size());
        CV_Assert(scores.size() == bounding_boxes.size());
        for (size_t im_idx = 0; im_idx < scores.size(); ++im_idx)
        {
            CV_Assert(scores[im_idx].size() == bounding_boxes[im_idx].size());
            obj_count += scores[im_idx].size();
        }
        /* preallocate id vector */
        sorted_ids.resize(obj_count);
        /* now copy across scores and indexes to preallocated vector */
        int flat_pos = 0;
        for (size_t im_idx = 0; im_idx < scores.size(); ++im_idx)
        {
            for (size_t ob_idx = 0; ob_idx < scores[im_idx].size(); ++ob_idx)
            {
                sorted_ids[flat_pos].score = scores[im_idx][ob_idx];
                sorted_ids[flat_pos].image_idx = (int)im_idx;
                sorted_ids[flat_pos].obj_idx = (int)ob_idx;
                ++flat_pos;
            }
        }
        /* and sort the vector in descending order of score */
        std::sort(sorted_ids.begin(),sorted_ids.end());
        std::reverse(sorted_ids.begin(),sorted_ids.end());
    }

    /* prepare ground truth + difficult vector (1st dimension) */
    vector<vector<char> >(images.size()).swap(ground_truth);
    vector<vector<char> >(images.size()).swap(detection_difficult);
    vector<vector<char> > detected(images.size());

    vector<vector<ObdObject> > img_objects(images.size());
    vector<vector<VocObjectData> > img_object_data(images.size());
    /* preload object ground truth bounding box data */
    {
        vector<vector<ObdObject> > img_objects_all(images.size());
        vector<vector<VocObjectData> > img_object_data_all(images.size());
        for (size_t image_idx = 0; image_idx < images.size(); ++image_idx)
        {
            /* prepopulate ground truth bounding boxes */
            getObjects(images[image_idx].id, img_objects_all[image_idx], img_object_data_all[image_idx]);
            /* meanwhile, also set length of target ground truth + difficult vector to same as number of object detections (2nd dimension) */
            ground_truth[image_idx].resize(bounding_boxes[image_idx].size());
            detection_difficult[image_idx].resize(bounding_boxes[image_idx].size());
        }

        /* save only instances of the object class concerned */
        for (size_t image_idx = 0; image_idx < images.size(); ++image_idx)
        {
            for (size_t obj_idx = 0; obj_idx < img_objects_all[image_idx].size(); ++obj_idx)
            {
                if (img_objects_all[image_idx][obj_idx].object_class == obj_class)
                {
                    img_objects[image_idx].push_back(img_objects_all[image_idx][obj_idx]);
                    img_object_data[image_idx].push_back(img_object_data_all[image_idx][obj_idx]);
                }
            }
            detected[image_idx].resize(img_objects[image_idx].size(), false);
        }
    }

    /* calculate the total number of objects in the ground truth for the current dataset */
    {
        vector<ObdImage> gt_images;
        vector<char> gt_object_present;
        getClassImages(obj_class, dataset, gt_images, gt_object_present);

        for (size_t image_idx = 0; image_idx < gt_images.size(); ++image_idx)
        {
            vector<ObdObject> gt_img_objects;
            vector<VocObjectData> gt_img_object_data;
            getObjects(gt_images[image_idx].id, gt_img_objects, gt_img_object_data);
            for (size_t obj_idx = 0; obj_idx < gt_img_objects.size(); ++obj_idx)
            {
                if (gt_img_objects[obj_idx].object_class == obj_class)
                {
                    if ((gt_img_object_data[obj_idx].difficult == false) || (ignore_difficult == false))
                        ++recall_normalization;
                }
            }
        }
    }

#ifdef PR_DEBUG
    int printed_count = 0;
#endif
    /* now iterate through detections in descending order of score, assigning to ground truth bounding boxes if possible */
    for (size_t detect_idx = 0; detect_idx < sorted_ids.size(); ++detect_idx)
    {
        //read in indexes to make following code easier to read
        int im_idx = sorted_ids[detect_idx].image_idx;
        int ob_idx = sorted_ids[detect_idx].obj_idx;
        //set ground truth for the current object to false by default
        ground_truth[im_idx][ob_idx] = false;
        detection_difficult[im_idx][ob_idx] = false;
        float maxov = -1.0;
        bool max_is_difficult = false;
        int max_gt_obj_idx = -1;
        //-- for each detected object iterate through objects present in the bounding box ground truth --
        for (size_t gt_obj_idx = 0; gt_obj_idx < img_objects[im_idx].size(); ++gt_obj_idx)
        {
            if (detected[im_idx][gt_obj_idx] == false)
            {
                //check if the detected object and ground truth object overlap by a sufficient margin
                float ov = testBoundingBoxesForOverlap(bounding_boxes[im_idx][ob_idx], img_objects[im_idx][gt_obj_idx].boundingBox);
                if (ov != -1.0)
                {
                    //if all conditions are met store the overlap score and index (as objects are assigned to the highest scoring match)
                    if (ov > maxov)
                    {
                        maxov = ov;
                        max_gt_obj_idx = (int)gt_obj_idx;
                        //store whether the maximum detection is marked as difficult or not
                        max_is_difficult = (img_object_data[im_idx][gt_obj_idx].difficult);
                    }
                }
            }
        }
        //-- if a match was found, set the ground truth of the current object to true --
        if (maxov != -1.0)
        {
            CV_Assert(max_gt_obj_idx != -1);
            ground_truth[im_idx][ob_idx] = true;
            //store whether the maximum detection was marked as 'difficult' or not
            detection_difficult[im_idx][ob_idx] = max_is_difficult;
            //remove the ground truth object so it doesn't match with subsequent detected objects
            //** this is the behaviour defined by the voc documentation **
            detected[im_idx][max_gt_obj_idx] = true;
        }
#ifdef PR_DEBUG
        if (printed_count < 10)
        {
            cout << printed_count << ": id=" << images[im_idx].id << ", score=" << scores[im_idx][ob_idx] << " (" << ob_idx << ") [" << bounding_boxes[im_idx][ob_idx].x << "," <<
                    bounding_boxes[im_idx][ob_idx].y << "," << bounding_boxes[im_idx][ob_idx].width + bounding_boxes[im_idx][ob_idx].x <<
                    "," << bounding_boxes[im_idx][ob_idx].height + bounding_boxes[im_idx][ob_idx].y << "] detected=" << ground_truth[im_idx][ob_idx] <<
                    ", difficult=" << detection_difficult[im_idx][ob_idx] << endl;
            ++printed_count;
            /* print ground truth */
            for (int gt_obj_idx = 0; gt_obj_idx < img_objects[im_idx].size(); ++gt_obj_idx)
            {
                cout << "    GT: [" << img_objects[im_idx][gt_obj_idx].boundingBox.x << "," <<
                        img_objects[im_idx][gt_obj_idx].boundingBox.y << "," << img_objects[im_idx][gt_obj_idx].boundingBox.width + img_objects[im_idx][gt_obj_idx].boundingBox.x <<
                        "," << img_objects[im_idx][gt_obj_idx].boundingBox.height + img_objects[im_idx][gt_obj_idx].boundingBox.y << "]";
                if (gt_obj_idx == max_gt_obj_idx) cout << " <--- (" << maxov << " overlap)";
                cout << endl;
            }
        }
#endif
    }

    return recall_normalization;
}

//Write VOC-compliant classifier results file
//-------------------------------------------
//INPUTS:
// - obj_class          The VOC object class identifier string
// - dataset            Specifies whether working with the training or test set
// - images             An array of ObdImage containing the images for which data will be saved to the result file
// - scores             A corresponding array of confidence scores given a query
// - (competition)      If specified, defines which competition the results are for (see VOC documentation - default 1)
//NOTES:
// The result file path and filename are determined automatically using m_results_directory as a base
void VocData::writeClassifierResultsFile( const string& out_dir, const string& obj_class, const ObdDatasetType dataset, const vector<ObdImage>& images, const vector<float>& scores, const int competition, const bool overwrite_ifexists)
{
    CV_Assert(images.size() == scores.size());

    string output_file_base, output_file;
    if (dataset == CV_OBD_TRAIN)
    {
        output_file_base = out_dir + "/comp" + integerToString(competition) + "_cls_" + m_train_set + "_" + obj_class;
    } else {
        output_file_base = out_dir + "/comp" + integerToString(competition) + "_cls_" + m_test_set + "_" + obj_class;
    }
    output_file = output_file_base + ".txt";

    //check if file exists, and if so create a numbered new file instead
    if (overwrite_ifexists == false)
    {
        struct stat stFileInfo;
        if (stat(output_file.c_str(),&stFileInfo) == 0)
        {
            string output_file_new;
            int filenum = 0;
            do
            {
                ++filenum;
                output_file_new = output_file_base + "_" + integerToString(filenum);
                output_file = output_file_new + ".txt";
            } while (stat(output_file.c_str(),&stFileInfo) == 0);
        }
    }

    //output data to file
    std::ofstream result_file(output_file.c_str());
    if (result_file.is_open())
    {
        for (size_t i = 0; i < images.size(); ++i)
        {
            result_file << images[i].id << " " << scores[i] << endl;
        }
        result_file.close();
    } else {
        string err_msg = "could not open classifier results file '" + output_file + "' for writing. Before running for the first time, a 'results' subdirectory should be created within the VOC dataset base directory. e.g. if the VOC data is stored in /VOC/VOC2010 then the path /VOC/results must be created.";
        CV_Error(CV_StsError,err_msg.c_str());
    }
}

//---------------------------------------
//CALCULATE METRICS FROM VOC RESULTS DATA
//---------------------------------------

//Utility function to construct a VOC-standard classification results filename
//----------------------------------------------------------------------------
//INPUTS:
// - obj_class          The VOC object class identifier string
// - task               Specifies whether to generate a filename for the classification or detection task
// - dataset            Specifies whether working with the training or test set
// - (competition)      If specified, defines which competition the results are for (see VOC documentation
//                      default of -1 means this is set to 1 for the classification task and 3 for the detection task)
// - (number)           If specified and above 0, defines which of a number of duplicate results file produced for a given set of
//                      of settings should be used (this number will be added as a postfix to the filename)
//NOTES:
// This is primarily useful for returning the filename of a classification file previously computed using writeClassifierResultsFile
// for example when calling calcClassifierPrecRecall
string VocData::getResultsFilename(const string& obj_class, const VocTask task, const ObdDatasetType dataset, const int competition, const int number)
{
    if ((competition < 1) && (competition != -1))
        CV_Error(CV_StsBadArg,"competition argument should be a positive non-zero number or -1 to accept the default");
    if ((number < 1) && (number != -1))
        CV_Error(CV_StsBadArg,"number argument should be a positive non-zero number or -1 to accept the default");

    string dset, task_type;

    if (dataset == CV_OBD_TRAIN)
    {
        dset = m_train_set;
    } else {
        dset = m_test_set;
    }

    int comp = competition;
    if (task == CV_VOC_TASK_CLASSIFICATION)
    {
        task_type = "cls";
        if (comp == -1) comp = 1;
    } else {
        task_type = "det";
        if (comp == -1) comp = 3;
    }

    stringstream ss;
    if (number < 1)
    {
        ss << "comp" << comp << "_" << task_type << "_" << dset << "_" << obj_class << ".txt";
    } else {
        ss << "comp" << comp << "_" << task_type << "_" << dset << "_" << obj_class << "_" << number << ".txt";
    }

    string filename = ss.str();
    return filename;
}

//Calculate metrics for classification results
//--------------------------------------------
//INPUTS:
// - ground_truth       A vector of booleans determining whether the currently tested class is present in each input image
// - scores             A vector containing the similarity score for each input image (higher is more similar)
//OUTPUTS:
// - precision          A vector containing the precision calculated at each datapoint of a p-r curve generated from the result set
// - recall             A vector containing the recall calculated at each datapoint of a p-r curve generated from the result set
// - ap                The ap metric calculated from the result set
// - (ranking)          A vector of the same length as 'ground_truth' and 'scores' containing the order of the indices in both of
//                      these arrays when sorting by the ranking score in descending order
//NOTES:
// The result file path and filename are determined automatically using m_results_directory as a base
void VocData::calcClassifierPrecRecall(const string& obj_class, const vector<ObdImage>& images, const vector<float>& scores, vector<float>& precision, vector<float>& recall, float& ap, vector<size_t>& ranking)
{
    vector<char> res_ground_truth;
    getClassifierGroundTruth(obj_class, images, res_ground_truth);

    calcPrecRecall_impl(res_ground_truth, scores, precision, recall, ap, ranking);
}

void VocData::calcClassifierPrecRecall(const string& obj_class, const vector<ObdImage>& images, const vector<float>& scores, vector<float>& precision, vector<float>& recall, float& ap)
{
    vector<char> res_ground_truth;
    getClassifierGroundTruth(obj_class, images, res_ground_truth);

    vector<size_t> ranking;
    calcPrecRecall_impl(res_ground_truth, scores, precision, recall, ap, ranking);
}

//< Overloaded version which accepts VOC classification result file input instead of array of scores/ground truth >
//INPUTS:
// - input_file         The path to the VOC standard results file to use for calculating precision/recall
//                      If a full path is not specified, it is assumed this file is in the VOC standard results directory
//                      A VOC standard filename can be retrieved (as used by writeClassifierResultsFile) by calling  getClassifierResultsFilename

void VocData::calcClassifierPrecRecall(const string& input_file, vector<float>& precision, vector<float>& recall, float& ap, bool outputRankingFile)
{
    //read in classification results file
    vector<string> res_image_codes;
    vector<float> res_scores;

    string input_file_std = checkFilenamePathsep(input_file);
    readClassifierResultsFile(input_file_std, res_image_codes, res_scores);

    //extract the object class and dataset from the results file filename
    string class_name, dataset_name;
    extractDataFromResultsFilename(input_file_std, class_name, dataset_name);

    //generate the ground truth for the images extracted from the results file
    vector<char> res_ground_truth;

    getClassifierGroundTruth(class_name, res_image_codes, res_ground_truth);

    if (outputRankingFile)
    {
        /* 1. store sorting order by score (descending) in 'order' */
        vector<std::pair<size_t, vector<float>::const_iterator> > order(res_scores.size());

        size_t n = 0;
        for (vector<float>::const_iterator it = res_scores.begin(); it != res_scores.end(); ++it, ++n)
            order[n] = make_pair(n, it);

        std::sort(order.begin(),order.end(),orderingSorter());

        /* 2. save ranking results to text file */
        string input_file_std = checkFilenamePathsep(input_file);
        size_t fnamestart = input_file_std.rfind("/");
        string scoregt_file_str = input_file_std.substr(0,fnamestart+1) + "scoregt_" + class_name + ".txt";
        std::ofstream scoregt_file(scoregt_file_str.c_str());
        if (scoregt_file.is_open())
        {
            for (size_t i = 0; i < res_scores.size(); ++i)
            {
                scoregt_file << res_image_codes[order[i].first] << " " << res_scores[order[i].first] << " " << res_ground_truth[order[i].first] << endl;
            }
            scoregt_file.close();
        } else {
            string err_msg = "could not open scoregt file '" + scoregt_file_str + "' for writing.";
            CV_Error(CV_StsError,err_msg.c_str());
        }
    }

    //finally, calculate precision+recall+ap
    vector<size_t> ranking;
    calcPrecRecall_impl(res_ground_truth,res_scores,precision,recall,ap,ranking);
}

//< Protected implementation of Precision-Recall calculation used by both calcClassifierPrecRecall and calcDetectorPrecRecall >

void VocData::calcPrecRecall_impl(const vector<char>& ground_truth, const vector<float>& scores, vector<float>& precision, vector<float>& recall, float& ap, vector<size_t>& ranking, int recall_normalization)
{
    CV_Assert(ground_truth.size() == scores.size());

    //add extra element for p-r at 0 recall (in case that first retrieved is positive)
    vector<float>(scores.size()+1).swap(precision);
    vector<float>(scores.size()+1).swap(recall);

    // SORT RESULTS BY THEIR SCORE
    /* 1. store sorting order in 'order' */
    VocData::getSortOrder(scores, ranking);

#ifdef PR_DEBUG
    std::ofstream scoregt_file("D:/pr.txt");
    if (scoregt_file.is_open())
    {
       for (int i = 0; i < scores.size(); ++i)
       {
           scoregt_file << scores[ranking[i]] << " " << ground_truth[ranking[i]] << endl;
       }
       scoregt_file.close();
    }
#endif

    // CALCULATE PRECISION+RECALL

    int retrieved_hits = 0;

    int recall_norm;
    if (recall_normalization != -1)
    {
        recall_norm = recall_normalization;
    } else {
        recall_norm = (int)std::count_if(ground_truth.begin(),ground_truth.end(),std::bind2nd(std::equal_to<bool>(),true));
    }

    ap = 0;
    recall[0] = 0;
    for (size_t idx = 0; idx < ground_truth.size(); ++idx)
    {
        if (ground_truth[ranking[idx]] != 0) ++retrieved_hits;

        precision[idx+1] = static_cast<float>(retrieved_hits)/static_cast<float>(idx+1);
        recall[idx+1] = static_cast<float>(retrieved_hits)/static_cast<float>(recall_norm);

        if (idx == 0)
        {
            //add further point at 0 recall with the same precision value as the first computed point
            precision[idx] = precision[idx+1];
        }
        if (recall[idx+1] == 1.0)
        {
            //if recall = 1, then end early as all positive images have been found
            recall.resize(idx+2);
            precision.resize(idx+2);
            break;
        }
    }

    /* ap calculation */
    if (m_sampled_ap == false)
    {
        // FOR VOC2010+ AP IS CALCULATED FROM ALL DATAPOINTS
        /* make precision monotonically decreasing for purposes of calculating ap */
        vector<float> precision_monot(precision.size());
        vector<float>::iterator prec_m_it = precision_monot.begin();
        for (vector<float>::iterator prec_it = precision.begin(); prec_it != precision.end(); ++prec_it, ++prec_m_it)
        {
            vector<float>::iterator max_elem;
            max_elem = std::max_element(prec_it,precision.end());
            (*prec_m_it) = (*max_elem);
        }
        /* calculate ap */
        for (size_t idx = 0; idx < (recall.size()-1); ++idx)
        {
            ap += (recall[idx+1] - recall[idx])*precision_monot[idx+1] +   //no need to take min of prec - is monotonically decreasing
                    0.5f*(recall[idx+1] - recall[idx])*std::abs(precision_monot[idx+1] - precision_monot[idx]);
        }
    } else {
        // FOR BEFORE VOC2010 AP IS CALCULATED BY SAMPLING PRECISION AT RECALL 0.0,0.1,..,1.0

        for (float recall_pos = 0.f; recall_pos <= 1.f; recall_pos += 0.1f)
        {
            //find iterator of the precision corresponding to the first recall >= recall_pos
            vector<float>::iterator recall_it = recall.begin();
            vector<float>::iterator prec_it = precision.begin();

            while ((*recall_it) < recall_pos)
            {
                ++recall_it;
                ++prec_it;
                if (recall_it == recall.end()) break;
            }

            /* if no recall >= recall_pos found, this level of recall is never reached so stop adding to ap */
            if (recall_it == recall.end()) break;

            /* if the prec_it is valid, compute the max precision at this level of recall or higher */
            vector<float>::iterator max_prec = std::max_element(prec_it,precision.end());

            ap += (*max_prec)/11;
        }
    }
}

/* functions for calculating confusion matrix rows */

//Calculate rows of a confusion matrix
//------------------------------------
//INPUTS:
// - obj_class          The VOC object class identifier string for the confusion matrix row to compute
// - images             An array of ObdImage containing the images to use for the computation
// - scores             A corresponding array of confidence scores for the presence of obj_class in each image
// - cond               Defines whether to use a cut off point based on recall (CV_VOC_CCOND_RECALL) or score
//                      (CV_VOC_CCOND_SCORETHRESH) the latter is useful for classifier detections where positive
//                      values are positive detections and negative values are negative detections
// - threshold          Threshold value for cond. In case of CV_VOC_CCOND_RECALL, is proportion recall (e.g. 0.5).
//                      In the case of CV_VOC_CCOND_SCORETHRESH is the value above which to count results.
//OUTPUTS:
// - output_headers     An output vector of object class headers for the confusion matrix row
// - output_values      An output vector of values for the confusion matrix row corresponding to the classes
//                      defined in output_headers
//NOTES:
// The methodology used by the classifier version of this function is that true positives have a single unit
// added to the obj_class column in the confusion matrix row, whereas false positives have a single unit
// distributed in proportion between all the columns in the confusion matrix row corresponding to the objects
// present in the image.
void VocData::calcClassifierConfMatRow(const string& obj_class, const vector<ObdImage>& images, const vector<float>& scores, const VocConfCond cond, const float threshold, vector<string>& output_headers, vector<float>& output_values)
{
    CV_Assert(images.size() == scores.size());

    // SORT RESULTS BY THEIR SCORE
    /* 1. store sorting order in 'ranking' */
    vector<size_t> ranking;
    VocData::getSortOrder(scores, ranking);

    // CALCULATE CONFUSION MATRIX ENTRIES
    /* prepare object category headers */
    output_headers = m_object_classes;
    vector<float>(output_headers.size(),0.0).swap(output_values);
    /* find the index of the target object class in the headers for later use */
    int target_idx;
    {
        vector<string>::iterator target_idx_it = std::find(output_headers.begin(),output_headers.end(),obj_class);
        /* if the target class can not be found, raise an exception */
        if (target_idx_it == output_headers.end())
        {
            string err_msg = "could not find the target object class '" + obj_class + "' in list of valid classes.";
            CV_Error(CV_StsError,err_msg.c_str());
        }
        /* convert iterator to index */
        target_idx = std::distance(output_headers.begin(),target_idx_it);
    }

    /* prepare variables related to calculating recall if using the recall threshold */
    int retrieved_hits = 0;
    int total_relevant = 0;
    if (cond == CV_VOC_CCOND_RECALL)
    {
        vector<char> ground_truth;
        /* in order to calculate the total number of relevant images for normalization of recall
            it's necessary to extract the ground truth for the images under consideration */
        getClassifierGroundTruth(obj_class, images, ground_truth);
        total_relevant = std::count_if(ground_truth.begin(),ground_truth.end(),std::bind2nd(std::equal_to<bool>(),true));
    }

    /* iterate through images */
    vector<ObdObject> img_objects;
    vector<VocObjectData> img_object_data;
    int total_images = 0;
    for (size_t image_idx = 0; image_idx < images.size(); ++image_idx)
    {
        /* if using the score as the break condition, check for it now */
        if (cond == CV_VOC_CCOND_SCORETHRESH)
        {
            if (scores[ranking[image_idx]] <= threshold) break;
        }
        /* if continuing for this iteration, increment the image counter for later normalization */
        ++total_images;
        /* for each image retrieve the objects contained */
        getObjects(images[ranking[image_idx]].id, img_objects, img_object_data);
        //check if the tested for object class is present
        if (getClassifierGroundTruthImage(obj_class, images[ranking[image_idx]].id))
        {
            //if the target class is present, assign fully to the target class element in the confusion matrix row
            output_values[target_idx] += 1.0;
            if (cond == CV_VOC_CCOND_RECALL) ++retrieved_hits;
        } else {
            //first delete all objects marked as difficult
            for (size_t obj_idx = 0; obj_idx < img_objects.size(); ++obj_idx)
            {
                if (img_object_data[obj_idx].difficult == true)
                {
                    vector<ObdObject>::iterator it1 = img_objects.begin();
                    std::advance(it1,obj_idx);
                    img_objects.erase(it1);
                    vector<VocObjectData>::iterator it2 = img_object_data.begin();
                    std::advance(it2,obj_idx);
                    img_object_data.erase(it2);
                    --obj_idx;
                }
            }
            //if the target class is not present, add values to the confusion matrix row in equal proportions to all objects present in the image
            for (size_t obj_idx = 0; obj_idx < img_objects.size(); ++obj_idx)
            {
                //find the index of the currently considered object
                vector<string>::iterator class_idx_it = std::find(output_headers.begin(),output_headers.end(),img_objects[obj_idx].object_class);
                //if the class name extracted from the ground truth file could not be found in the list of available classes, raise an exception
                if (class_idx_it == output_headers.end())
                {
                    string err_msg = "could not find object class '" + img_objects[obj_idx].object_class + "' specified in the ground truth file of '" + images[ranking[image_idx]].id +"'in list of valid classes.";
                    CV_Error(CV_StsError,err_msg.c_str());
                }
                /* convert iterator to index */
                int class_idx = std::distance(output_headers.begin(),class_idx_it);
                //add to confusion matrix row in proportion
                output_values[class_idx] += 1.f/static_cast<float>(img_objects.size());
            }
        }
        //check break conditions if breaking on certain level of recall
        if (cond == CV_VOC_CCOND_RECALL)
        {
            if(static_cast<float>(retrieved_hits)/static_cast<float>(total_relevant) >= threshold) break;
        }
    }
    /* finally, normalize confusion matrix row */
    for (vector<float>::iterator it = output_values.begin(); it < output_values.end(); ++it)
    {
        (*it) /= static_cast<float>(total_images);
    }
}

// NOTE: doesn't ignore repeated detections
void VocData::calcDetectorConfMatRow(const string& obj_class, const ObdDatasetType dataset, const vector<ObdImage>& images, const vector<vector<float> >& scores, const vector<vector<Rect> >& bounding_boxes, const VocConfCond cond, const float threshold, vector<string>& output_headers, vector<float>& output_values, bool ignore_difficult)
{
    CV_Assert(images.size() == scores.size());
    CV_Assert(images.size() == bounding_boxes.size());

    //collapse scores and ground_truth vectors into 1D vectors to allow ranking
    /* define final flat vectors */
    vector<string> images_flat;
    vector<float> scores_flat;
    vector<Rect> bounding_boxes_flat;
    {
        /* first count how many objects to allow preallocation */
        int obj_count = 0;
        CV_Assert(scores.size() == bounding_boxes.size());
        for (size_t img_idx = 0; img_idx < scores.size(); ++img_idx)
        {
            CV_Assert(scores[img_idx].size() == bounding_boxes[img_idx].size());
            for (size_t obj_idx = 0; obj_idx < scores[img_idx].size(); ++obj_idx)
            {
                ++obj_count;
            }
        }
        /* preallocate vectors */
        images_flat.resize(obj_count);
        scores_flat.resize(obj_count);
        bounding_boxes_flat.resize(obj_count);
        /* now copy across to preallocated vectors */
        int flat_pos = 0;
        for (size_t img_idx = 0; img_idx < scores.size(); ++img_idx)
        {
            for (size_t obj_idx = 0; obj_idx < scores[img_idx].size(); ++obj_idx)
            {
                images_flat[flat_pos] = images[img_idx].id;
                scores_flat[flat_pos] = scores[img_idx][obj_idx];
                bounding_boxes_flat[flat_pos] = bounding_boxes[img_idx][obj_idx];
                ++flat_pos;
            }
        }
    }

    // SORT RESULTS BY THEIR SCORE
    /* 1. store sorting order in 'ranking' */
    vector<size_t> ranking;
    VocData::getSortOrder(scores_flat, ranking);

    // CALCULATE CONFUSION MATRIX ENTRIES
    /* prepare object category headers */
    output_headers = m_object_classes;
    output_headers.push_back("background");
    vector<float>(output_headers.size(),0.0).swap(output_values);

    /* prepare variables related to calculating recall if using the recall threshold */
    int retrieved_hits = 0;
    int total_relevant = 0;
    if (cond == CV_VOC_CCOND_RECALL)
    {
//        vector<char> ground_truth;
//        /* in order to calculate the total number of relevant images for normalization of recall
//            it's necessary to extract the ground truth for the images under consideration */
//        getClassifierGroundTruth(obj_class, images, ground_truth);
//        total_relevant = std::count_if(ground_truth.begin(),ground_truth.end(),std::bind2nd(std::equal_to<bool>(),true));
        /* calculate the total number of objects in the ground truth for the current dataset */
        vector<ObdImage> gt_images;
        vector<char> gt_object_present;
        getClassImages(obj_class, dataset, gt_images, gt_object_present);

        for (size_t image_idx = 0; image_idx < gt_images.size(); ++image_idx)
        {
            vector<ObdObject> gt_img_objects;
            vector<VocObjectData> gt_img_object_data;
            getObjects(gt_images[image_idx].id, gt_img_objects, gt_img_object_data);
            for (size_t obj_idx = 0; obj_idx < gt_img_objects.size(); ++obj_idx)
            {
                if (gt_img_objects[obj_idx].object_class == obj_class)
                {
                    if ((gt_img_object_data[obj_idx].difficult == false) || (ignore_difficult == false))
                        ++total_relevant;
                }
            }
        }
    }

    /* iterate through objects */
    vector<ObdObject> img_objects;
    vector<VocObjectData> img_object_data;
    int total_objects = 0;
    for (size_t image_idx = 0; image_idx < images.size(); ++image_idx)
    {
        /* if using the score as the break condition, check for it now */
        if (cond == CV_VOC_CCOND_SCORETHRESH)
        {
            if (scores_flat[ranking[image_idx]] <= threshold) break;
        }
        /* increment the image counter for later normalization */
        ++total_objects;
        /* for each image retrieve the objects contained */
        getObjects(images[ranking[image_idx]].id, img_objects, img_object_data);

        //find the ground truth object which has the highest overlap score with the detected object
        float maxov = -1.0;
        int max_gt_obj_idx = -1;
        //-- for each detected object iterate through objects present in ground truth --
        for (size_t gt_obj_idx = 0; gt_obj_idx < img_objects.size(); ++gt_obj_idx)
        {
            //check difficulty flag
            if (ignore_difficult || (img_object_data[gt_obj_idx].difficult == false))
            {
                //if the class matches, then check if the detected object and ground truth object overlap by a sufficient margin
                float ov = testBoundingBoxesForOverlap(bounding_boxes_flat[ranking[image_idx]], img_objects[gt_obj_idx].boundingBox);
                if (ov != -1.f)
                {
                    //if all conditions are met store the overlap score and index (as objects are assigned to the highest scoring match)
                    if (ov > maxov)
                    {
                        maxov = ov;
                        max_gt_obj_idx = gt_obj_idx;
                    }
                }
            }
        }

        //assign to appropriate object class if an object was detected
        if (maxov != -1.0)
        {
            //find the index of the currently considered object
            vector<string>::iterator class_idx_it = std::find(output_headers.begin(),output_headers.end(),img_objects[max_gt_obj_idx].object_class);
            //if the class name extracted from the ground truth file could not be found in the list of available classes, raise an exception
            if (class_idx_it == output_headers.end())
            {
                string err_msg = "could not find object class '" + img_objects[max_gt_obj_idx].object_class + "' specified in the ground truth file of '" + images[ranking[image_idx]].id +"'in list of valid classes.";
                CV_Error(CV_StsError,err_msg.c_str());
            }
            /* convert iterator to index */
            int class_idx = std::distance(output_headers.begin(),class_idx_it);
            //add to confusion matrix row in proportion
            output_values[class_idx] += 1.0;
        } else {
            //otherwise assign to background class
            output_values[output_values.size()-1] += 1.0;
        }

        //check break conditions if breaking on certain level of recall
        if (cond == CV_VOC_CCOND_RECALL)
        {
            if(static_cast<float>(retrieved_hits)/static_cast<float>(total_relevant) >= threshold) break;
        }
    }

    /* finally, normalize confusion matrix row */
    for (vector<float>::iterator it = output_values.begin(); it < output_values.end(); ++it)
    {
        (*it) /= static_cast<float>(total_objects);
    }
}

//Save Precision-Recall results to a p-r curve in GNUPlot format
//--------------------------------------------------------------
//INPUTS:
// - output_file        The file to which to save the GNUPlot data file. If only a filename is specified, the data
//                      file is saved to the standard VOC results directory.
// - precision          Vector of precisions as returned from calcClassifier/DetectorPrecRecall
// - recall             Vector of recalls as returned from calcClassifier/DetectorPrecRecall
// - ap                ap as returned from calcClassifier/DetectorPrecRecall
// - (title)            Title to use for the plot (if not specified, just the ap is printed as the title)
//                      This also specifies the filename of the output file if printing to pdf
// - (plot_type)        Specifies whether to instruct GNUPlot to save to a PDF file (CV_VOC_PLOT_PDF) or directly
//                      to screen (CV_VOC_PLOT_SCREEN) in the datafile
//NOTES:
// The GNUPlot data file can be executed using GNUPlot from the commandline in the following way:
//      >> GNUPlot <output_file>
// This will then display the p-r curve on the screen or save it to a pdf file depending on plot_type

void VocData::savePrecRecallToGnuplot(const string& output_file, const vector<float>& precision, const vector<float>& recall, const float ap, const string title, const VocPlotType plot_type)
{
    string output_file_std = checkFilenamePathsep(output_file);

    //if no directory is specified, by default save the output file in the results directory
//    if (output_file_std.find("/") == output_file_std.npos)
//    {
//        output_file_std = m_results_directory + output_file_std;
//    }

    std::ofstream plot_file(output_file_std.c_str());

    if (plot_file.is_open())
    {
        plot_file << "set xrange [0:1]" << endl;
        plot_file << "set yrange [0:1]" << endl;
        plot_file << "set size square" << endl;
        string title_text = title;
        if (title_text.size() == 0) title_text = "Precision-Recall Curve";
        plot_file << "set title \"" << title_text << " (ap: " << ap << ")\"" << endl;
        plot_file << "set xlabel \"Recall\"" << endl;
        plot_file << "set ylabel \"Precision\"" << endl;
        plot_file << "set style data lines" << endl;
        plot_file << "set nokey" << endl;
        if (plot_type == CV_VOC_PLOT_PNG)
        {
            plot_file << "set terminal png" << endl;
            string pdf_filename;
            if (title.size() != 0)
            {
                pdf_filename = title;
            } else {
                pdf_filename = "prcurve";
            }
            plot_file << "set out \"" << title << ".png\"" << endl;
        }
        plot_file << "plot \"-\" using 1:2" << endl;
        plot_file << "# X Y" << endl;
        CV_Assert(precision.size() == recall.size());
        for (size_t i = 0; i < precision.size(); ++i)
        {
            plot_file << "  " << recall[i] << " " << precision[i] << endl;
        }
        plot_file << "end" << endl;
        if (plot_type == CV_VOC_PLOT_SCREEN)
        {
            plot_file << "pause -1" << endl;
        }
        plot_file.close();
    } else {
        string err_msg = "could not open plot file '" + output_file_std + "' for writing.";
        CV_Error(CV_StsError,err_msg.c_str());
    }
}

void VocData::readClassifierGroundTruth(const string& obj_class, const ObdDatasetType dataset, vector<ObdImage>& images, vector<char>& object_present)
{
    images.clear();

    string gtFilename = m_class_imageset_path;
    gtFilename.replace(gtFilename.find("%s"),2,obj_class);
    if (dataset == CV_OBD_TRAIN)
    {
        gtFilename.replace(gtFilename.find("%s"),2,m_train_set);
    } else {
        gtFilename.replace(gtFilename.find("%s"),2,m_test_set);
    }

    vector<string> image_codes;
    readClassifierGroundTruth(gtFilename, image_codes, object_present);

    convertImageCodesToObdImages(image_codes, images);
}

void VocData::readClassifierResultsFile(const std:: string& input_file, vector<ObdImage>& images, vector<float>& scores)
{
    images.clear();

    string input_file_std = checkFilenamePathsep(input_file);

    //if no directory is specified, by default search for the input file in the results directory
//    if (input_file_std.find("/") == input_file_std.npos)
//    {
//        input_file_std = m_results_directory + input_file_std;
//    }

    vector<string> image_codes;
    readClassifierResultsFile(input_file_std, image_codes, scores);

    convertImageCodesToObdImages(image_codes, images);
}

void VocData::readDetectorResultsFile(const string& input_file, vector<ObdImage>& images, vector<vector<float> >& scores, vector<vector<Rect> >& bounding_boxes)
{
    images.clear();

    string input_file_std = checkFilenamePathsep(input_file);

    //if no directory is specified, by default search for the input file in the results directory
//    if (input_file_std.find("/") == input_file_std.npos)
//    {
//        input_file_std = m_results_directory + input_file_std;
//    }

    vector<string> image_codes;
    readDetectorResultsFile(input_file_std, image_codes, scores, bounding_boxes);

    convertImageCodesToObdImages(image_codes, images);
}

const vector<string>& VocData::getObjectClasses()
{
    return m_object_classes;
}

//string VocData::getResultsDirectory()
//{
//    return m_results_directory;
//}

//---------------------------------------------------------
// Protected Functions ------------------------------------
//---------------------------------------------------------

string getVocName( const string& vocPath )
{
    size_t found = vocPath.rfind( '/' );
    if( found == string::npos )
    {
        found = vocPath.rfind( '\\' );
        if( found == string::npos )
            return vocPath;
    }
    return vocPath.substr(found + 1, vocPath.size() - found);
}

void VocData::initVoc( const string& vocPath, const bool useTestDataset )
{
    initVoc2007to2010( vocPath, useTestDataset );
}

//Initialize file paths and settings for the VOC 2010 dataset
//-----------------------------------------------------------
void VocData::initVoc2007to2010( const string& vocPath, const bool useTestDataset )
{
    //check format of root directory and modify if necessary

    m_vocName = getVocName( vocPath );

    CV_Assert( !m_vocName.compare("VOC2007") || !m_vocName.compare("VOC2008") ||
               !m_vocName.compare("VOC2009") || !m_vocName.compare("VOC2010") );

    m_vocPath = checkFilenamePathsep( vocPath, true );

    if (useTestDataset)
    {
        m_train_set = "trainval";
        m_test_set = "test";
    } else {
        m_train_set = "train";
        m_test_set = "val";
    }

    // initialize main classification/detection challenge paths
    m_annotation_path = m_vocPath + "/Annotations/%s.xml";
    m_image_path = m_vocPath + "/JPEGImages/%s.jpg";
    m_imageset_path = m_vocPath + "/ImageSets/Main/%s.txt";
    m_class_imageset_path = m_vocPath + "/ImageSets/Main/%s_%s.txt";

    //define available object_classes for VOC2010 dataset
    m_object_classes.push_back("aeroplane");
    m_object_classes.push_back("bicycle");
    m_object_classes.push_back("bird");
    m_object_classes.push_back("boat");
    m_object_classes.push_back("bottle");
    m_object_classes.push_back("bus");
    m_object_classes.push_back("car");
    m_object_classes.push_back("cat");
    m_object_classes.push_back("chair");
    m_object_classes.push_back("cow");
    m_object_classes.push_back("diningtable");
    m_object_classes.push_back("dog");
    m_object_classes.push_back("horse");
    m_object_classes.push_back("motorbike");
    m_object_classes.push_back("person");
    m_object_classes.push_back("pottedplant");
    m_object_classes.push_back("sheep");
    m_object_classes.push_back("sofa");
    m_object_classes.push_back("train");
    m_object_classes.push_back("tvmonitor");

    m_min_overlap = 0.5;

    //up until VOC 2010, ap was calculated by sampling p-r curve, not taking complete curve
    m_sampled_ap = ((m_vocName == "VOC2007") || (m_vocName == "VOC2008") || (m_vocName == "VOC2009"));
}

//Read a VOC classification ground truth text file for a given object class and dataset
//-------------------------------------------------------------------------------------
//INPUTS:
// - filename           The path of the text file to read
//OUTPUTS:
// - image_codes        VOC image codes extracted from the GT file in the form 20XX_XXXXXX where the first four
//                          digits specify the year of the dataset, and the last group specifies a unique ID
// - object_present     For each image in the 'image_codes' array, specifies whether the object class described
//                          in the loaded GT file is present or not
void VocData::readClassifierGroundTruth(const string& filename, vector<string>& image_codes, vector<char>& object_present)
{
    image_codes.clear();
    object_present.clear();

    std::ifstream gtfile(filename.c_str());
    if (!gtfile.is_open())
    {
        string err_msg = "could not open VOC ground truth textfile '" + filename + "'.";
        CV_Error(CV_StsError,err_msg.c_str());
    }

    string line;
    string image;
    int obj_present;
    while (!gtfile.eof())
    {
        std::getline(gtfile,line);
        std::istringstream iss(line);
        iss >> image >> obj_present;
        if (!iss.fail())
        {
            image_codes.push_back(image);
            object_present.push_back(obj_present == 1);
        } else {
            if (!gtfile.eof()) CV_Error(CV_StsParseError,"error parsing VOC ground truth textfile.");
        }
    }
    gtfile.close();
}

void VocData::readClassifierResultsFile(const string& input_file, vector<string>& image_codes, vector<float>& scores)
{
    //check if results file exists
    std::ifstream result_file(input_file.c_str());
    if (result_file.is_open())
    {
        string line;
        string image;
        float score;
        //read in the results file
        while (!result_file.eof())
        {
            std::getline(result_file,line);
            std::istringstream iss(line);
            iss >> image >> score;
            if (!iss.fail())
            {
                image_codes.push_back(image);
                scores.push_back(score);
            } else {
                if(!result_file.eof()) CV_Error(CV_StsParseError,"error parsing VOC classifier results file.");
            }
        }
        result_file.close();
    } else {
        string err_msg = "could not open classifier results file '" + input_file + "' for reading.";
        CV_Error(CV_StsError,err_msg.c_str());
    }
}

void VocData::readDetectorResultsFile(const string& input_file, vector<string>& image_codes, vector<vector<float> >& scores, vector<vector<Rect> >& bounding_boxes)
{
    image_codes.clear();
    scores.clear();
    bounding_boxes.clear();

    //check if results file exists
    std::ifstream result_file(input_file.c_str());
    if (result_file.is_open())
    {
        string line;
        string image;
        Rect bounding_box;
        float score;
        //read in the results file
        while (!result_file.eof())
        {
            std::getline(result_file,line);
            std::istringstream iss(line);
            iss >> image >> score >> bounding_box.x >> bounding_box.y >> bounding_box.width >> bounding_box.height;
            if (!iss.fail())
            {
                //convert right and bottom positions to width and height
                bounding_box.width -= bounding_box.x;
                bounding_box.height -= bounding_box.y;
                //convert to 0-indexing
                bounding_box.x -= 1;
                bounding_box.y -= 1;
                //store in output vectors
                /* first check if the current image code has been seen before */
                vector<string>::iterator image_codes_it = std::find(image_codes.begin(),image_codes.end(),image);
                if (image_codes_it == image_codes.end())
                {
                    image_codes.push_back(image);
                    vector<float> score_vect(1);
                    score_vect[0] = score;
                    scores.push_back(score_vect);
                    vector<Rect> bounding_box_vect(1);
                    bounding_box_vect[0] = bounding_box;
                    bounding_boxes.push_back(bounding_box_vect);
                } else {
                    /* if the image index has been seen before, add the current object below it in the 2D arrays */
                    int image_idx = std::distance(image_codes.begin(),image_codes_it);
                    scores[image_idx].push_back(score);
                    bounding_boxes[image_idx].push_back(bounding_box);
                }
            } else {
                if(!result_file.eof()) CV_Error(CV_StsParseError,"error parsing VOC detector results file.");
            }
        }
        result_file.close();
    } else {
        string err_msg = "could not open detector results file '" + input_file + "' for reading.";
        CV_Error(CV_StsError,err_msg.c_str());
    }
}


//Read a VOC annotation xml file for a given image
//------------------------------------------------
//INPUTS:
// - filename           The path of the xml file to read
//OUTPUTS:
// - objects            Array of VocObject describing all object instances present in the given image
void VocData::extractVocObjects(const string filename, vector<ObdObject>& objects, vector<VocObjectData>& object_data)
{
#ifdef PR_DEBUG
    int block = 1;
    cout << "SAMPLE VOC OBJECT EXTRACTION for " << filename << ":" << endl;
#endif
    objects.clear();
    object_data.clear();

    string contents, object_contents, tag_contents;

    readFileToString(filename, contents);

    //keep on extracting 'object' blocks until no more can be found
    if (extractXMLBlock(contents, "annotation", 0, contents) != -1)
    {
        int searchpos = 0;
        searchpos = extractXMLBlock(contents, "object", searchpos, object_contents);
        while (searchpos != -1)
        {
#ifdef PR_DEBUG
            cout << "SEARCHPOS:" << searchpos << endl;
            cout << "start block " << block << " ---------" << endl;
            cout << object_contents << endl;
            cout << "end block " << block << " -----------" << endl;
            ++block;
#endif

            ObdObject object;
            VocObjectData object_d;

            //object class -------------

            if (extractXMLBlock(object_contents, "name", 0, tag_contents) == -1) CV_Error(CV_StsError,"missing <name> tag in object definition of '" + filename + "'");
            object.object_class.swap(tag_contents);

            //object bounding box -------------

            int xmax, xmin, ymax, ymin;

            if (extractXMLBlock(object_contents, "xmax", 0, tag_contents) == -1) CV_Error(CV_StsError,"missing <xmax> tag in object definition of '" + filename + "'");
            xmax = stringToInteger(tag_contents);

            if (extractXMLBlock(object_contents, "xmin", 0, tag_contents) == -1) CV_Error(CV_StsError,"missing <xmin> tag in object definition of '" + filename + "'");
            xmin = stringToInteger(tag_contents);

            if (extractXMLBlock(object_contents, "ymax", 0, tag_contents) == -1) CV_Error(CV_StsError,"missing <ymax> tag in object definition of '" + filename + "'");
            ymax = stringToInteger(tag_contents);

            if (extractXMLBlock(object_contents, "ymin", 0, tag_contents) == -1) CV_Error(CV_StsError,"missing <ymin> tag in object definition of '" + filename + "'");
            ymin = stringToInteger(tag_contents);

            object.boundingBox.x = xmin-1;      //convert to 0-based indexing
            object.boundingBox.width = xmax - xmin;
            object.boundingBox.y = ymin-1;
            object.boundingBox.height = ymax - ymin;

            CV_Assert(xmin != 0);
            CV_Assert(xmax > xmin);
            CV_Assert(ymin != 0);
            CV_Assert(ymax > ymin);


            //object tags -------------

            if (extractXMLBlock(object_contents, "difficult", 0, tag_contents) != -1)
            {
                object_d.difficult = (tag_contents == "1");
            } else object_d.difficult = false;
            if (extractXMLBlock(object_contents, "occluded", 0, tag_contents) != -1)
            {
                object_d.occluded = (tag_contents == "1");
            } else object_d.occluded = false;
            if (extractXMLBlock(object_contents, "truncated", 0, tag_contents) != -1)
            {
                object_d.truncated = (tag_contents == "1");
            } else object_d.truncated = false;
            if (extractXMLBlock(object_contents, "pose", 0, tag_contents) != -1)
            {
                if (tag_contents == "Frontal") object_d.pose = CV_VOC_POSE_FRONTAL;
                if (tag_contents == "Rear") object_d.pose = CV_VOC_POSE_REAR;
                if (tag_contents == "Left") object_d.pose = CV_VOC_POSE_LEFT;
                if (tag_contents == "Right") object_d.pose = CV_VOC_POSE_RIGHT;
            }

            //add to array of objects
            objects.push_back(object);
            object_data.push_back(object_d);

            //extract next 'object' block from file if it exists
            searchpos = extractXMLBlock(contents, "object", searchpos, object_contents);
        }
    }
}

//Converts an image identifier string in the format YYYY_XXXXXX to a single index integer of form XXXXXXYYYY
//where Y represents a year and returns the image path
//----------------------------------------------------------------------------------------------------------
string VocData::getImagePath(const string& input_str)
{
    string path = m_image_path;
    path.replace(path.find("%s"),2,input_str);
    return path;
}

//Tests two boundary boxes for overlap (using the intersection over union metric) and returns the overlap if the objects
//defined by the two bounding boxes are considered to be matched according to the criterion outlined in
//the VOC documentation [namely intersection/union > some threshold] otherwise returns -1.0 (no match)
//----------------------------------------------------------------------------------------------------------
float VocData::testBoundingBoxesForOverlap(const Rect detection, const Rect ground_truth)
{
    int detection_x2 = detection.x + detection.width;
    int detection_y2 = detection.y + detection.height;
    int ground_truth_x2 = ground_truth.x + ground_truth.width;
    int ground_truth_y2 = ground_truth.y + ground_truth.height;
    //first calculate the boundaries of the intersection of the rectangles
    int intersection_x = std::max(detection.x, ground_truth.x); //rightmost left
    int intersection_y = std::max(detection.y, ground_truth.y); //bottommost top
    int intersection_x2 = std::min(detection_x2, ground_truth_x2); //leftmost right
    int intersection_y2 = std::min(detection_y2, ground_truth_y2); //topmost bottom
    //then calculate the width and height of the intersection rect
    int intersection_width = intersection_x2 - intersection_x + 1;
    int intersection_height = intersection_y2 - intersection_y + 1;
    //if there is no overlap then return false straight away
    if ((intersection_width <= 0) || (intersection_height <= 0)) return -1.0;
    //otherwise calculate the intersection
    int intersection_area = intersection_width*intersection_height;

    //now calculate the union
    int union_area = (detection.width+1)*(detection.height+1) + (ground_truth.width+1)*(ground_truth.height+1) - intersection_area;

    //calculate the intersection over union and use as threshold as per VOC documentation
    float overlap = static_cast<float>(intersection_area)/static_cast<float>(union_area);
    if (overlap > m_min_overlap)
    {
        return overlap;
    } else {
        return -1.0;
    }
}

//Extracts the object class and dataset from the filename of a VOC standard results text file, which takes
//the format 'comp<n>_{cls/det}_<dataset>_<objclass>.txt'
//----------------------------------------------------------------------------------------------------------
void VocData::extractDataFromResultsFilename(const string& input_file, string& class_name, string& dataset_name)
{
    string input_file_std = checkFilenamePathsep(input_file);

    size_t fnamestart = input_file_std.rfind("/");
    size_t fnameend = input_file_std.rfind(".txt");

    if ((fnamestart == input_file_std.npos) || (fnameend == input_file_std.npos))
        CV_Error(CV_StsError,"Could not extract filename of results file.");

    ++fnamestart;
    if (fnamestart >= fnameend)
        CV_Error(CV_StsError,"Could not extract filename of results file.");

    //extract dataset and class names, triggering exception if the filename format is not correct
    string filename = input_file_std.substr(fnamestart, fnameend-fnamestart);
    size_t datasetstart = filename.find("_");
    datasetstart = filename.find("_",datasetstart+1);
    size_t classstart = filename.find("_",datasetstart+1);
    //allow for appended index after a further '_' by discarding this part if it exists
    size_t classend = filename.find("_",classstart+1);
    if (classend == filename.npos) classend = filename.size();
    if ((datasetstart == filename.npos) || (classstart == filename.npos))
        CV_Error(CV_StsError,"Error parsing results filename. Is it in standard format of 'comp<n>_{cls/det}_<dataset>_<objclass>.txt'?");
    ++datasetstart;
    ++classstart;
    if (((datasetstart-classstart) < 1) || ((classend-datasetstart) < 1))
        CV_Error(CV_StsError,"Error parsing results filename. Is it in standard format of 'comp<n>_{cls/det}_<dataset>_<objclass>.txt'?");

    dataset_name = filename.substr(datasetstart,classstart-datasetstart-1);
    class_name = filename.substr(classstart,classend-classstart);
}

bool VocData::getClassifierGroundTruthImage(const string& obj_class, const string& id)
{
    /* if the classifier ground truth data for all images of the current class has not been loaded yet, load it now */
    if (m_classifier_gt_all_ids.empty() || (m_classifier_gt_class != obj_class))
    {
        m_classifier_gt_all_ids.clear();
        m_classifier_gt_all_present.clear();
        m_classifier_gt_class = obj_class;
        for (int i=0; i<2; ++i) //run twice (once over test set and once over training set)
        {
            //generate the filename of the classification ground-truth textfile for the object class
            string gtFilename = m_class_imageset_path;
            gtFilename.replace(gtFilename.find("%s"),2,obj_class);
            if (i == 0)
            {
                gtFilename.replace(gtFilename.find("%s"),2,m_train_set);
            } else {
                gtFilename.replace(gtFilename.find("%s"),2,m_test_set);
            }

            //parse the ground truth file, storing in two separate vectors
            //for the image code and the ground truth value
            vector<string> image_codes;
            vector<char> object_present;
            readClassifierGroundTruth(gtFilename, image_codes, object_present);

            m_classifier_gt_all_ids.insert(m_classifier_gt_all_ids.end(),image_codes.begin(),image_codes.end());
            m_classifier_gt_all_present.insert(m_classifier_gt_all_present.end(),object_present.begin(),object_present.end());

            CV_Assert(m_classifier_gt_all_ids.size() == m_classifier_gt_all_present.size());
        }
    }


    //search for the image code
    vector<string>::iterator it = find (m_classifier_gt_all_ids.begin(), m_classifier_gt_all_ids.end(), id);
    if (it != m_classifier_gt_all_ids.end())
    {
        //image found, so return corresponding ground truth
        return m_classifier_gt_all_present[std::distance(m_classifier_gt_all_ids.begin(),it)] != 0;
    } else {
        string err_msg = "could not find classifier ground truth for image '" + id + "' and class '" + obj_class + "'";
        CV_Error(CV_StsError,err_msg.c_str());
    }

    return true;
}

//-------------------------------------------------------------------
// Protected Functions (utility) ------------------------------------
//-------------------------------------------------------------------

//returns a vector containing indexes of the input vector in sorted ascending/descending order
void VocData::getSortOrder(const vector<float>& values, vector<size_t>& order, bool descending)
{
    /* 1. store sorting order in 'order_pair' */
    vector<std::pair<size_t, vector<float>::const_iterator> > order_pair(values.size());

    size_t n = 0;
    for (vector<float>::const_iterator it = values.begin(); it != values.end(); ++it, ++n)
        order_pair[n] = make_pair(n, it);

    std::sort(order_pair.begin(),order_pair.end(),orderingSorter());
    if (descending == false) std::reverse(order_pair.begin(),order_pair.end());

    vector<size_t>(order_pair.size()).swap(order);
    for (size_t i = 0; i < order_pair.size(); ++i)
    {
        order[i] = order_pair[i].first;
    }
}

void VocData::readFileToString(const string filename, string& file_contents)
{
    std::ifstream ifs(filename.c_str());
    if (ifs == false) CV_Error(CV_StsError,"could not open text file");

    stringstream oss;
    oss << ifs.rdbuf();

    file_contents = oss.str();
}

int VocData::stringToInteger(const string input_str)
{
    int result;

    stringstream ss(input_str);
    if ((ss >> result).fail())
    {
        CV_Error(CV_StsBadArg,"could not perform string to integer conversion");
    }
    return result;
}

string VocData::integerToString(const int input_int)
{
    string result;

    stringstream ss;
    if ((ss << input_int).fail())
    {
        CV_Error(CV_StsBadArg,"could not perform integer to string conversion");
    }
    result = ss.str();
    return result;
}

string VocData::checkFilenamePathsep( const string filename, bool add_trailing_slash )
{
    string filename_new = filename;

    size_t pos = filename_new.find("\\\\");
    while (pos != filename_new.npos)
    {
        filename_new.replace(pos,2,"/");
        pos = filename_new.find("\\\\", pos);
    }
    pos = filename_new.find("\\");
    while (pos != filename_new.npos)
    {
        filename_new.replace(pos,2,"/");
        pos = filename_new.find("\\", pos);
    }
    if (add_trailing_slash)
    {
        //add training slash if this is missing
        if (filename_new.rfind("/") != filename_new.length()-1) filename_new += "/";
    }

    return filename_new;
}

void VocData::convertImageCodesToObdImages(const vector<string>& image_codes, vector<ObdImage>& images)
{
    images.clear();
    images.reserve(image_codes.size());

    string path;
    //transfer to output arrays
    for (size_t i = 0; i < image_codes.size(); ++i)
    {
        //generate image path and indices from extracted string code
        path = getImagePath(image_codes[i]);
        images.push_back(ObdImage(image_codes[i], path));
    }
}

//Extract text from within a given tag from an XML file
//-----------------------------------------------------
//INPUTS:
// - src            XML source file
// - tag            XML tag delimiting block to extract
// - searchpos      position within src at which to start search
//OUTPUTS:
// - tag_contents   text extracted between <tag> and </tag> tags
//RETURN VALUE:
// - the position of the final character extracted in tag_contents within src
//      (can be used to call extractXMLBlock recursively to extract multiple blocks)
//      returns -1 if the tag could not be found
int VocData::extractXMLBlock(const string src, const string tag, const int searchpos, string& tag_contents)
{
    size_t startpos, next_startpos, endpos;
    int embed_count = 1;

    //find position of opening tag
    startpos = src.find("<" + tag + ">", searchpos);
    if (startpos == string::npos) return -1;

    //initialize endpos -
    // start searching for end tag anywhere after opening tag
    endpos = startpos;

    //find position of next opening tag
    next_startpos = src.find("<" + tag + ">", startpos+1);

    //match opening tags with closing tags, and only
    //accept final closing tag of same level as original
    //opening tag
    while (embed_count > 0)
    {
        endpos = src.find("</" + tag + ">", endpos+1);
        if (endpos == string::npos) return -1;

        //the next code is only executed if there are embedded tags with the same name
        if (next_startpos != string::npos)
        {
            while (next_startpos<endpos)
            {
                //counting embedded start tags
                ++embed_count;
                next_startpos = src.find("<" + tag + ">", next_startpos+1);
                if (next_startpos == string::npos) break;
            }
        }
        //passing end tag so decrement nesting level
        --embed_count;
    }

    //finally, extract the tag region
    startpos += tag.length() + 2;
    if (startpos > src.length()) return -1;
    if (endpos > src.length()) return -1;
    tag_contents = src.substr(startpos,endpos-startpos);
    return static_cast<int>(endpos);
}

/****************************************************************************************\
*                            Sample on image classification                             *
\****************************************************************************************/
//
// This part of the code was a little refactor
//
struct DDMParams
{
    DDMParams() : detectorType("SURF"), descriptorType("SURF"), matcherType("BruteForce") {}
    DDMParams( const string _detectorType, const string _descriptorType, const string& _matcherType ) :
        detectorType(_detectorType), descriptorType(_descriptorType), matcherType(_matcherType){}
    void read( const FileNode& fn )
    {
        fn["detectorType"] >> detectorType;
        fn["descriptorType"] >> descriptorType;
        fn["matcherType"] >> matcherType;
    }
    void write( FileStorage& fs ) const
    {
        fs << "detectorType" << detectorType;
        fs << "descriptorType" << descriptorType;
        fs << "matcherType" << matcherType;
    }
    void print() const
    {
        cout << "detectorType: " << detectorType << endl;
        cout << "descriptorType: " << descriptorType << endl;
        cout << "matcherType: " << matcherType << endl;
    }

    string detectorType;
    string descriptorType;
    string matcherType;
};

struct VocabTrainParams
{
    VocabTrainParams() : trainObjClass("chair"), vocabSize(1000), memoryUse(200), descProportion(0.3f) {}
    VocabTrainParams( const string _trainObjClass, size_t _vocabSize, size_t _memoryUse, float _descProportion ) :
            trainObjClass(_trainObjClass), vocabSize(_vocabSize), memoryUse(_memoryUse), descProportion(_descProportion) {}
    void read( const FileNode& fn )
    {
        fn["trainObjClass"] >> trainObjClass;
        fn["vocabSize"] >> vocabSize;
        fn["memoryUse"] >> memoryUse;
        fn["descProportion"] >> descProportion;
    }
    void write( FileStorage& fs ) const
    {
        fs << "trainObjClass" << trainObjClass;
        fs << "vocabSize" << vocabSize;
        fs << "memoryUse" << memoryUse;
        fs << "descProportion" << descProportion;
    }
    void print() const
    {
        cout << "trainObjClass: " << trainObjClass << endl;
        cout << "vocabSize: " << vocabSize << endl;
        cout << "memoryUse: " << memoryUse << endl;
        cout << "descProportion: " << descProportion << endl;
    }


    string trainObjClass; // Object class used for training visual vocabulary.
                          // It shouldn't matter which object class is specified here - visual vocab will still be the same.
    int vocabSize; //number of visual words in vocabulary to train
    int memoryUse; // Memory to preallocate (in MB) when training vocab.
                      // Change this depending on the size of the dataset/available memory.
    float descProportion; // Specifies the number of descriptors to use from each image as a proportion of the total num descs.
};

struct SVMTrainParamsExt
{
    SVMTrainParamsExt() : descPercent(0.5f), targetRatio(0.4f), balanceClasses(true) {}
    SVMTrainParamsExt( float _descPercent, float _targetRatio, bool _balanceClasses ) :
            descPercent(_descPercent), targetRatio(_targetRatio), balanceClasses(_balanceClasses) {}
    void read( const FileNode& fn )
    {
        fn["descPercent"] >> descPercent;
        fn["targetRatio"] >> targetRatio;
        fn["balanceClasses"] >> balanceClasses;
    }
    void write( FileStorage& fs ) const
    {
        fs << "descPercent" << descPercent;
        fs << "targetRatio" << targetRatio;
        fs << "balanceClasses" << balanceClasses;
    }
    void print() const
    {
        cout << "descPercent: " << descPercent << endl;
        cout << "targetRatio: " << targetRatio << endl;
        cout << "balanceClasses: " << balanceClasses << endl;
    }

    float descPercent; // Percentage of extracted descriptors to use for training.
    float targetRatio; // Try to get this ratio of positive to negative samples (minimum).
    bool balanceClasses;    // Balance class weights by number of samples in each (if true cSvmTrainTargetRatio is ignored).
};

void readUsedParams( const FileNode& fn, string& vocName, DDMParams& ddmParams, VocabTrainParams& vocabTrainParams, SVMTrainParamsExt& svmTrainParamsExt )
{
    fn["vocName"] >> vocName;

    FileNode currFn = fn;

    currFn = fn["ddmParams"];
    ddmParams.read( currFn );

    currFn = fn["vocabTrainParams"];
    vocabTrainParams.read( currFn );

    currFn = fn["svmTrainParamsExt"];
    svmTrainParamsExt.read( currFn );
}

void writeUsedParams( FileStorage& fs, const string& vocName, const DDMParams& ddmParams, const VocabTrainParams& vocabTrainParams, const SVMTrainParamsExt& svmTrainParamsExt )
{
    fs << "vocName" << vocName;

    fs << "ddmParams" << "{";
    ddmParams.write(fs);
    fs << "}";

    fs << "vocabTrainParams" << "{";
    vocabTrainParams.write(fs);
    fs << "}";

    fs << "svmTrainParamsExt" << "{";
    svmTrainParamsExt.write(fs);
    fs << "}";
}

void printUsedParams( const string& vocPath, const string& resDir,
                      const DDMParams& ddmParams, const VocabTrainParams& vocabTrainParams,
                      const SVMTrainParamsExt& svmTrainParamsExt )
{
    cout << "CURRENT CONFIGURATION" << endl;
    cout << "----------------------------------------------------------------" << endl;
    cout << "vocPath: " << vocPath << endl;
    cout << "resDir: " << resDir << endl;
    cout << endl; ddmParams.print();
    cout << endl; vocabTrainParams.print();
    cout << endl; svmTrainParamsExt.print();
    cout << "----------------------------------------------------------------" << endl << endl;
}

bool readVocabulary( const string& filename, Mat& vocabulary )
{
    cout << "Reading vocabulary...";
    FileStorage fs( filename, FileStorage::READ );
    if( fs.isOpened() )
    {
        fs["vocabulary"] >> vocabulary;
        cout << "done" << endl;
        return true;
    }
    return false;
}

bool writeVocabulary( const string& filename, const Mat& vocabulary )
{
    cout << "Saving vocabulary..." << endl;
    FileStorage fs( filename, FileStorage::WRITE );
    if( fs.isOpened() )
    {
        fs << "vocabulary" << vocabulary;
        return true;
    }
    return false;
}

Mat trainVocabulary( const string& filename, VocData& vocData, const VocabTrainParams& trainParams,
                     const Ptr<FeatureDetector>& fdetector, const Ptr<DescriptorExtractor>& dextractor )
{
    Mat vocabulary;
    if( !readVocabulary( filename, vocabulary) )
    {
        CV_Assert( dextractor->descriptorType() == CV_32FC1 );
        const int descByteSize = dextractor->descriptorSize()*4;
        const int maxDescCount = (trainParams.memoryUse * 1048576) / descByteSize; // Total number of descs to use for training.

        cout << "Extracting VOC data..." << endl;
        vector<ObdImage> images;
        vector<char> objectPresent;
        vocData.getClassImages( trainParams.trainObjClass, CV_OBD_TRAIN, images, objectPresent );

        cout << "Computing descriptors..." << endl;
        RNG& rng = theRNG();
        TermCriteria terminate_criterion;
        terminate_criterion.epsilon = FLT_EPSILON;
        BOWKMeansTrainer bowTrainer( trainParams.vocabSize, terminate_criterion, 3, KMEANS_PP_CENTERS );

        while( images.size() > 0 )
        {
            if( bowTrainer.descripotorsCount() >= maxDescCount )
            {
                assert( bowTrainer.descripotorsCount() == maxDescCount );
#ifdef DEBUG_DESC_PROGRESS
                cout << "Breaking due to full memory ( descriptors count = " << bowTrainer.descripotorsCount()
                        << "; descriptor size in bytes = " << descByteSize << "; all used memory = "
                        << bowTrainer.descripotorsCount()*descByteSize << endl;
#endif
                break;
            }

            // Randomly pick an image from the dataset which hasn't yet been seen
            // and compute the descriptors from that image.
            int randImgIdx = rng( images.size() );
            Mat colorImage = imread( images[randImgIdx].path );
            vector<KeyPoint> imageKeypoints;
            fdetector->detect( colorImage, imageKeypoints );
            Mat imageDescriptors;
            dextractor->compute( colorImage, imageKeypoints, imageDescriptors );

            //check that there were descriptors calculated for the current image
            if( !imageDescriptors.empty() )
            {
                int descCount = imageDescriptors.rows;
                // Extract trainParams.descProportion descriptors from the image, breaking if the 'allDescriptors' matrix becomes full
                int descsToExtract = static_cast<int>(trainParams.descProportion * static_cast<float>(descCount));
                // Fill mask of used descriptors
                vector<char> usedMask( descCount, false );
                fill( usedMask.begin(), usedMask.begin() + descsToExtract, true );
                for( int i = 0; i < descCount; i++ )
                {
                    int i1 = rng(descCount), i2 = rng(descCount);
                    char tmp = usedMask[i1]; usedMask[i1] = usedMask[i2]; usedMask[i2] = tmp;
                }

                for( int i = 0; i < descCount; i++ )
                {
                    if( usedMask[i] && bowTrainer.descripotorsCount() < maxDescCount )
                        bowTrainer.add( imageDescriptors.row(i) );
                }
            }

#ifdef DEBUG_DESC_PROGRESS
            cout << images.size() << " images left, " << images[randImgIdx].id << " processed - "
                    <</* descs_extracted << "/" << image_descriptors.rows << " extracted - " << */
                    cvRound((static_cast<double>(bowTrainer.descripotorsCount())/static_cast<double>(maxDescCount))*100.0)
                    << " % memory used" << ( imageDescriptors.empty() ? " -> no descriptors extracted, skipping" : "") << endl;
#endif

            // Delete the current element from images so it is not added again
            images.erase( images.begin() + randImgIdx );
        }

        cout << "Maximum allowed descriptor count: " << maxDescCount << ", Actual descriptor count: " << bowTrainer.descripotorsCount() << endl;

        cout << "Training vocabulary..." << endl;
        vocabulary = bowTrainer.cluster();

        if( !writeVocabulary(filename, vocabulary) )
        {
            cout << "Error: file " << filename << " can not be opened to write" << endl;
            exit(-1);
        }
    }
    return vocabulary;
}

bool readBowImageDescriptor( const string& file, Mat& bowImageDescriptor )
{
    FileStorage fs( file, FileStorage::READ );
    if( fs.isOpened() )
    {
        fs["imageDescriptor"] >> bowImageDescriptor;
        return true;
    }
    return false;
}

bool writeBowImageDescriptor( const string& file, const Mat& bowImageDescriptor )
{
    FileStorage fs( file, FileStorage::WRITE );
    if( fs.isOpened() )
    {
        fs << "imageDescriptor" << bowImageDescriptor;
        return true;
    }
    return false;
}

// Load in the bag of words vectors for a set of images, from file if possible
void calculateImageDescriptors( const vector<ObdImage>& images, vector<Mat>& imageDescriptors,
                                Ptr<BOWImgDescriptorExtractor>& bowExtractor, const Ptr<FeatureDetector>& fdetector,
                                const string& resPath )
{
    CV_Assert( !bowExtractor->getVocabulary().empty() );
    imageDescriptors.resize( images.size() );

    for( size_t i = 0; i < images.size(); i++ )
    {
        string filename = resPath + bowImageDescriptorsDir + "/" + images[i].id + ".xml.gz";
        if( readBowImageDescriptor( filename, imageDescriptors[i] ) )
        {
#ifdef DEBUG_DESC_PROGRESS
            cout << "Loaded bag of word vector for image " << i+1 << " of " << images.size() << " (" << images[i].id << ")" << endl;
#endif
        }
        else
        {
            Mat colorImage = imread( images[i].path );
#ifdef DEBUG_DESC_PROGRESS
            cout << "Computing descriptors for image " << i+1 << " of " << images.size() << " (" << images[i].id << ")" << flush;
#endif
            vector<KeyPoint> keypoints;
            fdetector->detect( colorImage, keypoints );
#ifdef DEBUG_DESC_PROGRESS
                cout << " + generating BoW vector" << std::flush;
#endif
            bowExtractor->compute( colorImage, keypoints, imageDescriptors[i] );
#ifdef DEBUG_DESC_PROGRESS
            cout << " ...DONE " << static_cast<int>(static_cast<float>(i+1)/static_cast<float>(images.size())*100.0)
                 << " % complete" << endl;
#endif
            if( !imageDescriptors[i].empty() )
            {
                if( !writeBowImageDescriptor( filename, imageDescriptors[i] ) )
                {
                    cout << "Error: file " << filename << "can not be opened to write bow image descriptor" << endl;
                    exit(-1);
                }
            }
        }
    }
}

void removeEmptyBowImageDescriptors( vector<ObdImage>& images, vector<Mat>& bowImageDescriptors,
                                     vector<char>& objectPresent )
{
    CV_Assert( !images.empty() );
    for( int i = (int)images.size() - 1; i >= 0; i-- )
    {
        bool res = bowImageDescriptors[i].empty();
        if( res )
        {
            cout << "Removing image " << images[i].id << " due to no descriptors..." << endl;
            images.erase( images.begin() + i );
            bowImageDescriptors.erase( bowImageDescriptors.begin() + i );
            objectPresent.erase( objectPresent.begin() + i );
        }
    }
}

void removeBowImageDescriptorsByCount( vector<ObdImage>& images, vector<Mat> bowImageDescriptors, vector<char> objectPresent,
                                       const SVMTrainParamsExt& svmParamsExt, int descsToDelete )
{
    RNG& rng = theRNG();
    int pos_ex = std::count( objectPresent.begin(), objectPresent.end(), true );
    int neg_ex = std::count( objectPresent.begin(), objectPresent.end(), false );

    while( descsToDelete != 0 )
    {
        int randIdx = rng(images.size());

        // Prefer positive training examples according to svmParamsExt.targetRatio if required
        if( objectPresent[randIdx] )
        {
            if( (static_cast<float>(pos_ex)/static_cast<float>(neg_ex+pos_ex)  < svmParamsExt.targetRatio) &&
                (neg_ex > 0) && (svmParamsExt.balanceClasses == false) )
            { continue; }
            else
            { pos_ex--; }
        }
        else
        { neg_ex--; }

        images.erase( images.begin() + randIdx );
        bowImageDescriptors.erase( bowImageDescriptors.begin() + randIdx );
        objectPresent.erase( objectPresent.begin() + randIdx );

        descsToDelete--;
    }
    CV_Assert( bowImageDescriptors.size() == objectPresent.size() );
}

void setSVMParams( CvSVMParams& svmParams, CvMat& class_wts_cv, const Mat& responses, bool balanceClasses )
{
    int pos_ex = countNonZero(responses == 1);
    int neg_ex = countNonZero(responses == -1);
    cout << pos_ex << " positive training samples; " << neg_ex << " negative training samples" << endl;

    svmParams.svm_type = CvSVM::C_SVC;
    svmParams.kernel_type = CvSVM::RBF;
    if( balanceClasses )
    {
        Mat class_wts( 2, 1, CV_32FC1 );
        // The first training sample determines the '+1' class internally, even if it is negative,
        // so store whether this is the case so that the class weights can be reversed accordingly.
        bool reversed_classes = (responses.at<float>(0) < 0.f);
        if( reversed_classes == false )
        {
            class_wts.at<float>(0) = static_cast<float>(pos_ex)/static_cast<float>(pos_ex+neg_ex); // weighting for costs of positive class + 1 (i.e. cost of false positive - larger gives greater cost)
            class_wts.at<float>(1) = static_cast<float>(neg_ex)/static_cast<float>(pos_ex+neg_ex); // weighting for costs of negative class - 1 (i.e. cost of false negative)
        }
        else
        {
            class_wts.at<float>(0) = static_cast<float>(neg_ex)/static_cast<float>(pos_ex+neg_ex);
            class_wts.at<float>(1) = static_cast<float>(pos_ex)/static_cast<float>(pos_ex+neg_ex);
        }
        class_wts_cv = class_wts;
        svmParams.class_weights = &class_wts_cv;
    }
}

void setSVMTrainAutoParams( CvParamGrid& c_grid, CvParamGrid& gamma_grid,
                            CvParamGrid& p_grid, CvParamGrid& nu_grid,
                            CvParamGrid& coef_grid, CvParamGrid& degree_grid )
{
    c_grid = CvSVM::get_default_grid(CvSVM::C);

    gamma_grid = CvSVM::get_default_grid(CvSVM::GAMMA);

    p_grid = CvSVM::get_default_grid(CvSVM::P);
    p_grid.step = 0;

    nu_grid = CvSVM::get_default_grid(CvSVM::NU);
    nu_grid.step = 0;

    coef_grid = CvSVM::get_default_grid(CvSVM::COEF);
    coef_grid.step = 0;

    degree_grid = CvSVM::get_default_grid(CvSVM::DEGREE);
    degree_grid.step = 0;
}

void trainSVMClassifier( CvSVM& svm, const SVMTrainParamsExt& svmParamsExt, const string& objClassName, VocData& vocData,
                         Ptr<BOWImgDescriptorExtractor>& bowExtractor, const Ptr<FeatureDetector>& fdetector,
                         const string& resPath )
{
    /* first check if a previously trained svm for the current class has been saved to file */
    string svmFilename = resPath + svmsDir + "/" + objClassName + ".xml.gz";

    FileStorage fs( svmFilename, FileStorage::READ);
    if( fs.isOpened() )
    {
        cout << "*** LOADING SVM CLASSIFIER FOR CLASS " << objClassName << " ***" << endl;
        svm.load( svmFilename.c_str() );
    }
    else
    {
        cout << "*** TRAINING CLASSIFIER FOR CLASS " << objClassName << " ***" << endl;
        cout << "CALCULATING BOW VECTORS FOR TRAINING SET OF " << objClassName << "..." << endl;

        // Get classification ground truth for images in the training set
        vector<ObdImage> images;
        vector<Mat> bowImageDescriptors;
        vector<char> objectPresent;
        vocData.getClassImages( objClassName, CV_OBD_TRAIN, images, objectPresent );

        // Compute the bag of words vector for each image in the training set.
        calculateImageDescriptors( images, bowImageDescriptors, bowExtractor, fdetector, resPath );

        // Remove any images for which descriptors could not be calculated
        removeEmptyBowImageDescriptors( images, bowImageDescriptors, objectPresent );

        CV_Assert( svmParamsExt.descPercent > 0.f && svmParamsExt.descPercent <= 1.f );
        if( svmParamsExt.descPercent < 1.f )
        {
            int descsToDelete = static_cast<int>(static_cast<float>(images.size())*(1.0-svmParamsExt.descPercent));

            cout << "Using " << (images.size() - descsToDelete) << " of " << images.size() <<
                    " descriptors for training (" << svmParamsExt.descPercent*100.0 << " %)" << endl;
            removeBowImageDescriptorsByCount( images, bowImageDescriptors, objectPresent, svmParamsExt, descsToDelete );
        }

        // Prepare the input matrices for SVM training.
        Mat trainData( images.size(), bowExtractor->getVocabulary().rows, CV_32FC1 );
        Mat responses( images.size(), 1, CV_32SC1 );

        // Transfer bag of words vectors and responses across to the training data matrices
        for( size_t imageIdx = 0; imageIdx < images.size(); imageIdx++ )
        {
            // Transfer image descriptor (bag of words vector) to training data matrix
            Mat submat = trainData.row(imageIdx);
            if( bowImageDescriptors[imageIdx].cols != bowExtractor->descriptorSize() )
            {
                cout << "Error: computed bow image descriptor size " << bowImageDescriptors[imageIdx].cols
                     << " differs from vocabulary size" << bowExtractor->getVocabulary().cols << endl;
                exit(-1);
            }
            bowImageDescriptors[imageIdx].copyTo( submat );

            // Set response value
            responses.at<int>(imageIdx) = objectPresent[imageIdx] ? 1 : -1;
        }

        cout << "TRAINING SVM FOR CLASS ..." << objClassName << "..." << endl;
        CvSVMParams svmParams;
        CvMat class_wts_cv;
        setSVMParams( svmParams, class_wts_cv, responses, svmParamsExt.balanceClasses );
        CvParamGrid c_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid;
        setSVMTrainAutoParams( c_grid, gamma_grid,  p_grid, nu_grid, coef_grid, degree_grid );
        svm.train_auto( trainData, responses, Mat(), Mat(), svmParams, 10, c_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid );
        cout << "SVM TRAINING FOR CLASS " << objClassName << " COMPLETED" << endl;

        svm.save( svmFilename.c_str() );
        cout << "SAVED CLASSIFIER TO FILE" << endl;
    }
}

void computeConfidences( CvSVM& svm, const string& objClassName, VocData& vocData,
                         Ptr<BOWImgDescriptorExtractor>& bowExtractor, const Ptr<FeatureDetector>& fdetector,
                         const string& resPath )
{
    cout << "*** CALCULATING CONFIDENCES FOR CLASS " << objClassName << " ***" << endl;
    cout << "CALCULATING BOW VECTORS FOR TEST SET OF " << objClassName << "..." << endl;
    // Get classification ground truth for images in the test set
    vector<ObdImage> images;
    vector<Mat> bowImageDescriptors;
    vector<char> objectPresent;
    vocData.getClassImages( objClassName, CV_OBD_TEST, images, objectPresent );

    // Compute the bag of words vector for each image in the test set
    calculateImageDescriptors( images, bowImageDescriptors, bowExtractor, fdetector, resPath );
    // Remove any images for which descriptors could not be calculated
    removeEmptyBowImageDescriptors( images, bowImageDescriptors, objectPresent);

    // Use the bag of words vectors to calculate classifier output for each image in test set
    cout << "CALCULATING CONFIDENCE SCORES FOR CLASS " << objClassName << "..." << endl;
    vector<float> confidences( images.size() );
    float signMul = 1.f;
    for( size_t imageIdx = 0; imageIdx < images.size(); imageIdx++ )
    {
        if( imageIdx == 0 )
        {
            // In the first iteration, determine the sign of the positive class
            float classVal = confidences[imageIdx] = svm.predict( bowImageDescriptors[imageIdx], false );
            float scoreVal = confidences[imageIdx] = svm.predict( bowImageDescriptors[imageIdx], true );
            signMul = (classVal < 0) == (scoreVal < 0) ? 1.f : -1.f;
        }
        // svm output of decision function
        confidences[imageIdx] = signMul * svm.predict( bowImageDescriptors[imageIdx], true );
    }

    cout << "WRITING QUERY RESULTS TO VOC RESULTS FILE FOR CLASS " << objClassName << "..." << endl;
    vocData.writeClassifierResultsFile( resPath + plotsDir, objClassName, CV_OBD_TEST, images, confidences, 1, true );

    cout << "DONE - " << objClassName << endl;
    cout << "---------------------------------------------------------------" << endl;
}

void computeGnuPlotOutput( const string& resPath, const string& objClassName, VocData& vocData )
{
    vector<float> precision, recall;
    float ap;

    const string resultFile = vocData.getResultsFilename( objClassName, CV_VOC_TASK_CLASSIFICATION, CV_OBD_TEST);
    const string plotFile = resultFile.substr(0, resultFile.size()-4) + ".plt";

    cout << "Calculating precision recall curve for class '" <<objClassName << "'" << endl;
    vocData.calcClassifierPrecRecall( resPath + plotsDir + "/" + resultFile, precision, recall, ap, true );
    cout << "Outputting to GNUPlot file..." << endl;
    vocData.savePrecRecallToGnuplot( resPath + plotsDir + "/" + plotFile, precision, recall, ap, objClassName, CV_VOC_PLOT_PNG );
}




int main(int argc, char** argv)
{
    if( argc != 3 && argc != 6 )
    {
    	help(argv);
        return -1;
    }

    const string vocPath = argv[1], resPath = argv[2];

    // Read or set default parameters
    string vocName;
    DDMParams ddmParams;
    VocabTrainParams vocabTrainParams;
    SVMTrainParamsExt svmTrainParamsExt;

    makeUsedDirs( resPath );

    FileStorage paramsFS( resPath + "/" + paramsFile, FileStorage::READ );
    if( paramsFS.isOpened() )
    {
       readUsedParams( paramsFS.root(), vocName, ddmParams, vocabTrainParams, svmTrainParamsExt );
       CV_Assert( vocName == getVocName(vocPath) );
    }
    else
    {
        vocName = getVocName(vocPath);
        if( argc!= 6 )
        {
            cout << "Feature detector, descriptor extractor, descriptor matcher must be set" << endl;
            return -1;
        }
        ddmParams = DDMParams( argv[3], argv[4], argv[5] ); // from command line
        // vocabTrainParams and svmTrainParamsExt is set by defaults
        paramsFS.open( resPath + "/" + paramsFile, FileStorage::WRITE );
        if( paramsFS.isOpened() )
        {
            writeUsedParams( paramsFS, vocName, ddmParams, vocabTrainParams, svmTrainParamsExt );
            paramsFS.release();
        }
        else
        {
            cout << "File " << (resPath + "/" + paramsFile) << "can not be opened to write" << endl;
            return -1;
        }
    }

    // Create detector, descriptor, matcher.
    Ptr<FeatureDetector> featureDetector = FeatureDetector::create( ddmParams.detectorType );
    Ptr<DescriptorExtractor> descExtractor = DescriptorExtractor::create( ddmParams.descriptorType );
    Ptr<BOWImgDescriptorExtractor> bowExtractor;
    if( featureDetector.empty() || descExtractor.empty() )
    {
        cout << "featureDetector or descExtractor was not created" << endl;
        return -1;
    }
    {
        Ptr<DescriptorMatcher> descMatcher = DescriptorMatcher::create( ddmParams.matcherType );
        if( featureDetector.empty() || descExtractor.empty() || descMatcher.empty() )
        {
            cout << "descMatcher was not created" << endl;
            return -1;
        }
        bowExtractor = new BOWImgDescriptorExtractor( descExtractor, descMatcher );
    }

    // Print configuration to screen
    printUsedParams( vocPath, resPath, ddmParams, vocabTrainParams, svmTrainParamsExt );
    // Create object to work with VOC
    VocData vocData( vocPath, false );

    // 1. Train visual word vocabulary if a pre-calculated vocabulary file doesn't already exist from previous run
    Mat vocabulary = trainVocabulary( resPath + "/" + vocabularyFile, vocData, vocabTrainParams,
                                      featureDetector, descExtractor );
    bowExtractor->setVocabulary( vocabulary );

    // 2. Train a classifier and run a sample query for each object class
    const vector<string>& objClasses = vocData.getObjectClasses(); // object class list
    for( size_t classIdx = 0; classIdx < objClasses.size(); ++classIdx )
    {
        // Train a classifier on train dataset
        CvSVM svm;
        trainSVMClassifier( svm, svmTrainParamsExt, objClasses[classIdx], vocData,
                            bowExtractor, featureDetector, resPath );

        // Now use the classifier over all images on the test dataset and rank according to score order
        // also calculating precision-recall etc.
        computeConfidences( svm, objClasses[classIdx], vocData,
                            bowExtractor, featureDetector, resPath );
        // Calculate precision/recall/ap and use GNUPlot to output to a pdf file
        computeGnuPlotOutput( resPath, objClasses[classIdx], vocData );
    }
    return 0;
}
