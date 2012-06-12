Face Recognition with OpenCV
=============================

.. highlight:: cpp

This document is going to explain how to perform face recognition with the new
:ocv:class:`FaceRecognizer`, which is available from OpenCV 2.4 onwards. 

FaceRecognizer
--------------

All face recognition models in OpenCV are derived from the abstract base 
class :ocv:class:`FaceRecognizer`, which provides a unified access to all face 
recongition algorithms in OpenCV.

.. ocv:class:: FaceRecognizer

  class FaceRecognizer : public Algorithm
  {
  public:
      //! virtual destructor
      virtual ~FaceRecognizer() {}

      // Trains a FaceRecognizer.
      virtual void train(InputArray src, InputArray labels) = 0;

      // Gets a prediction from a FaceRecognizer.
      virtual int predict(InputArray src) const = 0;

      // Predicts the label and confidence for a given sample.
      virtual void predict(InputArray src, int &label, double &dist) const = 0;

      // Serializes this object to a given filename.
      virtual void save(const string& filename) const;

      // Deserializes this object from a given filename.
      virtual void load(const string& filename);

      // Serializes this object to a given cv::FileStorage.
      virtual void save(FileStorage& fs) const = 0;

      // Deserializes this object from a given cv::FileStorage.
      virtual void load(const FileStorage& fs) = 0;

  };
  
  Ptr<FaceRecognizer> createEigenFaceRecognizer(int num_components = 0, double threshold = DBL_MAX);
  Ptr<FaceRecognizer> createFisherFaceRecognizer(int num_components = 0, double threshold = DBL_MAX);
  Ptr<FaceRecognizer> createLBPHFaceRecognizer(int radius=1, int neighbors=8, int grid_x=8, int grid_y=8, double threshold = DBL_MAX);

FaceRecognizer::train
---------------------

Trains a FaceRecognizer with given data and associated labels.

.. ocv:function:: void FaceRecognizer::train(InputArray src, InputArray labels)

Every model subclassing :ocv:class:`FaceRecognizer` is able to work with 
image data in ``src``, which must be given as a ``vector<Mat>``. A ``vector<Mat>`` 
was chosen, so that no preliminary assumptions about the input samples is made. 
The Local Binary Patterns for example process 2D images of different size, while 
Eigenfaces and Fisherfaces method reshape all images in ``src`` to a data matrix.

The associated labels in ``labels`` have to be given either in a ``Mat``
with one row/column of type ``CV_32SC1`` or simply as a ``vector<int>``.

The following example shows how to learn a Eigenfaces model with OpenCV:

.. code-block:: cpp

  // holds images and labels
  vector<Mat> images;
  vector<int> labels;
  // images for first person
  images.push_back(imread("person0/0.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
  images.push_back(imread("person0/1.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
  images.push_back(imread("person0/2.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
  // images for second person
  images.push_back(imread("person1/0.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(1);
  images.push_back(imread("person1/1.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(1);
  images.push_back(imread("person1/2.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(1);
  // create a new Fisherfaces model
  Ptr<FaceRecognizer> model =  createEigenFaceRecognizer();
  // and learn it
  model->train(images,labels);

FaceRecognizer::predict
-----------------------

Predicts the label for a given query image in ``src``. 

.. ocv:function:: int FaceRecognizer::predict(InputArray src) const

Predicts the label for a given query image in ``src``. 

.. ocv:function:: void FaceRecognizer::predict(InputArray src, int &label, double &dist) const


The suffix ``const`` means that prediction does not affect the internal model 
state, so the method can be safely called from within different threads.

The following example shows how to get a prediction from a trained model:

.. code-block:: cpp

  int predictedLabel = model->predict(testSample);
  
To get the confidence of a prediction call the model with:

.. code-block:: cpp

  int predictedLabel = -1;
  double confidence = 0.0;
  model->predict(testSample, predictedLabel, confidence);

FaceRecognizer::save
--------------------

Saves a :ocv:class:`FaceRecognizer` and its model state.

.. ocv:function:: void FaceRecognizer::save(const string& filename) const
.. ocv:function:: void FaceRecognizer::save(FileStorage& fs) const

Every :ocv:class:`FaceRecognizer` overwrites ``FaceRecognizer::save(FileStorage& fs)``
to save its internal model state. You can then either call ``FaceRecognizer::save(FileStorage& fs)`` 
to save the model or use ``FaceRecognizer::save(const string& filename)``, which eases saving a 
model.

The suffix ``const`` means that prediction does not affect the internal model 
state, so the method can be safely called from within different threads.

FaceRecognizer::load
--------------------

Loads a :ocv:class:`FaceRecognizer` and its model state.

.. ocv:function:: void FaceRecognizer::load(const string& filename)
.. ocv:function:: void FaceRecognizer::load(FileStorage& fs)

Loads a persisted model and state from a given XML or YAML file . Every 
:ocv:class:`FaceRecognizer` has overwrites ``FaceRecognizer::load(FileStorage& fs)`` 
to enable loading the internal model state. ``FaceRecognizer::load(const string& filename)``
 eases saving a model, so you just need to call it on the filename.

createEigenFaceRecognizer
-------------------------

Creates an Eigenfaces model with given number of components (if given) and threshold (if given).

.. ocv:function:: Ptr<FaceRecognizer> createEigenFaceRecognizer(int num_components = 0, double threshold = DBL_MAX)

This model implements the Eigenfaces method as described in [TP91]_.

* ``num_components`` (default 0) number of components are kept for classification. If no number of 
components is given, it is automatically determined from given data in 
:ocv:func:`FaceRecognizer::train`. If (and only if) ``num_components`` <= 0, then 
``num_components`` is set to (N-1) in ocv:func:`Eigenfaces::train`, with *N* being the 
total number of samples in ``src``.

* ``threshold`` (default DBL_MAX) 

Internal model data, which can be accessed through cv::Algorithm:

 * ``ncomponents`` 
 
 * ``threshold``
 
 * ``eigenvectors``

 * ``eigenvalues``
 
 * ``mean``
 
 * ``labels``
 
 * ``projections``

createFisherFaceRecognizer
--------------------------

Creates a Fisherfaces model for given a given number of components and threshold.

.. ocv:function:: Ptr<FaceRecognizer> createFisherFaceRecognizer(int num_components = 0, double threshold = DBL_MAX)

This model implements the Fisherfaces method as described in [BHK97]_.

* ``num_components`` number of components are kept for classification. If no number 
of components is given (default 0), it is automatically determined from given data 
in :ocv:func:`Fisherfaces::train` (model implementation). If (and only if) 
``num_components`` <= 0, then ``num_components`` is set to (C-1) in 
ocv:func:`train`, with *C* being the number of unique classes in ``labels``.

* ``threshold`` (default DBL_MAX) 

Internal model data, which can be accessed through cv::Algorithm:

 * ``ncomponents``
 
 * ``threshold``
 
 * ``projections``
 
 * ``labels``
 
 * ``eigenvectors``
 
 * ``eigenvalues``

 * ``mean``

createLBPHFaceRecognizer
------------------------

Implements face recognition with Local Binary Patterns Histograms as described in [Ahonen04]_.

.. ocv:function:: Ptr<FaceRecognizer> createLBPHFaceRecognizer(int radius=1, int neighbors=8, int grid_x=8, int grid_y=8, double threshold = DBL_MAX);

Internal model data, which can be accessed through cv::Algorithm:


 * ``radius``
 
 * ``neighbors``
 
 * ``grid_x``
 
 * ``grid_y``
 
 * ``threshold``
 
 * ``histograms``
 
 * ``labels``

Example: Working with a cv::FaceRecognizer
===========================================

In this tutorial you'll see how to do face recognition with OpenCV on real image data. We'll work through a complete example, so you know how to work with it. While this example is based on Eigenfaces, it works the same for all the other available :ocv:class:`FaceRecognizer` implementations. 

Getting Image Data
------------------

We are doing face recognition, so you'll need some face images first! You can decide to either create your own database or start with one of the many available datasets. `face-rec.org/databases <http://face-rec.org/databases/>`_ gives an up-to-date overview of public available datasets (parts of the following descriptions are quoted from there). 

Three interesting databases are:

* `AT&T Facedatabase <http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html>`_ The AT&T Facedatabase, sometimes also referred to as *ORL Database of Faces*, contains ten different images of each of 40 distinct subjects. For some subjects, the images were taken at different times, varying the lighting, facial expressions (open / closed eyes, smiling / not smiling) and facial details (glasses / no glasses). All the images were taken against a dark homogeneous background with the subjects in an upright, frontal position (with tolerance for some side movement).
  
* `Yale Facedatabase A <http://cvc.yale.edu/projects/yalefaces/yalefaces.html>`_ The AT&T Facedatabase is good for initial tests, but it's a fairly easy database. The Eigenfaces method already has a 97% recognition rate, so you won't see any improvements with other algorithms. The Yale Facedatabase A is a more appropriate dataset for initial experiments, because the recognition problem is harder. The database consists of 15 people (14 male, 1 female) each with 11 grayscale images sized :math:`320 \times 243` pixel. There are changes in the light conditions (center light, left light, right light), facial expressions (happy, normal, sad, sleepy, surprised, wink) and glasses (glasses, no-glasses). 
  
*  `Extended Yale Facedatabase B <http://vision.ucsd.edu/~leekc/ExtYaleDatabase/ExtYaleB.html>`_ The Extended Yale Facedatabase B contains 2414 images of 38 different people in its cropped version. The focus of this database is set on extracting features that are robust to illumination, the images have almost no variation in emotion/occlusion/... . I personally think, that this dataset is too large for the experiments I perform in this document. You better use the `AT&T Facedatabase <http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html>`_ for intial testing. A first version of the Yale Facedatabase B was used in [Belhumeur97]_ to see how the Eigenfaces and Fisherfaces method perform under heavy illumination changes. [Lee2005]_ used the same setup to take 16128 images of 28 people. The Extended Yale Facedatabase B is the merge of the two databases, which is now known as Extended Yalefacedatabase B.

For this tutorial I am going to use the `AT&T Facedatabase <http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html>`_, which is available from: `http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html <http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html>`_. All credit for this dataset is given to the *AT&T Laboratories, Cambridge*, also make sure to read the  README

Reading the Image Data
-----------------------

In the demo I have decided to read the images from a very simple CSV file. Why? Because it's the simplest platform-independent approach I can think of. However, if you know a simpler solution please ping me about it. Basically all the CSV file needs to contain are lines composed of a **filename** followed by a **;** followed by the **label** (as integer number), making up a line like this: 

.. code-block:: none

  /path/to/at/s1/1.pgm;0

Think of the **label** as the subject (the person) this image belongs to, so same subjects (persons) should have the same label. Let's make up an example. Download the AT&T Facedatabase from `http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html <http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html>`_ and extract it to a folder of your choice. I am referring to the path you have chosen as **/path/to** in the following listings. You'll now have a folder structure like this:

.. code-block:: none

  philipp@mango:~/path/to/at$ tree
  .
  |-- README
  |-- s1
  |   |-- 1.pgm
  |   |-- ...
  |   |-- 10.pgm
  |-- s2
  |   |-- 1.pgm
  |   |-- ...
  |   |-- 10.pgm
  ...
  |-- s40
  |   |-- 1.pgm
  |   |-- ...
  |   |-- 10.pgm

So that's actually very simple to map to the CSV file. You don't have to take care about the order of the label or anything, just make sure the same persons have the same label:

.. code-block:: none

  /path/to/at/s1/1.pgm;0
  /path/to/at/s1/2.pgm;0
  ...
  /path/to/at/s2/1.pgm;1
  /path/to/at/s2/2.pgm;1
  ...
  /path/to/at/s40/1.pgm;39
  /path/to/at/s40/2.pgm;39

You don't need to create this file yourself for the AT&T Face Database, because there's already a template file in ``opencv/samples/cpp/facerec_at_t.txt``. You just need to replace the **/path/to** with the folder, where you extracted the archive to. An example: imagine I have extracted the files to D:/data/at. Then I would simply Search & Replace **/path/to** with **D:/data**. You can do that in an editor of your choice, every sufficiently advanced editor can do this! Once you have a CSV file with *valid filenames* and labels, you can run the demo by with the path to the CSV file as parameter.

The demo application (opencv/samples/cpp/facerec_demo.cpp)
----------------------------------------------------------

The following is the demo application which can be found in ``opencv/samples/cpp/facerec_demo.cpp``. If you have chosen to build OpenCV with the samples, chances are good you have the executable already! However you don't need to copy and paste this code, because it's the same as in ``opencv/samples/cpp/facerec_demo.cpp``. I am going to simply paste the source code listing here, as there is an extensive description in the comments within the file.

.. code-block:: cpp

  #include "opencv2/core/core.hpp"
  #include "opencv2/highgui/highgui.hpp"
  #include "opencv2/contrib/contrib.hpp"

  #include <iostream>
  #include <fstream>
  #include <sstream>

  using namespace cv;
  using namespace std;

  static Mat toGrayscale(InputArray _src) {
      Mat src = _src.getMat();
      // only allow one channel
      if(src.channels() != 1) {
          CV_Error(CV_StsBadArg, "Only Matrices with one channel are supported");
      }
      // create and return normalized image
      Mat dst;
      cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
      return dst;
  }

  static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
      std::ifstream file(filename.c_str(), ifstream::in);
      if (!file) {
          string error_message = "No valid input file was given, please check the given filename.";
          CV_Error(CV_StsBadArg, error_message);
      }
      string line, path, classlabel;
      while (getline(file, line)) {
          stringstream liness(line);
          getline(liness, path, separator);
          getline(liness, classlabel);
          if(!path.empty() && !classlabel.empty()) {
              images.push_back(imread(path, 0));
              labels.push_back(atoi(classlabel.c_str()));
          }
      }
  }

  int main(int argc, const char *argv[]) {
      // Check for valid command line arguments, print usage
      // if no arguments were given.
      if (argc != 2) {
          cout << "usage: " << argv[0] << " <csv.ext>" << endl;
          exit(1);
      }
      // Get the path to your CSV.
      string fn_csv = string(argv[1]);
      // These vectors hold the images and corresponding labels.
      vector<Mat> images;
      vector<int> labels;
      // Read in the data. This can fail if no valid
      // input filename is given.
      try {
          read_csv(fn_csv, images, labels);
      } catch (cv::Exception& e) {
          cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
          // nothing more we can do
          exit(1);
      }
      // Quit if there are not enough images for this demo.
      if(images.size() <= 1) {
          string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
          CV_Error(CV_StsError, error_message);
      }
      // Get the height from the first image. We'll need this
      // later in code to reshape the images to their original
      // size:
      int height = images[0].rows;
      // The following lines simply get the last images from
      // your dataset and remove it from the vector. This is
      // done, so that the training data (which we learn the
      // cv::FaceRecognizer on) and the test data we test
      // the model with, do not overlap.
      Mat testSample = images[images.size() - 1];
      int testLabel = labels[labels.size() - 1];
      images.pop_back();
      labels.pop_back();
      // The following lines create an Eigenfaces model for
      // face recognition and train it with the images and
      // labels read from the given CSV file.
      // This here is a full PCA, if you just want to keep
      // 10 principal components (read Eigenfaces), then call
      // the factory method like this:
      //
      //      cv::createEigenFaceRecognizer(10);
      //
      // If you want to create a FaceRecognizer with a
      // confidennce threshold, call it with:
      //
      //      cv::createEigenFaceRecognizer(10, 123.0);
      //
      Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
      model->train(images, labels);
      // The following line predicts the label of a given
      // test image:
      int predictedLabel = model->predict(testSample);
      //
      // To get the confidence of a prediction call the model with:
      //
      //      int predictedLabel = -1;
      //      double confidence = 0.0;
      //      model->predict(testSample, predictedLabel, confidence);
      //
      string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
      cout << result_message << endl;
      // Sometimes you'll need to get/set internal model data,
      // which isn't exposed by the public cv::FaceRecognizer.
      // Since each cv::FaceRecognizer is derived from a
      // cv::Algorithm, you can query the data.
      //
      // First we'll use it to set the threshold of the FaceRecognizer
      // to 0.0 without retraining the model. This can be useful if
      // you are evaluating the model:
      //
      model->set("threshold", 0.0);
      // Now the threshold of this model is set to 0.0. A prediction
      // now returns -1, as it's impossible to have a distance below
      // it
      predictedLabel = model->predict(testSample);
      cout << "Predicted class = " << predictedLabel << endl;
      // Here is how to get the eigenvalues of this Eigenfaces model:
      Mat eigenvalues = model->getMat("eigenvalues");
      // And we can do the same to display the Eigenvectors (read Eigenfaces):
      Mat W = model->getMat("eigenvectors");
      // From this we will display the (at most) first 10 Eigenfaces:
      for (int i = 0; i < min(10, W.cols); i++) {
          string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
          cout << msg << endl;
          // get eigenvector #i
          Mat ev = W.col(i).clone();
          // Reshape to original size & normalize to [0...255] for imshow.
          Mat grayscale = toGrayscale(ev.reshape(1, height));
          // Show the image & apply a Jet colormap for better sensing.
          Mat cgrayscale;
          applyColorMap(grayscale, cgrayscale, COLORMAP_JET);
          imshow(format("%d", i), cgrayscale);
      }
      waitKey(0);

      return 0;
  }

Running the demo application
----------------------------

.. code-block:: none

  TODO

Results
-------

.. code-block:: none

  TODO

Saving and Loading a cv::FaceRecognizer
=======================================

Saving and loading a :ocv:class:`FaceRecognizer` is a very important task, because 
training a :ocv:class:`FaceRecognizer` can be a very time-intense task for large 
datasets (depending on your algorithm). In OpenCV you only have to call 
:ocv:func:`FaceRecognizer::load` for loading, and :ocv:func:`FaceRecognizer::save` 
for saving the internal state of a :ocv:class:`FaceRecognizer`.

Imagine we are using the same example as above. We want to learn the Eigenfaces of 
the `AT&T Facedatabase <http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html>`_, 
but store the model to a YAML file so we can load it from somewhere else. 

To see if everything went fine, we'll have a look at the stored data and the first 10 Eigenfaces.

Demo application
----------------

.. code-block:: none

  TODO
  
Results
-------

``eigenfaces_at.yml`` contains the model state, we'll simply show the first 10 
lines with ``head eigenfaces_at.yml``: 

.. code-block:: none

  philipp@mango:~/github/libfacerec-build$ head eigenfaces_at.yml
  %YAML:1.0
  num_components: 399
  mean: !!opencv-matrix
     rows: 1
     cols: 10304
     dt: d
     data: [ 8.5558897243107765e+01, 8.5511278195488714e+01,
         8.5854636591478695e+01, 8.5796992481203006e+01,
         8.5952380952380949e+01, 8.6162907268170414e+01,
         8.6082706766917283e+01, 8.5776942355889716e+01,

And here are the Eigenfaces:


Credits
=======

The Database of Faces
---------------------

*Important: when using these images, please give credit to "AT&T Laboratories, Cambridge."*

The Database of Faces, formerly "The ORL Database of Faces", contains a set of face images taken between April 1992 and April 1994. The database was used in the context of a face recognition project carried out in collaboration with the Speech, Vision and Robotics Group of the Cambridge University Engineering Department.

There are ten different images of each of 40 distinct subjects. For some subjects, the images were taken at different times, varying the lighting, facial expressions (open / closed eyes, smiling / not smiling) and facial details (glasses / no glasses). All the images were taken against a dark homogeneous background with the subjects in an upright, frontal position (with tolerance for some side movement).

The files are in PGM format. The size of each image is 92x112 pixels, with 256 grey levels per pixel. The images are organised in 40 directories (one for each subject), which have names of the form sX, where X indicates the subject number (between 1 and 40). In each of these directories, there are ten different images of that subject, which have names of the form Y.pgm, where Y is the image number for that subject (between 1 and 10).

A copy of the database can be retrieved from:

`http://www.cl.cam.ac.uk/research/dtg/attarchive/pub/data/att_faces.zip <http://www.cl.cam.ac.uk/research/dtg/attarchive/pub/data/att_faces.zip>_`

Literature
==========

.. [Ahonen04] Ahonen, T., Hadid, A., and Pietikainen, M. *Face Recognition with Local Binary Patterns.* Computer Vision - ECCV 2004 (2004), 469–481.

.. [Fisher36] Fisher, R. A. *The use of multiple measurements in taxonomic problems.* Annals Eugen. 7 (1936), 179–188.

.. [BHK97] Belhumeur, P. N., Hespanha, J., and Kriegman, D. *Eigenfaces vs. Fisherfaces: Recognition Using Class Specific Linear Projection.* IEEE Transactions on Pattern Analysis and Machine Intelligence 19, 7 (1997), 711–720.

.. [TP91] Turk, M., and Pentland, A. *Eigenfaces for recognition.* Journal of Cognitive Neuroscience 3 (1991), 71–86.

.. [Tan10] Tan, X., and Triggs, B. *Enhanced local texture feature sets for face recognition under difficult lighting conditions.* IEEE Transactions on Image Processing 19 (2010), 1635–650.

.. [Zhao03] Zhao, W., Chellappa, R., Phillips, P., and Rosenfeld, A. Face recognition: A literature survey. ACM Computing Surveys (CSUR) 35, 4 (2003), 399–458.

.. [Tu06] Chiara Turati, Viola Macchi Cassia, F. S., and Leo, I. *Newborns face recognition: Role of inner and outer facial features. Child Development* 77, 2 (2006), 297–311.

.. [Kanade73] Kanade, T. *Picture processing system by computer complex and recognition of human faces.* PhD thesis, Kyoto University, November 1973


