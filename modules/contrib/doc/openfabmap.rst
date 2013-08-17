OpenFABMAP
========================================

.. highlight:: cpp

The openFABMAP package has been integrated into OpenCV from the openFABMAP <http://code.google.com/p/openfabmap/> project [ICRA2011]_. OpenFABMAP is an open and modifiable code-source which implements the Fast Appearance-based Mapping algorithm (FAB-MAP) developed by Mark Cummins and Paul Newman. The algorithms used in openFABMAP were developed using only the relevant FAB-MAP publications.

FAB-MAP is an approach to appearance-based place recognition. FAB-MAP compares images of locations that have been visited and determines the probability of re-visiting a location, as well as providing a measure of the probability of being at a new, previously unvisited location. Camera images form the sole input to the system, from which visual bag-of-words models are formed through the extraction of appearance-based (e.g. SURF) features.

openFABMAP requires training data (e.g. a collection of images from a similar but not identical environment) to construct a visual vocabulary for the visual bag-of-words model, along with a Chow-Liu tree representation of feature likelihood and for use in the Sampled new place method (see below).

.. note::

   * An example using the openFABMAP package can be found at opencv_source_code/samples/cpp/fabmap_sample.cpp

of2::FabMap
--------------------

.. ocv:class:: of2::FabMap

The main FabMap class performs the comparison between visual bags-of-words extracted from one or more images. The FabMap class is instantiated as one of the four inherited FabMap classes (FabMap1, FabMapLUT, FabMapFBO, FabMap2). Each inherited class performs the comparison differently based on algorithm iterations as published (see each class below for specifics). A Chow-Liu tree, detector model parameters and some option flags are common to all Fabmap variants and are supplied on class creation. Training data (visual bag-of-words) is supplied to the class if using the SAMPLED new place method. Test data (visual bag-of-words) is supplied as images to which query bag-of-words are compared against. The common flags are listed below: ::

    enum {
        MEAN_FIELD,
        SAMPLED,
        NAIVE_BAYES,
        CHOW_LIU,
        MOTION_MODEL
    };

#. MEAN_FIELD: Use the Mean Field approximation to determine the new place likelihood (cannot be used for FabMap2).
#. SAMPLED: Use the Sampled approximation to determine the new place likelihood. Requires training data (see below).
#. NAIVE_BAYES: Assume a naive Bayes approximation to feature distribution (i.e. all features are independent). Note that a Chow-Liu tree is still required but only the absolute word probabilities are used, feature co-occurrance information is discarded.
#. CHOW_LIU: Use the full Chow-Liu tree to approximate feature distribution.
#. MOTION_MODEL: Update the location distribution using the previous distribution as a (weak) prior. Used for matching in sequences (i.e. successive video frames).

Training Data
++++++++++++++++++++

Training data is required to use the SAMPLED new place method. The SAMPLED method was shown to have improved performance over the alternative MEAN_FIELD method. Training data can be added singularly or as a batch.

.. ocv:function:: virtual void addTraining(const Mat& queryImgDescriptor)

    :param queryImgDescriptor: bag-of-words image descriptors stored as rows in a Mat

.. ocv:function:: virtual void addTraining(const vector<Mat>& queryImgDescriptors)

    :param queryImgDescriptors: a vector containing multiple bag-of-words image descriptors

.. ocv:function:: const vector<Mat>& getTrainingImgDescriptors() const

    Returns a vector containing multiple bag-of-words image descriptors

Test Data
++++++++++++++++++++

Test Data is the database of images represented using bag-of-words models. When a compare function is called, each query point is compared to the test data.

.. ocv:function:: virtual void add(const Mat& queryImgDescriptor)

    :param queryImgDescriptor: bag-of-words image descriptors stored as rows in a Mat

.. ocv:function:: virtual void add(const vector<Mat>& queryImgDescriptors)

    :param queryImgDescriptors: a vector containing multiple bag-of-words image descriptors

.. ocv:function:: const vector<Mat>& getTestImgDescriptors() const

    Returns a vector containing multiple bag-of-words image descriptors

Image Comparison
++++++++++++++++++++

Image matching is performed calling the compare function. Query bag-of-words image descriptors are provided and compared to test data added to the FabMap class. Alternatively test data can be provided with the call to compare to which the comparison is performed. Results are written to the 'matches' argument.

.. ocv:function:: void compare(const Mat& queryImgDescriptor, vector<IMatch>& matches, bool addQuery = false, const Mat& mask = Mat())

    :param queryImgDescriptor: bag-of-words image descriptors stored as rows in a Mat

    :param matches: a vector of image match probabilities

    :param addQuery: if true the queryImg Descriptor is added to the test data after the comparison is performed.

    :param mask: *not implemented*

.. ocv:function:: void compare(const Mat& queryImgDescriptor, const Mat& testImgDescriptors, vector<IMatch>& matches, const Mat& mask = Mat())

    :param testImgDescriptors: bag-of-words image descriptors stored as rows in a Mat

.. ocv:function:: void compare(const Mat& queryImgDescriptor, const vector<Mat>& testImgDescriptors, vector<IMatch>& matches, const Mat& mask = Mat())

    :param testImgDescriptors:  a vector of multiple bag-of-words image descriptors

.. ocv:function:: void compare(const vector<Mat>& queryImgDescriptors, vector<IMatch>& matches, bool addQuery = false, const Mat& mask = Mat())

    :param queryImgDescriptors: a vector of multiple bag-of-words image descriptors

.. ocv:function:: void compare(const vector<Mat>& queryImgDescriptors, const vector<Mat>& testImgDescriptors, vector<IMatch>& matches, const Mat& mask = Mat())



FabMap classes
++++++++++++++++++++

.. ocv:class:: FabMap1 : public FabMap

The original FAB-MAP algorithm without any computational improvements as published in [IJRR2008]_

.. ocv:function:: FabMap1::FabMap1(const Mat& clTree, double PzGe, double PzGNe, int flags, int numSamples = 0)

    :param clTree: a Chow-Liu tree class

    :param PzGe: the dector model recall. The probability of the feature detector extracting a feature from an object given it is in the scene. This is used to account for detector noise.

    :param PzGNe: the dector model precision. The probability of the feature detector falsing extracting a feature representing an object that is not in the scene.

    :param numSamples: the number of samples to use for the SAMPLED new place calculation

.. ocv:class:: FabMapLUT : public FabMap

The original FAB-MAP algorithm implemented as a look-up table for speed enhancements [ICRA2011]_

.. ocv:function:: FabMapLUT::FabMapLUT(const Mat& clTree, double PzGe, double PzGNe, int flags, int numSamples = 0, int precision = 6)

    :param precision: the precision with which to store the pre-computed likelihoods

.. ocv:class:: FabMapFBO : public FabMap

The accelerated FAB-MAP using a 'fast bail-out' approach as in [TRO2010]_

.. ocv:function:: FabMapFBO::FabMapFBO(const Mat& clTree, double PzGe, double PzGNe, int flags, int numSamples = 0, double rejectionThreshold = 1e-8, double PsGd = 1e-8, int bisectionStart = 512, int bisectionIts = 9)

    :param rejectionThreshold: images are not considered a match when the likelihood falls below the Bennett bound by the amount given by the rejectionThreshold. The threshold provides a speed/accuracy trade-off. A lower bound will be more accurate

    :param PsGd: used to calculate the Bennett bound. Provides a speed/accuracy trade-off. A lower bound will be more accurate

    :param bisectionStart: Used to estimate the bound using the bisection method. Must be larger than the largest expected difference between maximum and minimum image likelihoods

    :param bisectionIts: The number of iterations for which to perform the bisection method


.. ocv:class:: FabMap2 : public FabMap

The inverted index FAB-MAP as in [IJRR2010]_. This version of FAB-MAP is the fastest without any loss of accuracy.

.. ocv:function:: FabMap2::FabMap2(const Mat& clTree, double PzGe, double PzGNe, int flags)

.. [IJRR2008] M. Cummins and P. Newman, "FAB-MAP: Probabilistic Localization and Mapping in the Space of Appearance," The International Journal of Robotics Research, vol. 27(6), pp. 647-665, 2008

.. [TRO2010] M. Cummins and P. Newman, "Accelerating FAB-MAP with concentration inequalities," IEEE Transactions on Robotics, vol. 26(6), pp. 1042-1050, 2010

.. [IJRR2010] M. Cummins and P. Newman, "Appearance-only SLAM at large scale with FAB-MAP 2.0," The International Journal of Robotics Research, vol. 30(9), pp. 1100-1123, 2010

.. [ICRA2011] A. Glover, et al., "OpenFABMAP: An Open Source Toolbox for Appearance-based Loop Closure Detection," in IEEE International Conference on Robotics and Automation, St Paul, Minnesota, 2011

of2::IMatch
--------------------

.. ocv:struct:: of2::IMatch

FAB-MAP comparison results are stored in a vector of IMatch structs. Each IMatch structure provides the index of the provided query bag-of-words, the index of the test bag-of-words, the raw log-likelihood of the match (independent of other comparisons), and the match probability (normalised over other comparison likelihoods).

::

    struct IMatch {

        IMatch() :
            queryIdx(-1), imgIdx(-1), likelihood(-DBL_MAX), match(-DBL_MAX) {
        }
        IMatch(int _queryIdx, int _imgIdx, double _likelihood, double _match) :
            queryIdx(_queryIdx), imgIdx(_imgIdx), likelihood(_likelihood), match(
                    _match) {
        }

        int queryIdx;    //query index
        int imgIdx;      //test index

        double likelihood;  //raw loglikelihood
        double match;      //normalised probability

        bool operator<(const IMatch& m) const {
            return match < m.match;
        }

    };

of2::ChowLiuTree
--------------------

.. ocv:class:: of2::ChowLiuTree

The Chow-Liu tree is a probabilistic model of the environment in terms of feature occurance and co-occurance. The Chow-Liu tree is a form of Bayesian network. FAB-MAP uses the model when calculating bag-of-words similarity by taking into account feature saliency. Training data is provided to the ChowLiuTree class in the form of bag-of-words image descriptors. The make function produces a cv::Mat that encodes the tree structure.

.. ocv:function:: of2::ChowLiuTree::ChowLiuTree()

.. ocv:function:: void of2::ChowLiuTree::add(const Mat& imgDescriptor)

    :param imgDescriptor:  bag-of-words image descriptors stored as rows in a Mat

.. ocv:function:: void of2::ChowLiuTree::add(const vector<Mat>& imgDescriptors)

    :param imgDescriptors: a vector containing multiple bag-of-words image descriptors

.. ocv:function:: const vector<Mat>& of2::ChowLiuTree::getImgDescriptors() const

    Returns a vector containing multiple bag-of-words image descriptors

.. ocv:function:: Mat of2::ChowLiuTree::make(double infoThreshold = 0.0)

    :param infoThreshold: a threshold can be set to reduce the amount of memory used when making the Chow-Liu tree, which can occur with large vocabulary sizes. This function can fail if the threshold is set too high. If memory is an issue the value must be set by trial and error (~0.0005)


of2::BOWMSCTrainer
--------------------

.. ocv:class:: of2::BOWMSCTrainer : public of2::BOWTrainer

BOWMSCTrainer is a custom clustering algorithm used to produce the feature vocabulary required to create bag-of-words representations. The algorithm is an implementation of [AVC2007]_. Arguments against using K-means for the FAB-MAP algorithm are discussed in [IJRR2010]_. The BOWMSCTrainer inherits from the cv::BOWTrainer class, overwriting the cluster function.

.. ocv:function::   of2::BOWMSCTrainer::BOWMSCTrainer(double clusterSize = 0.4)

    :param clusterSize: the specificity of the vocabulary produced. A smaller cluster size will instigate a larger vocabulary.

.. ocv:function::  virtual Mat of2::BOWMSCTrainer::cluster() const

Cluster using features added to the class

.. ocv:function:: virtual Mat of2::BOWMSCTrainer::cluster(const Mat& descriptors) const

    :param descriptors: feature descriptors provided as rows of the Mat.

.. [AVC2007] Alexandra Teynor and Hans Burkhardt, "Fast Codebook Generation by Sequential Data Analysis for Object Classification", in Advances in Visual Computing, pp. 610-620, 2007
