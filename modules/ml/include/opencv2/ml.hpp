/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2014, Itseez Inc, all rights reserved.
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

#ifndef __OPENCV_ML_HPP__
#define __OPENCV_ML_HPP__

#ifdef __cplusplus
#  include "opencv2/core.hpp"
#endif

#ifdef __cplusplus

#include <float.h>
#include <map>
#include <iostream>

/**
  @defgroup ml Machine Learning
  @{
@defgroup ml_stat Statistical Models
@defgroup ml_bayes Normal Bayes Classifier

This simple classification model assumes that feature vectors from each class are normally
distributed (though, not necessarily independently distributed). So, the whole data distribution
function is assumed to be a Gaussian mixture, one component per class. Using the training data the
algorithm estimates mean vectors and covariance matrices for every class, and then it uses them for
prediction.

@defgroup ml_knearest K-Nearest Neighbors

The algorithm caches all training samples and predicts the response for a new sample by analyzing a
certain number (**K**) of the nearest neighbors of the sample using voting, calculating weighted
sum, and so on. The method is sometimes referred to as "learning by example" because for prediction
it looks for the feature vector with a known response that is closest to the given vector.

@defgroup ml_svm Support Vector Machines

Originally, support vector machines (SVM) was a technique for building an optimal binary (2-class)
classifier. Later the technique was extended to regression and clustering problems. SVM is a partial
case of kernel-based methods. It maps feature vectors into a higher-dimensional space using a kernel
function and builds an optimal linear discriminating function in this space or an optimal
hyper-plane that fits into the training data. In case of SVM, the kernel is not defined explicitly.
Instead, a distance between any 2 points in the hyper-space needs to be defined.

The solution is optimal, which means that the margin between the separating hyper-plane and the
nearest feature vectors from both classes (in case of 2-class classifier) is maximal. The feature
vectors that are the closest to the hyper-plane are called *support vectors*, which means that the
position of other vectors does not affect the hyper-plane (the decision function).

SVM implementation in OpenCV is based on @cite LibSVM .

Prediction with SVM
-------------------

StatModel::predict(samples, results, flags) should be used. Pass flags=StatModel::RAW_OUTPUT to get
the raw response from SVM (in the case of regression, 1-class or 2-class classification problem).

@defgroup ml_decsiontrees Decision Trees

The ML classes discussed in this section implement Classification and Regression Tree algorithms
described in @cite Breiman84 .

The class cv::ml::DTrees represents a single decision tree or a collection of decision trees. It's
also a base class for RTrees and Boost.

A decision tree is a binary tree (tree where each non-leaf node has two child nodes). It can be used
either for classification or for regression. For classification, each tree leaf is marked with a
class label; multiple leaves may have the same label. For regression, a constant is also assigned to
each tree leaf, so the approximation function is piecewise constant.

Predicting with Decision Trees
------------------------------

To reach a leaf node and to obtain a response for the input feature vector, the prediction procedure
starts with the root node. From each non-leaf node the procedure goes to the left (selects the left
child node as the next observed node) or to the right based on the value of a certain variable whose
index is stored in the observed node. The following variables are possible:

-   **Ordered variables.** The variable value is compared with a threshold that is also stored in
    the node. If the value is less than the threshold, the procedure goes to the left. Otherwise, it
    goes to the right. For example, if the weight is less than 1 kilogram, the procedure goes to the
    left, else to the right.

-   **Categorical variables.** A discrete variable value is tested to see whether it belongs to a
    certain subset of values (also stored in the node) from a limited set of values the variable
    could take. If it does, the procedure goes to the left. Otherwise, it goes to the right. For
    example, if the color is green or red, go to the left, else to the right.

So, in each node, a pair of entities (variable_index , `decision_rule (threshold/subset)` ) is
used. This pair is called a *split* (split on the variable variable_index ). Once a leaf node is
reached, the value assigned to this node is used as the output of the prediction procedure.

Sometimes, certain features of the input vector are missed (for example, in the darkness it is
difficult to determine the object color), and the prediction procedure may get stuck in the certain
node (in the mentioned example, if the node is split by color). To avoid such situations, decision
trees use so-called *surrogate splits*. That is, in addition to the best "primary" split, every tree
node may also be split to one or more other variables with nearly the same results.

Training Decision Trees
-----------------------

The tree is built recursively, starting from the root node. All training data (feature vectors and
responses) is used to split the root node. In each node the optimum decision rule (the best
"primary" split) is found based on some criteria. In machine learning, gini "purity" criteria are
used for classification, and sum of squared errors is used for regression. Then, if necessary, the
surrogate splits are found. They resemble the results of the primary split on the training data. All
the data is divided using the primary and the surrogate splits (like it is done in the prediction
procedure) between the left and the right child node. Then, the procedure recursively splits both
left and right nodes. At each node the recursive procedure may stop (that is, stop splitting the
node further) in one of the following cases:

-   Depth of the constructed tree branch has reached the specified maximum value.
-   Number of training samples in the node is less than the specified threshold when it is not
    statistically representative to split the node further.
-   All the samples in the node belong to the same class or, in case of regression, the variation is
    too small.
-   The best found split does not give any noticeable improvement compared to a random choice.

When the tree is built, it may be pruned using a cross-validation procedure, if necessary. That is,
some branches of the tree that may lead to the model overfitting are cut off. Normally, this
procedure is only applied to standalone decision trees. Usually tree ensembles build trees that are
small enough and use their own protection schemes against overfitting.

Variable Importance
-------------------

Besides the prediction that is an obvious use of decision trees, the tree can be also used for
various data analyses. One of the key properties of the constructed decision tree algorithms is an
ability to compute the importance (relative decisive power) of each variable. For example, in a spam
filter that uses a set of words occurred in the message as a feature vector, the variable importance
rating can be used to determine the most "spam-indicating" words and thus help keep the dictionary
size reasonable.

Importance of each variable is computed over all the splits on this variable in the tree, primary
and surrogate ones. Thus, to compute variable importance correctly, the surrogate splits must be
enabled in the training parameters, even if there is no missing data.

@defgroup ml_boost Boosting

A common machine learning task is supervised learning. In supervised learning, the goal is to learn
the functional relationship \f$F: y = F(x)\f$ between the input \f$x\f$ and the output \f$y\f$ . Predicting the
qualitative output is called *classification*, while predicting the quantitative output is called
*regression*.

Boosting is a powerful learning concept that provides a solution to the supervised classification
learning task. It combines the performance of many "weak" classifiers to produce a powerful
committee @cite HTF01 . A weak classifier is only required to be better than chance, and thus can be
very simple and computationally inexpensive. However, many of them smartly combine results to a
strong classifier that often outperforms most "monolithic" strong classifiers such as SVMs and
Neural Networks.

Decision trees are the most popular weak classifiers used in boosting schemes. Often the simplest
decision trees with only a single split node per tree (called stumps ) are sufficient.

The boosted model is based on \f$N\f$ training examples \f${(x_i,y_i)}1N\f$ with \f$x_i \in{R^K}\f$ and
\f$y_i \in{-1, +1}\f$ . \f$x_i\f$ is a \f$K\f$ -component vector. Each component encodes a feature relevant to
the learning task at hand. The desired two-class output is encoded as -1 and +1.

Different variants of boosting are known as Discrete Adaboost, Real AdaBoost, LogitBoost, and Gentle
AdaBoost @cite FHT98 . All of them are very similar in their overall structure. Therefore, this chapter
focuses only on the standard two-class Discrete AdaBoost algorithm, outlined below. Initially the
same weight is assigned to each sample (step 2). Then, a weak classifier \f$f_{m(x)}\f$ is trained on
the weighted training data (step 3a). Its weighted training error and scaling factor \f$c_m\f$ is
computed (step 3b). The weights are increased for training samples that have been misclassified
(step 3c). All weights are then normalized, and the process of finding the next weak classifier
continues for another \f$M\f$ -1 times. The final classifier \f$F(x)\f$ is the sign of the weighted sum over
the individual weak classifiers (step 4).

**Two-class Discrete AdaBoost Algorithm**

-   Set \f$N\f$ examples \f${(x_i,y_i)}1N\f$ with \f$x_i \in{R^K}, y_i \in{-1, +1}\f$ .

-   Assign weights as \f$w_i = 1/N, i = 1,...,N\f$ .

-   Repeat for \f$m = 1,2,...,M\f$ :

    3.1. Fit the classifier \f$f_m(x) \in{-1,1}\f$, using weights \f$w_i\f$ on the training data.

    3.2. Compute \f$err_m = E_w [1_{(y \neq f_m(x))}], c_m = log((1 - err_m)/err_m)\f$ .

    3.3. Set \f$w_i \Leftarrow w_i exp[c_m 1_{(y_i \neq f_m(x_i))}], i = 1,2,...,N,\f$ and renormalize
    so that \f$\Sigma i w_i = 1\f$ .

1.  Classify new samples *x* using the formula: \f$\textrm{sign} (\Sigma m = 1M c_m f_m(x))\f$ .

@note Similar to the classical boosting methods, the current implementation supports two-class
classifiers only. For M \> 2 classes, there is the **AdaBoost.MH** algorithm (described in
@cite FHT98) that reduces the problem to the two-class problem, yet with a much larger training set.
To reduce computation time for boosted models without substantially losing accuracy, the influence
trimming technique can be employed. As the training algorithm proceeds and the number of trees in
the ensemble is increased, a larger number of the training samples are classified correctly and with
increasing confidence, thereby those samples receive smaller weights on the subsequent iterations.
Examples with a very low relative weight have a small impact on the weak classifier training. Thus,
such examples may be excluded during the weak classifier training without having much effect on the
induced classifier. This process is controlled with the weight_trim_rate parameter. Only examples
with the summary fraction weight_trim_rate of the total weight mass are used in the weak
classifier training. Note that the weights for **all** training examples are recomputed at each
training iteration. Examples deleted at a particular iteration may be used again for learning some
of the weak classifiers further @cite FHT98 .

Prediction with Boost
---------------------
StatModel::predict(samples, results, flags) should be used. Pass flags=StatModel::RAW_OUTPUT to get
the raw sum from Boost classifier.

@defgroup ml_randomtrees Random Trees

Random trees have been introduced by Leo Breiman and Adele Cutler:
<http://www.stat.berkeley.edu/users/breiman/RandomForests/> . The algorithm can deal with both
classification and regression problems. Random trees is a collection (ensemble) of tree predictors
that is called *forest* further in this section (the term has been also introduced by L. Breiman).
The classification works as follows: the random trees classifier takes the input feature vector,
classifies it with every tree in the forest, and outputs the class label that received the majority
of "votes". In case of a regression, the classifier response is the average of the responses over
all the trees in the forest.

All the trees are trained with the same parameters but on different training sets. These sets are
generated from the original training set using the bootstrap procedure: for each training set, you
randomly select the same number of vectors as in the original set ( =N ). The vectors are chosen
with replacement. That is, some vectors will occur more than once and some will be absent. At each
node of each trained tree, not all the variables are used to find the best split, but a random
subset of them. With each node a new subset is generated. However, its size is fixed for all the
nodes and all the trees. It is a training parameter set to \f$\sqrt{number_of_variables}\f$ by
default. None of the built trees are pruned.

In random trees there is no need for any accuracy estimation procedures, such as cross-validation or
bootstrap, or a separate test set to get an estimate of the training error. The error is estimated
internally during the training. When the training set for the current tree is drawn by sampling with
replacement, some vectors are left out (so-called *oob (out-of-bag) data* ). The size of oob data is
about N/3 . The classification error is estimated by using this oob-data as follows:

-   Get a prediction for each vector, which is oob relative to the i-th tree, using the very i-th
    tree.

-   After all the trees have been trained, for each vector that has ever been oob, find the
    class-*winner* for it (the class that has got the majority of votes in the trees where the
    vector was oob) and compare it to the ground-truth response.

-   Compute the classification error estimate as a ratio of the number of misclassified oob vectors
    to all the vectors in the original data. In case of regression, the oob-error is computed as the
    squared error for oob vectors difference divided by the total number of vectors.

For the random trees usage example, please, see letter_recog.cpp sample in OpenCV distribution.

**References:**

-   *Machine Learning*, Wald I, July 2002.
<http://stat-www.berkeley.edu/users/breiman/wald2002-1.pdf>
-   *Looking Inside the Black Box*, Wald II, July 2002.
<http://stat-www.berkeley.edu/users/breiman/wald2002-2.pdf>
-   *Software for the Masses*, Wald III, July 2002.
<http://stat-www.berkeley.edu/users/breiman/wald2002-3.pdf>
-   And other articles from the web site
<http://www.stat.berkeley.edu/users/breiman/RandomForests/cc_home.htm>

@defgroup ml_em Expectation Maximization

The Expectation Maximization(EM) algorithm estimates the parameters of the multivariate probability
density function in the form of a Gaussian mixture distribution with a specified number of mixtures.

Consider the set of the N feature vectors { \f$x_1, x_2,...,x_{N}\f$ } from a d-dimensional Euclidean
space drawn from a Gaussian mixture:

\f[p(x;a_k,S_k, \pi _k) =  \sum _{k=1}^{m} \pi _kp_k(x),  \quad \pi _k  \geq 0,  \quad \sum _{k=1}^{m} \pi _k=1,\f]

\f[p_k(x)= \varphi (x;a_k,S_k)= \frac{1}{(2\pi)^{d/2}\mid{S_k}\mid^{1/2}} exp \left \{ - \frac{1}{2} (x-a_k)^TS_k^{-1}(x-a_k) \right \} ,\f]

where \f$m\f$ is the number of mixtures, \f$p_k\f$ is the normal distribution density with the mean \f$a_k\f$
and covariance matrix \f$S_k\f$, \f$\pi_k\f$ is the weight of the k-th mixture. Given the number of mixtures
\f$M\f$ and the samples \f$x_i\f$, \f$i=1..N\f$ the algorithm finds the maximum-likelihood estimates (MLE) of
all the mixture parameters, that is, \f$a_k\f$, \f$S_k\f$ and \f$\pi_k\f$ :

\f[L(x, \theta )=logp(x, \theta )= \sum _{i=1}^{N}log \left ( \sum _{k=1}^{m} \pi _kp_k(x) \right ) \to \max _{ \theta \in \Theta },\f]

\f[\Theta = \left \{ (a_k,S_k, \pi _k): a_k  \in \mathbbm{R} ^d,S_k=S_k^T>0,S_k  \in \mathbbm{R} ^{d  \times d}, \pi _k \geq 0, \sum _{k=1}^{m} \pi _k=1 \right \} .\f]

The EM algorithm is an iterative procedure. Each iteration includes two steps. At the first step
(Expectation step or E-step), you find a probability \f$p_{i,k}\f$ (denoted \f$\alpha_{i,k}\f$ in the
formula below) of sample i to belong to mixture k using the currently available mixture parameter
estimates:

\f[\alpha _{ki} =  \frac{\pi_k\varphi(x;a_k,S_k)}{\sum\limits_{j=1}^{m}\pi_j\varphi(x;a_j,S_j)} .\f]

At the second step (Maximization step or M-step), the mixture parameter estimates are refined using
the computed probabilities:

\f[\pi _k= \frac{1}{N} \sum _{i=1}^{N} \alpha _{ki},  \quad a_k= \frac{\sum\limits_{i=1}^{N}\alpha_{ki}x_i}{\sum\limits_{i=1}^{N}\alpha_{ki}} ,  \quad S_k= \frac{\sum\limits_{i=1}^{N}\alpha_{ki}(x_i-a_k)(x_i-a_k)^T}{\sum\limits_{i=1}^{N}\alpha_{ki}}\f]

Alternatively, the algorithm may start with the M-step when the initial values for \f$p_{i,k}\f$ can be
provided. Another alternative when \f$p_{i,k}\f$ are unknown is to use a simpler clustering algorithm to
pre-cluster the input samples and thus obtain initial \f$p_{i,k}\f$ . Often (including machine learning)
the k-means algorithm is used for that purpose.

One of the main problems of the EM algorithm is a large number of parameters to estimate. The
majority of the parameters reside in covariance matrices, which are \f$d \times d\f$ elements each where
\f$d\f$ is the feature space dimensionality. However, in many practical problems, the covariance
matrices are close to diagonal or even to \f$\mu_k*I\f$ , where \f$I\f$ is an identity matrix and \f$\mu_k\f$ is
a mixture-dependent "scale" parameter. So, a robust computation scheme could start with harder
constraints on the covariance matrices and then use the estimated parameters as an input for a less
constrained optimization problem (often a diagonal covariance matrix is already a good enough
approximation).

References:
-   Bilmes98 J. A. Bilmes. *A Gentle Tutorial of the EM Algorithm and its Application to Parameter
    Estimation for Gaussian Mixture and Hidden Markov Models*. Technical Report TR-97-021,
    International Computer Science Institute and Computer Science Division, University of California
    at Berkeley, April 1998.

@defgroup ml_neural Neural Networks

ML implements feed-forward artificial neural networks or, more particularly, multi-layer perceptrons
(MLP), the most commonly used type of neural networks. MLP consists of the input layer, output
layer, and one or more hidden layers. Each layer of MLP includes one or more neurons directionally
linked with the neurons from the previous and the next layer. The example below represents a 3-layer
perceptron with three inputs, two outputs, and the hidden layer including five neurons:

![image](pics/mlp.png)

All the neurons in MLP are similar. Each of them has several input links (it takes the output values
from several neurons in the previous layer as input) and several output links (it passes the
response to several neurons in the next layer). The values retrieved from the previous layer are
summed up with certain weights, individual for each neuron, plus the bias term. The sum is
transformed using the activation function \f$f\f$ that may be also different for different neurons.

![image](pics/neuron_model.png)

In other words, given the outputs \f$x_j\f$ of the layer \f$n\f$ , the outputs \f$y_i\f$ of the layer \f$n+1\f$ are
computed as:

\f[u_i =  \sum _j (w^{n+1}_{i,j}*x_j) + w^{n+1}_{i,bias}\f]

\f[y_i = f(u_i)\f]

Different activation functions may be used. ML implements three standard functions:

-   Identity function ( ANN_MLP::IDENTITY ): \f$f(x)=x\f$

-   Symmetrical sigmoid ( ANN_MLP::SIGMOID_SYM ): \f$f(x)=\beta*(1-e^{-\alpha x})/(1+e^{-\alpha x}\f$
    ), which is the default choice for MLP. The standard sigmoid with \f$\beta =1, \alpha =1\f$ is shown
    below:

    ![image](pics/sigmoid_bipolar.png)

-   Gaussian function ( ANN_MLP::GAUSSIAN ): \f$f(x)=\beta e^{-\alpha x*x}\f$ , which is not completely
    supported at the moment.

In ML, all the neurons have the same activation functions, with the same free parameters (
\f$\alpha, \beta\f$ ) that are specified by user and are not altered by the training algorithms.

So, the whole trained network works as follows:

1.  Take the feature vector as input. The vector size is equal to the size of the input layer.
2.  Pass values as input to the first hidden layer.
3.  Compute outputs of the hidden layer using the weights and the activation functions.
4.  Pass outputs further downstream until you compute the output layer.

So, to compute the network, you need to know all the weights \f$w^{n+1)}_{i,j}\f$ . The weights are
computed by the training algorithm. The algorithm takes a training set, multiple input vectors with
the corresponding output vectors, and iteratively adjusts the weights to enable the network to give
the desired response to the provided input vectors.

The larger the network size (the number of hidden layers and their sizes) is, the more the potential
network flexibility is. The error on the training set could be made arbitrarily small. But at the
same time the learned network also "learns" the noise present in the training set, so the error on
the test set usually starts increasing after the network size reaches a limit. Besides, the larger
networks are trained much longer than the smaller ones, so it is reasonable to pre-process the data,
using PCA::operator() or similar technique, and train a smaller network on only essential features.

Another MLP feature is an inability to handle categorical data as is. However, there is a
workaround. If a certain feature in the input or output (in case of n -class classifier for \f$n>2\f$ )
layer is categorical and can take \f$M>2\f$ different values, it makes sense to represent it as a binary
tuple of M elements, where the i -th element is 1 if and only if the feature is equal to the i -th
value out of M possible. It increases the size of the input/output layer but speeds up the training
algorithm convergence and at the same time enables "fuzzy" values of such variables, that is, a
tuple of probabilities instead of a fixed value.

ML implements two algorithms for training MLP's. The first algorithm is a classical random
sequential back-propagation algorithm. The second (default) one is a batch RPROP algorithm.

@defgroup ml_lr Logistic Regression

ML implements logistic regression, which is a probabilistic classification technique. Logistic
Regression is a binary classification algorithm which is closely related to Support Vector Machines
(SVM). Like SVM, Logistic Regression can be extended to work on multi-class classification problems
like digit recognition (i.e. recognizing digitis like 0,1 2, 3,... from the given images). This
version of Logistic Regression supports both binary and multi-class classifications (for multi-class
it creates a multiple 2-class classifiers). In order to train the logistic regression classifier,
Batch Gradient Descent and Mini-Batch Gradient Descent algorithms are used (see <http://en.wikipedia.org/wiki/Gradient_descent_optimization>).
Logistic Regression is a discriminative classifier (see <http://www.cs.cmu.edu/~tom/NewChapters.html> for more details).
Logistic Regression is implemented as a C++ class in LogisticRegression.

In Logistic Regression, we try to optimize the training paramater \f$\theta\f$ such that the hypothesis
\f$0 \leq h_\theta(x) \leq 1\f$ is acheived. We have \f$h_\theta(x) = g(h_\theta(x))\f$ and
\f$g(z) = \frac{1}{1+e^{-z}}\f$ as the logistic or sigmoid function. The term "Logistic" in Logistic
Regression refers to this function. For given data of a binary classification problem of classes 0
and 1, one can determine that the given data instance belongs to class 1 if \f$h_\theta(x) \geq 0.5\f$
or class 0 if \f$h_\theta(x) < 0.5\f$ .

In Logistic Regression, choosing the right parameters is of utmost importance for reducing the
training error and ensuring high training accuracy. LogisticRegression::Params is the structure that
defines parameters that are required to train a Logistic Regression classifier. The learning rate is
determined by LogisticRegression::Params.alpha. It determines how faster we approach the solution.
It is a positive real number. Optimization algorithms like Batch Gradient Descent and Mini-Batch
Gradient Descent are supported in LogisticRegression. It is important that we mention the number of
iterations these optimization algorithms have to run. The number of iterations are mentioned by
LogisticRegression::Params.num_iters. The number of iterations can be thought as number of steps
taken and learning rate specifies if it is a long step or a short step. These two parameters define
how fast we arrive at a possible solution. In order to compensate for overfitting regularization is
performed, which can be enabled by setting LogisticRegression::Params.regularized to a positive
integer (greater than zero). One can specify what kind of regularization has to be performed by
setting LogisticRegression::Params.norm to LogisticRegression::REG_L1 or
LogisticRegression::REG_L2 values. LogisticRegression provides a choice of 2 training methods with
Batch Gradient Descent or the Mini-Batch Gradient Descent. To specify this, set
LogisticRegression::Params.train_method to either LogisticRegression::BATCH or
LogisticRegression::MINI_BATCH. If LogisticRegression::Params is set to
LogisticRegression::MINI_BATCH, the size of the mini batch has to be to a postive integer using
LogisticRegression::Params.mini_batch_size.

A sample set of training parameters for the Logistic Regression classifier can be initialized as
follows:
@code
    LogisticRegression::Params params;
    params.alpha = 0.5;
    params.num_iters = 10000;
    params.norm = LogisticRegression::REG_L2;
    params.regularized = 1;
    params.train_method = LogisticRegression::MINI_BATCH;
    params.mini_batch_size = 10;
@endcode

@defgroup ml_data Training Data

In machine learning algorithms there is notion of training data. Training data includes several
components:

-   A set of training samples. Each training sample is a vector of values (in Computer Vision it's
    sometimes referred to as feature vector). Usually all the vectors have the same number of
    components (features); OpenCV ml module assumes that. Each feature can be ordered (i.e. its
    values are floating-point numbers that can be compared with each other and strictly ordered,
    i.e. sorted) or categorical (i.e. its value belongs to a fixed set of values that can be
    integers, strings etc.).
-   Optional set of responses corresponding to the samples. Training data with no responses is used
    in unsupervised learning algorithms that learn structure of the supplied data based on distances
    between different samples. Training data with responses is used in supervised learning
    algorithms, which learn the function mapping samples to responses. Usually the responses are
    scalar values, ordered (when we deal with regression problem) or categorical (when we deal with
    classification problem; in this case the responses are often called "labels"). Some algorithms,
    most noticeably Neural networks, can handle not only scalar, but also multi-dimensional or
    vector responses.
-   Another optional component is the mask of missing measurements. Most algorithms require all the
    components in all the training samples be valid, but some other algorithms, such as decision
    tress, can handle the cases of missing measurements.
-   In the case of classification problem user may want to give different weights to different
    classes. This is useful, for example, when
    -   user wants to shift prediction accuracy towards lower false-alarm rate or higher hit-rate.
    -   user wants to compensate for significantly different amounts of training samples from
        different classes.
-   In addition to that, each training sample may be given a weight, if user wants the algorithm to
    pay special attention to certain training samples and adjust the training model accordingly.
-   Also, user may wish not to use the whole training data at once, but rather use parts of it, e.g.
    to do parameter optimization via cross-validation procedure.

As you can see, training data can have rather complex structure; besides, it may be very big and/or
not entirely available, so there is need to make abstraction for this concept. In OpenCV ml there is
cv::ml::TrainData class for that.

  @}
 */

namespace cv
{

namespace ml
{

//! @addtogroup ml
//! @{

/* Variable type */
enum
{
    VAR_NUMERICAL    =0,
    VAR_ORDERED      =0,
    VAR_CATEGORICAL  =1
};

enum
{
    TEST_ERROR = 0,
    TRAIN_ERROR = 1
};

enum
{
    ROW_SAMPLE = 0,
    COL_SAMPLE = 1
};

//! @addtogroup ml_svm
//! @{

/** @brief The structure represents the logarithmic grid range of statmodel parameters.

It is used for optimizing statmodel accuracy by varying model parameters, the accuracy estimate
being computed by cross-validation.
-   member double ParamGrid::minVal
Minimum value of the statmodel parameter.
-   member double ParamGrid::maxVal
Maximum value of the statmodel parameter.
-   member double ParamGrid::logStep
Logarithmic step for iterating the statmodel parameter.
The grid determines the following iteration sequence of the statmodel parameter values:

\f[(minVal, minVal*step, minVal*{step}^2, \dots,  minVal*{logStep}^n),\f]

where \f$n\f$ is the maximal index satisfying

\f[\texttt{minVal} * \texttt{logStep} ^n <  \texttt{maxVal}\f]

The grid is logarithmic, so logStep must always be greater then 1.
 */
class CV_EXPORTS_W_MAP ParamGrid
{
public:
    /** @brief The constructors.

    The full constructor initializes corresponding members. The default constructor creates a dummy
    grid:
    @code
        ParamGrid::ParamGrid()
        {
            minVal = maxVal = 0;
            logStep = 1;
        }
    @endcode
     */
    ParamGrid();
    ParamGrid(double _minVal, double _maxVal, double _logStep);

    CV_PROP_RW double minVal;
    CV_PROP_RW double maxVal;
    CV_PROP_RW double logStep;
};

//! @} ml_svm

//! @addtogroup ml_data
//! @{

/** @brief Class encapsulating training data.

Please note that the class only specifies the interface of training data, but not implementation.
All the statistical model classes in ml take Ptr\<TrainData\>. In other words, you can create your
own class derived from TrainData and supply smart pointer to the instance of this class into
StatModel::train.
 */
class CV_EXPORTS TrainData
{
public:
    static inline float missingValue() { return FLT_MAX; }
    virtual ~TrainData();

    virtual int getLayout() const = 0;
    virtual int getNTrainSamples() const = 0;
    virtual int getNTestSamples() const = 0;
    virtual int getNSamples() const = 0;
    virtual int getNVars() const = 0;
    virtual int getNAllVars() const = 0;

    virtual void getSample(InputArray varIdx, int sidx, float* buf) const = 0;
    virtual Mat getSamples() const = 0;
    virtual Mat getMissing() const = 0;

    /** @brief Returns matrix of train samples

    @param layout The requested layout. If it's different from the initial one, the matrix is
    transposed.
    @param compressSamples if true, the function returns only the training samples (specified by
    sampleIdx)
    @param compressVars if true, the function returns the shorter training samples, containing only
    the active variables.

    In current implementation the function tries to avoid physical data copying and returns the matrix
    stored inside TrainData (unless the transposition or compression is needed).
     */
    virtual Mat getTrainSamples(int layout=ROW_SAMPLE,
                                bool compressSamples=true,
                                bool compressVars=true) const = 0;

    /** @brief Returns the vector of responses

    The function returns ordered or the original categorical responses. Usually it's used in regression
    algorithms.
     */
    virtual Mat getTrainResponses() const = 0;

    /** @brief Returns the vector of normalized categorical responses

    The function returns vector of responses. Each response is integer from 0 to \<number of
    classes\>-1. The actual label value can be retrieved then from the class label vector, see
    TrainData::getClassLabels.
     */
    virtual Mat getTrainNormCatResponses() const = 0;
    virtual Mat getTestResponses() const = 0;
    virtual Mat getTestNormCatResponses() const = 0;
    virtual Mat getResponses() const = 0;
    virtual Mat getNormCatResponses() const = 0;
    virtual Mat getSampleWeights() const = 0;
    virtual Mat getTrainSampleWeights() const = 0;
    virtual Mat getTestSampleWeights() const = 0;
    virtual Mat getVarIdx() const = 0;
    virtual Mat getVarType() const = 0;
    virtual int getResponseType() const = 0;
    virtual Mat getTrainSampleIdx() const = 0;
    virtual Mat getTestSampleIdx() const = 0;
    virtual void getValues(int vi, InputArray sidx, float* values) const = 0;
    virtual void getNormCatValues(int vi, InputArray sidx, int* values) const = 0;
    virtual Mat getDefaultSubstValues() const = 0;

    virtual int getCatCount(int vi) const = 0;

    /** @brief Returns the vector of class labels

    The function returns vector of unique labels occurred in the responses.
     */
    virtual Mat getClassLabels() const = 0;

    virtual Mat getCatOfs() const = 0;
    virtual Mat getCatMap() const = 0;

    virtual void setTrainTestSplit(int count, bool shuffle=true) = 0;

    /** @brief Splits the training data into the training and test parts

    The function selects a subset of specified relative size and then returns it as the training set. If
    the function is not called, all the data is used for training. Please, note that for each of
    TrainData::getTrain\* there is corresponding TrainData::getTest\*, so that the test subset can be
    retrieved and processed as well.
     */
    virtual void setTrainTestSplitRatio(double ratio, bool shuffle=true) = 0;
    virtual void shuffleTrainTest() = 0;

    static Mat getSubVector(const Mat& vec, const Mat& idx);

    /** @brief Reads the dataset from a .csv file and returns the ready-to-use training data.

    @param filename The input file name
    @param headerLineCount The number of lines in the beginning to skip; besides the header, the
    function also skips empty lines and lines staring with '\#'
    @param responseStartIdx Index of the first output variable. If -1, the function considers the last
    variable as the response
    @param responseEndIdx Index of the last output variable + 1. If -1, then there is single response
    variable at responseStartIdx.
    @param varTypeSpec The optional text string that specifies the variables' types. It has the format ord[n1-n2,n3,n4-n5,...]cat[n6,n7-n8,...]. That is, variables from n1 to n2 (inclusive range), n3, n4 to n5 ... are considered ordered and n6, n7 to n8 ... are considered as categorical. The range [n1..n2] + [n3] + [n4..n5] + ... + [n6] + [n7..n8] should cover all the variables. If varTypeSpec is not specified, then algorithm uses the following rules:
    # all input variables are considered ordered by default. If some column contains has
    non-numerical values, e.g. 'apple', 'pear', 'apple', 'apple', 'mango', the corresponding
    variable is considered categorical.
    # if there are several output variables, they are all considered as ordered. Error is
    reported when non-numerical values are used.
    # if there is a single output variable, then if its values are non-numerical or are all
    integers, then it's considered categorical. Otherwise, it's considered ordered.
    @param delimiter The character used to separate values in each line.
    @param missch The character used to specify missing measurements. It should not be a digit.
    Although it's a non-numerical value, it surely does not affect the decision of whether the
    variable ordered or categorical.
     */
    static Ptr<TrainData> loadFromCSV(const String& filename,
                                      int headerLineCount,
                                      int responseStartIdx=-1,
                                      int responseEndIdx=-1,
                                      const String& varTypeSpec=String(),
                                      char delimiter=',',
                                      char missch='?');
    /** @brief Creates training data from in-memory arrays.

    @param samples matrix of samples. It should have CV_32F type.
    @param layout it's either ROW_SAMPLE, which means that each training sample is a row of samples,
    or COL_SAMPLE, which means that each training sample occupies a column of samples.
    @param responses matrix of responses. If the responses are scalar, they should be stored as a
    single row or as a single column. The matrix should have type CV_32F or CV_32S (in the former
    case the responses are considered as ordered by default; in the latter case - as categorical)
    @param varIdx vector specifying which variables to use for training. It can be an integer vector
    (CV_32S) containing 0-based variable indices or byte vector (CV_8U) containing a mask of active
    variables.
    @param sampleIdx vector specifying which samples to use for training. It can be an integer vector
    (CV_32S) containing 0-based sample indices or byte vector (CV_8U) containing a mask of training
    samples.
    @param sampleWeights optional vector with weights for each sample. It should have CV_32F type.
    @param varType optional vector of type CV_8U and size \<number_of_variables_in_samples\> +
    \<number_of_variables_in_responses\>, containing types of each input and output variable. The
    ordered variables are denoted by value VAR_ORDERED, and categorical - by VAR_CATEGORICAL.
     */
    static Ptr<TrainData> create(InputArray samples, int layout, InputArray responses,
                                 InputArray varIdx=noArray(), InputArray sampleIdx=noArray(),
                                 InputArray sampleWeights=noArray(), InputArray varType=noArray());
};

//! @} ml_data

//! @addtogroup ml_stat
//! @{

/** @brief Base class for statistical models in OpenCV ML.
 */
class CV_EXPORTS_W StatModel : public Algorithm
{
public:
    enum { UPDATE_MODEL = 1, RAW_OUTPUT=1, COMPRESSED_INPUT=2, PREPROCESSED_INPUT=4 };
    virtual void clear();

    /** @brief Returns the number of variables in training samples

    The method must be overwritten in the derived classes.
     */
    virtual int getVarCount() const = 0;

    /** @brief Returns true if the model is trained

    The method must be overwritten in the derived classes.
     */
    virtual bool isTrained() const = 0;
    /** @brief Returns true if the model is classifier

    The method must be overwritten in the derived classes.
     */
    virtual bool isClassifier() const = 0;

    /** @brief Trains the statistical model

    @param trainData training data that can be loaded from file using TrainData::loadFromCSV or
    created with TrainData::create.
    @param flags optional flags, depending on the model. Some of the models can be updated with the
    new training samples, not completely overwritten (such as NormalBayesClassifier or ANN_MLP).

    There are 2 instance methods and 2 static (class) template methods. The first two train the already
    created model (the very first method must be overwritten in the derived classes). And the latter two
    variants are convenience methods that construct empty model and then call its train method.
     */
    virtual bool train( const Ptr<TrainData>& trainData, int flags=0 );
    /** @overload
    @param samples training samples
    @param layout ROW_SAMPLE (training samples are the matrix rows) or COL_SAMPLE (training samples
    are the matrix columns)
    @param responses vector of responses associated with the training samples.
    */
    virtual bool train( InputArray samples, int layout, InputArray responses );

    /** @brief Computes error on the training or test dataset

    @param data the training data
    @param test if true, the error is computed over the test subset of the data, otherwise it's
    computed over the training subset of the data. Please note that if you loaded a completely
    different dataset to evaluate already trained classifier, you will probably want not to set the
    test subset at all with TrainData::setTrainTestSplitRatio and specify test=false, so that the
    error is computed for the whole new set. Yes, this sounds a bit confusing.
    @param resp the optional output responses.

    The method uses StatModel::predict to compute the error. For regression models the error is computed
    as RMS, for classifiers - as a percent of missclassified samples (0%-100%).
     */
    virtual float calcError( const Ptr<TrainData>& data, bool test, OutputArray resp ) const;

    /** @brief Predicts response(s) for the provided sample(s)

    @param samples The input samples, floating-point matrix
    @param results The optional output matrix of results.
    @param flags The optional flags, model-dependent. Some models, such as Boost, SVM recognize
    StatModel::RAW_OUTPUT flag, which makes the method return the raw results (the sum), not the
    class label.
     */
    virtual float predict( InputArray samples, OutputArray results=noArray(), int flags=0 ) const = 0;

    /** @brief Loads model from the file

    This is static template method of StatModel. It's usage is following (in the case of SVM): :

        Ptr<SVM> svm = StatModel::load<SVM>("my_svm_model.xml");

    In order to make this method work, the derived class must overwrite
    Algorithm::read(const FileNode& fn).
     */
    template<typename _Tp> static Ptr<_Tp> load(const String& filename)
    {
        FileStorage fs(filename, FileStorage::READ);
        Ptr<_Tp> model = _Tp::create();
        model->read(fs.getFirstTopLevelNode());
        return model->isTrained() ? model : Ptr<_Tp>();
    }

    template<typename _Tp> static Ptr<_Tp> train(const Ptr<TrainData>& data, const typename _Tp::Params& p, int flags=0)
    {
        Ptr<_Tp> model = _Tp::create(p);
        return !model.empty() && model->train(data, flags) ? model : Ptr<_Tp>();
    }

    template<typename _Tp> static Ptr<_Tp> train(InputArray samples, int layout, InputArray responses,
                                                 const typename _Tp::Params& p, int flags=0)
    {
        Ptr<_Tp> model = _Tp::create(p);
        return !model.empty() && model->train(TrainData::create(samples, layout, responses), flags) ? model : Ptr<_Tp>();
    }

    /** @brief Saves the model to a file.

    In order to make this method work, the derived class must overwrite
    Algorithm::write(FileStorage& fs).
     */
    virtual void save(const String& filename) const;
    virtual String getDefaultModelName() const = 0;
};

//! @} ml_stat

/****************************************************************************************\
*                                 Normal Bayes Classifier                                *
\****************************************************************************************/

//! @addtogroup ml_bayes
//! @{

/** @brief Bayes classifier for normally distributed data.
 */
class CV_EXPORTS_W NormalBayesClassifier : public StatModel
{
public:
    class CV_EXPORTS_W Params
    {
    public:
        Params();
    };
    /** @brief Predicts the response for sample(s).

    The method estimates the most probable classes for input vectors. Input vectors (one or more) are
    stored as rows of the matrix inputs. In case of multiple input vectors, there should be one output
    vector outputs. The predicted class for a single input vector is returned by the method. The vector
    outputProbs contains the output probabilities corresponding to each element of result.
     */
    virtual float predictProb( InputArray inputs, OutputArray outputs,
                               OutputArray outputProbs, int flags=0 ) const = 0;
    virtual void setParams(const Params& params) = 0;
    virtual Params getParams() const = 0;

    /** @brief Creates empty model

    @param params The model parameters. There is none so far, the structure is used as a placeholder
    for possible extensions.

    Use StatModel::train to train the model,
    StatModel::train\<NormalBayesClassifier\>(traindata, params) to create and train the model,
    StatModel::load\<NormalBayesClassifier\>(filename) to load the pre-trained model.
     */
    static Ptr<NormalBayesClassifier> create(const Params& params=Params());
};

//! @} ml_bayes

/****************************************************************************************\
*                          K-Nearest Neighbour Classifier                                *
\****************************************************************************************/

//! @addtogroup ml_knearest
//! @{

/** @brief The class implements K-Nearest Neighbors model as described in the beginning of this section.

@note
   -   (Python) An example of digit recognition using KNearest can be found at
        opencv_source/samples/python2/digits.py
    -   (Python) An example of grid search digit recognition using KNearest can be found at
        opencv_source/samples/python2/digits_adjust.py
    -   (Python) An example of video digit recognition using KNearest can be found at
        opencv_source/samples/python2/digits_video.py
 */
class CV_EXPORTS_W KNearest : public StatModel
{
public:
    class CV_EXPORTS_W_MAP Params
    {
    public:
        Params(int defaultK=10, bool isclassifier_=true, int Emax_=INT_MAX, int algorithmType_=BRUTE_FORCE);

        CV_PROP_RW int defaultK;
        CV_PROP_RW bool isclassifier;
        CV_PROP_RW int Emax; // for implementation with KDTree
        CV_PROP_RW int algorithmType;
    };
    virtual void setParams(const Params& p) = 0;
    virtual Params getParams() const = 0;

    /** @brief Finds the neighbors and predicts responses for input vectors.

    @param samples Input samples stored by rows. It is a single-precision floating-point matrix of
    \<number_of_samples\> \* k size.
    @param k Number of used nearest neighbors. Should be greater than 1.
    @param results Vector with results of prediction (regression or classification) for each input
    sample. It is a single-precision floating-point vector with \<number_of_samples\> elements.
    @param neighborResponses Optional output values for corresponding neighbors. It is a
    single-precision floating-point matrix of \<number_of_samples\> \* k size.
    @param dist Optional output distances from the input vectors to the corresponding neighbors. It is
    a single-precision floating-point matrix of \<number_of_samples\> \* k size.

    For each input vector (a row of the matrix samples), the method finds the k nearest neighbors. In
    case of regression, the predicted result is a mean value of the particular vector's neighbor
    responses. In case of classification, the class is determined by voting.

    For each input vector, the neighbors are sorted by their distances to the vector.

    In case of C++ interface you can use output pointers to empty matrices and the function will
    allocate memory itself.

    If only a single input vector is passed, all output matrices are optional and the predicted value is
    returned by the method.

    The function is parallelized with the TBB library.
     */
    virtual float findNearest( InputArray samples, int k,
                               OutputArray results,
                               OutputArray neighborResponses=noArray(),
                               OutputArray dist=noArray() ) const = 0;

    enum { BRUTE_FORCE=1, KDTREE=2 };

    /** @brief Creates the empty model

    @param params The model parameters: default number of neighbors to use in predict method (in
    KNearest::findNearest this number must be passed explicitly) and the flag on whether
    classification or regression model should be trained.

    The static method creates empty KNearest classifier. It should be then trained using train method
    (see StatModel::train). Alternatively, you can load boost model from file using
    StatModel::load\<KNearest\>(filename).
     */
    static Ptr<KNearest> create(const Params& params=Params());
};

//! @} ml_knearest

/****************************************************************************************\
*                                   Support Vector Machines                              *
\****************************************************************************************/

//! @addtogroup ml_svm
//! @{

/** @brief Support Vector Machines.

@note
   -   (Python) An example of digit recognition using SVM can be found at
        opencv_source/samples/python2/digits.py
    -   (Python) An example of grid search digit recognition using SVM can be found at
        opencv_source/samples/python2/digits_adjust.py
    -   (Python) An example of video digit recognition using SVM can be found at
        opencv_source/samples/python2/digits_video.py
 */
class CV_EXPORTS_W SVM : public StatModel
{
public:
    /** @brief SVM training parameters.

    The structure must be initialized and passed to the training method of SVM.
     */
    class CV_EXPORTS_W_MAP Params
    {
    public:
        Params();
        /** @brief The constructors

        @param svm_type Type of a SVM formulation. Possible values are:
        -   **SVM::C_SVC** C-Support Vector Classification. n-class classification (n \f$\geq\f$ 2), allows
        imperfect separation of classes with penalty multiplier C for outliers.
        -   **SVM::NU_SVC** \f$\nu\f$-Support Vector Classification. n-class classification with possible
        imperfect separation. Parameter \f$\nu\f$ (in the range 0..1, the larger the value, the smoother
        the decision boundary) is used instead of C.
        -   **SVM::ONE_CLASS** Distribution Estimation (One-class SVM). All the training data are from
        the same class, SVM builds a boundary that separates the class from the rest of the feature
        space.
        -   **SVM::EPS_SVR** \f$\epsilon\f$-Support Vector Regression. The distance between feature vectors
        from the training set and the fitting hyper-plane must be less than p. For outliers the
        penalty multiplier C is used.
        -   **SVM::NU_SVR** \f$\nu\f$-Support Vector Regression. \f$\nu\f$ is used instead of p.
        See @cite LibSVM for details.
        @param kernel_type Type of a SVM kernel. Possible values are:
        -   **SVM::LINEAR** Linear kernel. No mapping is done, linear discrimination (or regression) is
        done in the original feature space. It is the fastest option. \f$K(x_i, x_j) = x_i^T x_j\f$.
        -   **SVM::POLY** Polynomial kernel:
        \f$K(x_i, x_j) = (\gamma x_i^T x_j + coef0)^{degree}, \gamma > 0\f$.
        -   **SVM::RBF** Radial basis function (RBF), a good choice in most cases.
        \f$K(x_i, x_j) = e^{-\gamma ||x_i - x_j||^2}, \gamma > 0\f$.
        -   **SVM::SIGMOID** Sigmoid kernel: \f$K(x_i, x_j) = \tanh(\gamma x_i^T x_j + coef0)\f$.
        -   **SVM::CHI2** Exponential Chi2 kernel, similar to the RBF kernel:
        \f$K(x_i, x_j) = e^{-\gamma \chi^2(x_i,x_j)}, \chi^2(x_i,x_j) = (x_i-x_j)^2/(x_i+x_j), \gamma > 0\f$.
        -   **SVM::INTER** Histogram intersection kernel. A fast kernel. \f$K(x_i, x_j) = min(x_i,x_j)\f$.
        @param degree Parameter degree of a kernel function (POLY).
        @param gamma Parameter \f$\gamma\f$ of a kernel function (POLY / RBF / SIGMOID / CHI2).
        @param coef0 Parameter coef0 of a kernel function (POLY / SIGMOID).
        @param Cvalue Parameter C of a SVM optimization problem (C_SVC / EPS_SVR / NU_SVR).
        @param nu Parameter \f$\nu\f$ of a SVM optimization problem (NU_SVC / ONE_CLASS / NU_SVR).
        @param p Parameter \f$\epsilon\f$ of a SVM optimization problem (EPS_SVR).
        @param classWeights Optional weights in the C_SVC problem , assigned to particular classes. They
        are multiplied by C so the parameter C of class \#i becomes classWeights(i) \* C. Thus these
        weights affect the misclassification penalty for different classes. The larger weight, the larger
        penalty on misclassification of data from the corresponding class.
        @param termCrit Termination criteria of the iterative SVM training procedure which solves a
        partial case of constrained quadratic optimization problem. You can specify tolerance and/or the
        maximum number of iterations.

        The default constructor initialize the structure with following values:
        @code
            SVMParams::SVMParams() :
                svmType(SVM::C_SVC), kernelType(SVM::RBF), degree(0),
                gamma(1), coef0(0), C(1), nu(0), p(0), classWeights(0)
            {
                termCrit = TermCriteria( TermCriteria::MAX_ITER+TermCriteria::EPS, 1000, FLT_EPSILON );
            }
        @endcode
        A comparison of different kernels on the following 2D test case with four classes. Four C_SVC SVMs
        have been trained (one against rest) with auto_train. Evaluation on three different kernels (CHI2,
        INTER, RBF). The color depicts the class with max score. Bright means max-score \> 0, dark means
        max-score \< 0.

        ![image](pics/SVM_Comparison.png)
         */
        Params( int svm_type, int kernel_type,
                double degree, double gamma, double coef0,
                double Cvalue, double nu, double p,
                const Mat& classWeights, TermCriteria termCrit );

        CV_PROP_RW int         svmType;
        CV_PROP_RW int         kernelType;
        CV_PROP_RW double      gamma, coef0, degree;

        CV_PROP_RW double      C;  // for CV_SVM_C_SVC, CV_SVM_EPS_SVR and CV_SVM_NU_SVR
        CV_PROP_RW double      nu; // for CV_SVM_NU_SVC, CV_SVM_ONE_CLASS, and CV_SVM_NU_SVR
        CV_PROP_RW double      p; // for CV_SVM_EPS_SVR
        CV_PROP_RW Mat         classWeights; // for CV_SVM_C_SVC
        CV_PROP_RW TermCriteria termCrit; // termination criteria
    };

    class CV_EXPORTS Kernel : public Algorithm
    {
    public:
        virtual int getType() const = 0;
        virtual void calc( int vcount, int n, const float* vecs, const float* another, float* results ) = 0;
    };

    // SVM type
    enum { C_SVC=100, NU_SVC=101, ONE_CLASS=102, EPS_SVR=103, NU_SVR=104 };

    // SVM kernel type
    enum { CUSTOM=-1, LINEAR=0, POLY=1, RBF=2, SIGMOID=3, CHI2=4, INTER=5 };

    // SVM params type
    enum { C=0, GAMMA=1, P=2, NU=3, COEF=4, DEGREE=5 };

    /** @brief Trains an SVM with optimal parameters.

    @param data the training data that can be constructed using TrainData::create or
    TrainData::loadFromCSV.
    @param kFold Cross-validation parameter. The training set is divided into kFold subsets. One
    subset is used to test the model, the others form the train set. So, the SVM algorithm is executed
    kFold times.
    @param Cgrid
    @param gammaGrid
    @param pGrid
    @param nuGrid
    @param coeffGrid
    @param degreeGrid Iteration grid for the corresponding SVM parameter.
    @param balanced If true and the problem is 2-class classification then the method creates more
    balanced cross-validation subsets that is proportions between classes in subsets are close to such
    proportion in the whole train dataset.

    The method trains the SVM model automatically by choosing the optimal parameters C, gamma, p, nu,
    coef0, degree from SVM::Params. Parameters are considered optimal when the cross-validation estimate
    of the test set error is minimal.

    If there is no need to optimize a parameter, the corresponding grid step should be set to any value
    less than or equal to 1. For example, to avoid optimization in gamma, set gammaGrid.step = 0,
    gammaGrid.minVal, gamma_grid.maxVal as arbitrary numbers. In this case, the value params.gamma is
    taken for gamma.

    And, finally, if the optimization in a parameter is required but the corresponding grid is unknown,
    you may call the function SVM::getDefaulltGrid. To generate a grid, for example, for gamma, call
    SVM::getDefaulltGrid(SVM::GAMMA).

    This function works for the classification (params.svmType=SVM::C_SVC or
    params.svmType=SVM::NU_SVC) as well as for the regression (params.svmType=SVM::EPS_SVR or
    params.svmType=SVM::NU_SVR). If params.svmType=SVM::ONE_CLASS, no optimization is made and the
    usual SVM with parameters specified in params is executed.
     */
    virtual bool trainAuto( const Ptr<TrainData>& data, int kFold = 10,
                    ParamGrid Cgrid = SVM::getDefaultGrid(SVM::C),
                    ParamGrid gammaGrid  = SVM::getDefaultGrid(SVM::GAMMA),
                    ParamGrid pGrid      = SVM::getDefaultGrid(SVM::P),
                    ParamGrid nuGrid     = SVM::getDefaultGrid(SVM::NU),
                    ParamGrid coeffGrid  = SVM::getDefaultGrid(SVM::COEF),
                    ParamGrid degreeGrid = SVM::getDefaultGrid(SVM::DEGREE),
                    bool balanced=false) = 0;

    /** @brief Retrieves all the support vectors

    The method returns all the support vector as floating-point matrix, where support vectors are stored
    as matrix rows.
     */
    CV_WRAP virtual Mat getSupportVectors() const = 0;

    virtual void setParams(const Params& p, const Ptr<Kernel>& customKernel=Ptr<Kernel>()) = 0;

    /** @brief Returns the current SVM parameters.

    This function may be used to get the optimal parameters obtained while automatically training
    SVM::trainAuto.
     */
    virtual Params getParams() const = 0;
    virtual Ptr<Kernel> getKernel() const = 0;

    /** @brief Retrieves the decision function

    @param i the index of the decision function. If the problem solved is regression, 1-class or
    2-class classification, then there will be just one decision function and the index should always
    be 0. Otherwise, in the case of N-class classification, there will be N\*(N-1)/2 decision
    functions.
    @param alpha the optional output vector for weights, corresponding to different support vectors.
    In the case of linear SVM all the alpha's will be 1's.
    @param svidx the optional output vector of indices of support vectors within the matrix of support
    vectors (which can be retrieved by SVM::getSupportVectors). In the case of linear SVM each
    decision function consists of a single "compressed" support vector.

    The method returns rho parameter of the decision function, a scalar subtracted from the weighted sum
    of kernel responses.
     */
    virtual double getDecisionFunction(int i, OutputArray alpha, OutputArray svidx) const = 0;

    /** @brief Generates a grid for SVM parameters.

    @param param_id SVM parameters IDs that must be one of the following:
    -   **SVM::C**
    -   **SVM::GAMMA**
    -   **SVM::P**
    -   **SVM::NU**
    -   **SVM::COEF**
    -   **SVM::DEGREE**
    The grid is generated for the parameter with this ID.

    The function generates a grid for the specified parameter of the SVM algorithm. The grid may be
    passed to the function SVM::trainAuto.
     */
    static ParamGrid getDefaultGrid( int param_id );

    /** @brief Creates empty model

    @param p SVM parameters
    @param customKernel the optional custom kernel to use. It must implement SVM::Kernel interface.

    Use StatModel::train to train the model, StatModel::train\<RTrees\>(traindata, params) to create and
    train the model, StatModel::load\<RTrees\>(filename) to load the pre-trained model. Since SVM has
    several parameters, you may want to find the best parameters for your problem. It can be done with
    SVM::trainAuto.
     */
    static Ptr<SVM> create(const Params& p=Params(), const Ptr<Kernel>& customKernel=Ptr<Kernel>());
};

//! @} ml_svm

/****************************************************************************************\
*                              Expectation - Maximization                                *
\****************************************************************************************/

//! @addtogroup ml_em
//! @{

/** @brief The class implements the EM algorithm as described in the beginning of this section.
 */
class CV_EXPORTS_W EM : public StatModel
{
public:
    // Type of covariation matrices
    enum {COV_MAT_SPHERICAL=0, COV_MAT_DIAGONAL=1, COV_MAT_GENERIC=2, COV_MAT_DEFAULT=COV_MAT_DIAGONAL};

    // Default parameters
    enum {DEFAULT_NCLUSTERS=5, DEFAULT_MAX_ITERS=100};

    // The initial step
    enum {START_E_STEP=1, START_M_STEP=2, START_AUTO_STEP=0};

    /** @brief The class describes EM training parameters.
    */
    class CV_EXPORTS_W_MAP Params
    {
    public:
        /** @brief The constructor

        @param nclusters The number of mixture components in the Gaussian mixture model. Default value of
        the parameter is EM::DEFAULT_NCLUSTERS=5. Some of EM implementation could determine the optimal
        number of mixtures within a specified value range, but that is not the case in ML yet.
        @param covMatType Constraint on covariance matrices which defines type of matrices. Possible
        values are:
        -   **EM::COV_MAT_SPHERICAL** A scaled identity matrix \f$\mu_k * I\f$. There is the only
        parameter \f$\mu_k\f$ to be estimated for each matrix. The option may be used in special cases,
        when the constraint is relevant, or as a first step in the optimization (for example in case
        when the data is preprocessed with PCA). The results of such preliminary estimation may be
        passed again to the optimization procedure, this time with
        covMatType=EM::COV_MAT_DIAGONAL.
        -   **EM::COV_MAT_DIAGONAL** A diagonal matrix with positive diagonal elements. The number of
        free parameters is d for each matrix. This is most commonly used option yielding good
        estimation results.
        -   **EM::COV_MAT_GENERIC** A symmetric positively defined matrix. The number of free
        parameters in each matrix is about \f$d^2/2\f$. It is not recommended to use this option, unless
        there is pretty accurate initial estimation of the parameters and/or a huge number of
        training samples.
        @param termCrit The termination criteria of the EM algorithm. The EM algorithm can be terminated
        by the number of iterations termCrit.maxCount (number of M-steps) or when relative change of
        likelihood logarithm is less than termCrit.epsilon. Default maximum number of iterations is
        EM::DEFAULT_MAX_ITERS=100.
         */
        explicit Params(int nclusters=DEFAULT_NCLUSTERS, int covMatType=EM::COV_MAT_DIAGONAL,
                        const TermCriteria& termCrit=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,
                                                                  EM::DEFAULT_MAX_ITERS, 1e-6));
        CV_PROP_RW int nclusters;
        CV_PROP_RW int covMatType;
        CV_PROP_RW TermCriteria termCrit;
    };

    virtual void setParams(const Params& p) = 0;
    virtual Params getParams() const = 0;
    /** @brief Returns weights of the mixtures

    Returns vector with the number of elements equal to the number of mixtures.
     */
    virtual Mat getWeights() const = 0;
    /** @brief Returns the cluster centers (means of the Gaussian mixture)

    Returns matrix with the number of rows equal to the number of mixtures and number of columns equal
    to the space dimensionality.
     */
    virtual Mat getMeans() const = 0;
    /** @brief Returns covariation matrices

    Returns vector of covariation matrices. Number of matrices is the number of gaussian mixtures, each
    matrix is a square floating-point matrix NxN, where N is the space dimensionality.
     */
    virtual void getCovs(std::vector<Mat>& covs) const = 0;

    /** @brief Returns a likelihood logarithm value and an index of the most probable mixture component for the
    given sample.

    @param sample A sample for classification. It should be a one-channel matrix of \f$1 \times dims\f$ or
    \f$dims \times 1\f$ size.
    @param probs Optional output matrix that contains posterior probabilities of each component given
    the sample. It has \f$1 \times nclusters\f$ size and CV_64FC1 type.

    The method returns a two-element double vector. Zero element is a likelihood logarithm value for the
    sample. First element is an index of the most probable mixture component for the given sample.
     */
    CV_WRAP virtual Vec2d predict2(InputArray sample, OutputArray probs) const = 0;

    virtual bool train( const Ptr<TrainData>& trainData, int flags=0 ) = 0;

    /** @brief Static methods that estimate the Gaussian mixture parameters from a samples set

    @param samples Samples from which the Gaussian mixture model will be estimated. It should be a
    one-channel matrix, each row of which is a sample. If the matrix does not have CV_64F type it
    will be converted to the inner matrix of such type for the further computing.
    @param logLikelihoods The optional output matrix that contains a likelihood logarithm value for
    each sample. It has \f$nsamples \times 1\f$ size and CV_64FC1 type.
    @param labels The optional output "class label" for each sample:
    \f$\texttt{labels}_i=\texttt{arg max}_k(p_{i,k}), i=1..N\f$ (indices of the most probable mixture
    component for each sample). It has \f$nsamples \times 1\f$ size and CV_32SC1 type.
    @param probs The optional output matrix that contains posterior probabilities of each Gaussian
    mixture component given the each sample. It has \f$nsamples \times nclusters\f$ size and CV_64FC1
    type.
    @param params The Gaussian mixture params, see EM::Params description
    @return true if the Gaussian mixture model was trained successfully, otherwise it returns
    false.

    Starts with Expectation step. Initial values of the model parameters will be estimated by the
    k-means algorithm.

    Unlike many of the ML models, EM is an unsupervised learning algorithm and it does not take
    responses (class labels or function values) as input. Instead, it computes the *Maximum Likelihood
    Estimate* of the Gaussian mixture parameters from an input sample set, stores all the parameters
    inside the structure: \f$p_{i,k}\f$ in probs, \f$a_k\f$ in means , \f$S_k\f$ in covs[k], \f$\pi_k\f$ in weights ,
    and optionally computes the output "class label" for each sample:
    \f$\texttt{labels}_i=\texttt{arg max}_k(p_{i,k}), i=1..N\f$ (indices of the most probable mixture
    component for each sample).

    The trained model can be used further for prediction, just like any other classifier. The trained
    model is similar to the NormalBayesClassifier.
     */
    static Ptr<EM> train(InputArray samples,
                          OutputArray logLikelihoods=noArray(),
                          OutputArray labels=noArray(),
                          OutputArray probs=noArray(),
                          const Params& params=Params());

    /** Starts with Expectation step. You need to provide initial means \f$a_k\f$ of mixture
    components. Optionally you can pass initial weights \f$\pi_k\f$ and covariance matrices
    \f$S_k\f$ of mixture components.

    @param samples Samples from which the Gaussian mixture model will be estimated. It should be a
    one-channel matrix, each row of which is a sample. If the matrix does not have CV_64F type it
    will be converted to the inner matrix of such type for the further computing.
    @param means0 Initial means \f$a_k\f$ of mixture components. It is a one-channel matrix of
    \f$nclusters \times dims\f$ size. If the matrix does not have CV_64F type it will be converted to the
    inner matrix of such type for the further computing.
    @param covs0 The vector of initial covariance matrices \f$S_k\f$ of mixture components. Each of
    covariance matrices is a one-channel matrix of \f$dims \times dims\f$ size. If the matrices do not
    have CV_64F type they will be converted to the inner matrices of such type for the further
    computing.
    @param weights0 Initial weights \f$\pi_k\f$ of mixture components. It should be a one-channel
    floating-point matrix with \f$1 \times nclusters\f$ or \f$nclusters \times 1\f$ size.
    @param logLikelihoods The optional output matrix that contains a likelihood logarithm value for
    each sample. It has \f$nsamples \times 1\f$ size and CV_64FC1 type.
    @param labels The optional output "class label" for each sample:
    \f$\texttt{labels}_i=\texttt{arg max}_k(p_{i,k}), i=1..N\f$ (indices of the most probable mixture
    component for each sample). It has \f$nsamples \times 1\f$ size and CV_32SC1 type.
    @param probs The optional output matrix that contains posterior probabilities of each Gaussian
    mixture component given the each sample. It has \f$nsamples \times nclusters\f$ size and CV_64FC1
    type.
    @param params The Gaussian mixture params, see EM::Params description
    */
    static Ptr<EM> train_startWithE(InputArray samples, InputArray means0,
                                     InputArray covs0=noArray(),
                                     InputArray weights0=noArray(),
                                     OutputArray logLikelihoods=noArray(),
                                     OutputArray labels=noArray(),
                                     OutputArray probs=noArray(),
                                     const Params& params=Params());

    /** Starts with Maximization step. You need to provide initial probabilities \f$p_{i,k}\f$ to
    use this option.

    @param samples Samples from which the Gaussian mixture model will be estimated. It should be a
    one-channel matrix, each row of which is a sample. If the matrix does not have CV_64F type it
    will be converted to the inner matrix of such type for the further computing.
    @param probs0
    @param logLikelihoods The optional output matrix that contains a likelihood logarithm value for
    each sample. It has \f$nsamples \times 1\f$ size and CV_64FC1 type.
    @param labels The optional output "class label" for each sample:
    \f$\texttt{labels}_i=\texttt{arg max}_k(p_{i,k}), i=1..N\f$ (indices of the most probable mixture
    component for each sample). It has \f$nsamples \times 1\f$ size and CV_32SC1 type.
    @param probs The optional output matrix that contains posterior probabilities of each Gaussian
    mixture component given the each sample. It has \f$nsamples \times nclusters\f$ size and CV_64FC1
    type.
    @param params The Gaussian mixture params, see EM::Params description
    */
    static Ptr<EM> train_startWithM(InputArray samples, InputArray probs0,
                                     OutputArray logLikelihoods=noArray(),
                                     OutputArray labels=noArray(),
                                     OutputArray probs=noArray(),
                                     const Params& params=Params());

    /** @brief Creates empty EM model

    @param params EM parameters

    The model should be trained then using StatModel::train(traindata, flags) method. Alternatively, you
    can use one of the EM::train\* methods or load it from file using StatModel::load\<EM\>(filename).
     */
    static Ptr<EM> create(const Params& params=Params());
};

//! @} ml_em

/****************************************************************************************\
*                                      Decision Tree                                     *
\****************************************************************************************/

//! @addtogroup ml_decsiontrees
//! @{

/** @brief The class represents a single decision tree or a collection of decision trees. The current public
interface of the class allows user to train only a single decision tree, however the class is
capable of storing multiple decision trees and using them for prediction (by summing responses or
using a voting schemes), and the derived from DTrees classes (such as RTrees and Boost) use this
capability to implement decision tree ensembles.
 */
class CV_EXPORTS_W DTrees : public StatModel
{
public:
    enum { PREDICT_AUTO=0, PREDICT_SUM=(1<<8), PREDICT_MAX_VOTE=(2<<8), PREDICT_MASK=(3<<8) };

    /** @brief The structure contains all the decision tree training parameters. You can initialize it by default
    constructor and then override any parameters directly before training, or the structure may be fully
    initialized using the advanced variant of the constructor.
     */
    class CV_EXPORTS_W_MAP Params
    {
    public:
        Params();
        /** @brief The constructors

        @param maxDepth The maximum possible depth of the tree. That is the training algorithms attempts
        to split a node while its depth is less than maxDepth. The root node has zero depth. The actual
        depth may be smaller if the other termination criteria are met (see the outline of the training
        procedure in the beginning of the section), and/or if the tree is pruned.
        @param minSampleCount If the number of samples in a node is less than this parameter then the node
        will not be split.
        @param regressionAccuracy Termination criteria for regression trees. If all absolute differences
        between an estimated value in a node and values of train samples in this node are less than this
        parameter then the node will not be split further.
        @param useSurrogates If true then surrogate splits will be built. These splits allow to work with
        missing data and compute variable importance correctly.

        @note currently it's not implemented.

        @param maxCategories Cluster possible values of a categorical variable into K\<=maxCategories
        clusters to find a suboptimal split. If a discrete variable, on which the training procedure
        tries to make a split, takes more than maxCategories values, the precise best subset estimation
        may take a very long time because the algorithm is exponential. Instead, many decision trees
        engines (including our implementation) try to find sub-optimal split in this case by clustering
        all the samples into maxCategories clusters that is some categories are merged together. The
        clustering is applied only in n \> 2-class classification problems for categorical variables
        with N \> max_categories possible values. In case of regression and 2-class classification the
        optimal split can be found efficiently without employing clustering, thus the parameter is not
        used in these cases.

        @param CVFolds If CVFolds \> 1 then algorithms prunes the built decision tree using K-fold
        cross-validation procedure where K is equal to CVFolds.

        @param use1SERule If true then a pruning will be harsher. This will make a tree more compact and
        more resistant to the training data noise but a bit less accurate.

        @param truncatePrunedTree If true then pruned branches are physically removed from the tree.
        Otherwise they are retained and it is possible to get results from the original unpruned (or
        pruned less aggressively) tree.

        @param priors The array of a priori class probabilities, sorted by the class label value. The
        parameter can be used to tune the decision tree preferences toward a certain class. For example,
        if you want to detect some rare anomaly occurrence, the training base will likely contain much
        more normal cases than anomalies, so a very good classification performance will be achieved
        just by considering every case as normal. To avoid this, the priors can be specified, where the
        anomaly probability is artificially increased (up to 0.5 or even greater), so the weight of the
        misclassified anomalies becomes much bigger, and the tree is adjusted properly. You can also
        think about this parameter as weights of prediction categories which determine relative weights
        that you give to misclassification. That is, if the weight of the first category is 1 and the
        weight of the second category is 10, then each mistake in predicting the second category is
        equivalent to making 10 mistakes in predicting the first category.

        The default constructor initializes all the parameters with the default values tuned for the
        standalone classification tree:
        @code
            DTrees::Params::Params()
            {
                maxDepth = INT_MAX;
                minSampleCount = 10;
                regressionAccuracy = 0.01f;
                useSurrogates = false;
                maxCategories = 10;
                CVFolds = 10;
                use1SERule = true;
                truncatePrunedTree = true;
                priors = Mat();
            }
        @endcode
         */
        Params( int maxDepth, int minSampleCount,
               double regressionAccuracy, bool useSurrogates,
               int maxCategories, int CVFolds,
               bool use1SERule, bool truncatePrunedTree,
               const Mat& priors );

        CV_PROP_RW int   maxCategories;
        CV_PROP_RW int   maxDepth;
        CV_PROP_RW int   minSampleCount;
        CV_PROP_RW int   CVFolds;
        CV_PROP_RW bool  useSurrogates;
        CV_PROP_RW bool  use1SERule;
        CV_PROP_RW bool  truncatePrunedTree;
        CV_PROP_RW float regressionAccuracy;
        CV_PROP_RW Mat priors;
    };

    /** @brief The class represents a decision tree node. It has public members:
    -   member double value
    Value at the node: a class label in case of classification or estimated function value in case
    of regression.
    -   member int classIdx
    Class index normalized to 0..class_count-1 range and assigned to the node. It is used
    internally in classification trees and tree ensembles.
    -   member int parent
    Index of the parent node
    -   member int left
    Index of the left child node
    -   member int right
    Index of right child node.
    -   member int defaultDir
    Default direction where to go (-1: left or +1: right). It helps in the case of missing values.
    -   member int split
    Index of the first split
     */
    class CV_EXPORTS Node
    {
    public:
        Node();
        double value;
        int classIdx;

        int parent;
        int left;
        int right;
        int defaultDir;

        int split;
    };

    /** @brief The class represents split in a decision tree. It has public members:
    -   member int varIdx
    Index of variable on which the split is created.
    -   member bool inversed
    If true, then the inverse split rule is used (i.e. left and right branches are exchanged in
    the rule expressions below).
    -   member float quality
    The split quality, a positive number. It is used to choose the best split.
    -   member int next
    Index of the next split in the list of splits for the node
    -   member float c
    The threshold value in case of split on an ordered variable. The rule is: :
    if var_value < c
    then next_node<-left
    else next_node<-right
    -   member int subsetOfs
    Offset of the bitset used by the split on a categorical variable. The rule is: :
    if bitset[var_value] == 1
    then next_node <- left
    else next_node <- right
     */
    class CV_EXPORTS Split
    {
    public:
        Split();
        int varIdx;
        bool inversed;
        float quality;
        int next;
        float c;
        int subsetOfs;
    };

    /** @brief Sets the training parameters

    @param p Training parameters of type DTrees::Params.

    The method sets the training parameters.
     */
    virtual void setDParams(const Params& p);
    /** @brief Returns the training parameters

    The method returns the training parameters.
     */
    virtual Params getDParams() const;

    /** @brief Returns indices of root nodes
    */
    virtual const std::vector<int>& getRoots() const = 0;
    /** @brief Returns all the nodes

    all the node indices, mentioned above (left, right, parent, root indices) are indices in the
    returned vector
     */
    virtual const std::vector<Node>& getNodes() const = 0;
    /** @brief Returns all the splits

    all the split indices, mentioned above (split, next etc.) are indices in the returned vector
     */
    virtual const std::vector<Split>& getSplits() const = 0;
    /** @brief Returns all the bitsets for categorical splits

    Split::subsetOfs is an offset in the returned vector
     */
    virtual const std::vector<int>& getSubsets() const = 0;

    /** @brief Creates the empty model

    The static method creates empty decision tree with the specified parameters. It should be then
    trained using train method (see StatModel::train). Alternatively, you can load the model from file
    using StatModel::load\<DTrees\>(filename).
     */
    static Ptr<DTrees> create(const Params& params=Params());
};

//! @} ml_decsiontrees

/****************************************************************************************\
*                                   Random Trees Classifier                              *
\****************************************************************************************/

//! @addtogroup ml_randomtrees
//! @{

/** @brief The class implements the random forest predictor as described in the beginning of this section.
 */
class CV_EXPORTS_W RTrees : public DTrees
{
public:
    /** @brief The set of training parameters for the forest is a superset of the training
    parameters for a single tree.

    However, random trees do not need all the functionality/features of decision trees. Most
    noticeably, the trees are not pruned, so the cross-validation parameters are not used.
     */
    class CV_EXPORTS_W_MAP Params : public DTrees::Params
    {
    public:
        Params();
        /** @brief The constructors

        @param maxDepth the depth of the tree. A low value will likely underfit and conversely a high
        value will likely overfit. The optimal value can be obtained using cross validation or other
        suitable methods.
        @param minSampleCount minimum samples required at a leaf node for it to be split. A reasonable
        value is a small percentage of the total data e.g. 1%.
        @param regressionAccuracy
        @param useSurrogates
        @param maxCategories Cluster possible values of a categorical variable into K \<= maxCategories
        clusters to find a suboptimal split. If a discrete variable, on which the training procedure tries
        to make a split, takes more than max_categories values, the precise best subset estimation may
        take a very long time because the algorithm is exponential. Instead, many decision trees engines
        (including ML) try to find sub-optimal split in this case by clustering all the samples into
        maxCategories clusters that is some categories are merged together. The clustering is applied only
        in n\>2-class classification problems for categorical variables with N \> max_categories possible
        values. In case of regression and 2-class classification the optimal split can be found
        efficiently without employing clustering, thus the parameter is not used in these cases.
        @param priors
        @param calcVarImportance If true then variable importance will be calculated and then it can be
        retrieved by RTrees::getVarImportance.
        @param nactiveVars The size of the randomly selected subset of features at each tree node and that
        are used to find the best split(s). If you set it to 0 then the size will be set to the square
        root of the total number of features.
        @param termCrit The termination criteria that specifies when the training algorithm stops - either
        when the specified number of trees is trained and added to the ensemble or when sufficient
        accuracy (measured as OOB error) is achieved. Typically the more trees you have the better the
        accuracy. However, the improvement in accuracy generally diminishes and asymptotes pass a certain
        number of trees. Also to keep in mind, the number of tree increases the prediction time linearly.

        The default constructor sets all parameters to default values which are different from default
        values of `DTrees::Params`:
        @code
            RTrees::Params::Params() : DTrees::Params( 5, 10, 0, false, 10, 0, false, false, Mat() ),
                calcVarImportance(false), nactiveVars(0)
            {
                termCrit = cvTermCriteria( TermCriteria::MAX_ITERS + TermCriteria::EPS, 50, 0.1 );
            }
        @endcode
         */
        Params( int maxDepth, int minSampleCount,
                double regressionAccuracy, bool useSurrogates,
                int maxCategories, const Mat& priors,
                bool calcVarImportance, int nactiveVars,
                TermCriteria termCrit );

        CV_PROP_RW bool calcVarImportance; // true <=> RF processes variable importance
        CV_PROP_RW int nactiveVars;
        CV_PROP_RW TermCriteria termCrit;
    };

    virtual void setRParams(const Params& p) = 0;
    virtual Params getRParams() const = 0;

    /** @brief Returns the variable importance array.

    The method returns the variable importance vector, computed at the training stage when
    RTParams::calcVarImportance is set to true. If this flag was set to false, the empty matrix is
    returned.
     */
    virtual Mat getVarImportance() const = 0;

    /** @brief Creates the empty model

    Use StatModel::train to train the model, StatModel::train to create and
    train the model, StatModel::load to load the pre-trained model.
     */
    static Ptr<RTrees> create(const Params& params=Params());
};

//! @} ml_randomtrees

/****************************************************************************************\
*                                   Boosted tree classifier                              *
\****************************************************************************************/

//! @addtogroup ml_boost
//! @{

/** @brief Boosted tree classifier derived from DTrees
 */
class CV_EXPORTS_W Boost : public DTrees
{
public:
    /** @brief The structure is derived from DTrees::Params but not all of the decision tree parameters are
    supported. In particular, cross-validation is not supported.

    All parameters are public. You can initialize them by a constructor and then override some of them
    directly if you want.
     */
    class CV_EXPORTS_W_MAP Params : public DTrees::Params
    {
    public:
        CV_PROP_RW int boostType;
        CV_PROP_RW int weakCount;
        CV_PROP_RW double weightTrimRate;

        Params();
        /** @brief The constructors.

        @param boostType Type of the boosting algorithm. Possible values are:
        -   **Boost::DISCRETE** Discrete AdaBoost.
        -   **Boost::REAL** Real AdaBoost. It is a technique that utilizes confidence-rated predictions
        and works well with categorical data.
        -   **Boost::LOGIT** LogitBoost. It can produce good regression fits.
        -   **Boost::GENTLE** Gentle AdaBoost. It puts less weight on outlier data points and for that
        reason is often good with regression data.
        Gentle AdaBoost and Real AdaBoost are often the preferable choices.
        @param weakCount The number of weak classifiers.
        @param weightTrimRate A threshold between 0 and 1 used to save computational time. Samples
        with summary weight \f$\leq 1 - weight_trim_rate\f$ do not participate in the *next* iteration of
        training. Set this parameter to 0 to turn off this functionality.
        @param maxDepth
        @param useSurrogates
        @param priors

        See DTrees::Params for description of other parameters.

        Default parameters are:
        @code
            Boost::Params::Params()
            {
                boostType = Boost::REAL;
                weakCount = 100;
                weightTrimRate = 0.95;
                CVFolds = 0;
                maxDepth = 1;
            }
        @endcode
         */
        Params( int boostType, int weakCount, double weightTrimRate,
                int maxDepth, bool useSurrogates, const Mat& priors );
    };

    // Boosting type
    enum { DISCRETE=0, REAL=1, LOGIT=2, GENTLE=3 };

    /** @brief Returns the boosting parameters

    The method returns the training parameters.
     */
    virtual Params getBParams() const = 0;
    /** @brief Sets the boosting parameters

    @param p Training parameters of type Boost::Params.

    The method sets the training parameters.
     */
    virtual void setBParams(const Params& p) = 0;

    /** @brief Creates the empty model

    Use StatModel::train to train the model, StatModel::train\<Boost\>(traindata, params) to create and
    train the model, StatModel::load\<Boost\>(filename) to load the pre-trained model.
     */
    static Ptr<Boost> create(const Params& params=Params());
};

//! @} ml_boost

/****************************************************************************************\
*                                   Gradient Boosted Trees                               *
\****************************************************************************************/

/*class CV_EXPORTS_W GBTrees : public DTrees
{
public:
    struct CV_EXPORTS_W_MAP Params : public DTrees::Params
    {
        CV_PROP_RW int weakCount;
        CV_PROP_RW int lossFunctionType;
        CV_PROP_RW float subsamplePortion;
        CV_PROP_RW float shrinkage;

        Params();
        Params( int lossFunctionType, int weakCount, float shrinkage,
                float subsamplePortion, int maxDepth, bool useSurrogates );
    };

    enum {SQUARED_LOSS=0, ABSOLUTE_LOSS, HUBER_LOSS=3, DEVIANCE_LOSS};

    virtual void setK(int k) = 0;

    virtual float predictSerial( InputArray samples,
                                 OutputArray weakResponses, int flags) const = 0;

    static Ptr<GBTrees> create(const Params& p);
};*/

/****************************************************************************************\
*                              Artificial Neural Networks (ANN)                          *
\****************************************************************************************/

/////////////////////////////////// Multi-Layer Perceptrons //////////////////////////////

//! @addtogroup ml_neural
//! @{

/** @brief MLP model.

Unlike many other models in ML that are constructed and trained at once, in the MLP model these
steps are separated. First, a network with the specified topology is created using the non-default
constructor or the method ANN_MLP::create. All the weights are set to zeros. Then, the network is
trained using a set of input and output vectors. The training procedure can be repeated more than
once, that is, the weights can be adjusted based on the new training data.
 */
class CV_EXPORTS_W ANN_MLP : public StatModel
{
public:
    /** @brief Parameters of the MLP and of the training algorithm.

    You can initialize the structure by a constructor or the individual parameters can be adjusted
    after the structure is created.
    The network structure:
    -   member Mat layerSizes
    The number of elements in each layer of network. The very first element specifies the number
    of elements in the input layer. The last element - number of elements in the output layer.
    -   member int activateFunc
    The activation function. Currently the only fully supported activation function is
    ANN_MLP::SIGMOID_SYM.
    -   member double fparam1
    The first parameter of activation function, 0 by default.
    -   member double fparam2
    The second parameter of the activation function, 0 by default.
    @note
       If you are using the default ANN_MLP::SIGMOID_SYM activation function with the default
        parameter values fparam1=0 and fparam2=0 then the function used is y = 1.7159\*tanh(2/3 \* x),
        so the output will range from [-1.7159, 1.7159], instead of [0,1].

    The back-propagation algorithm parameters:
    -   member double bpDWScale
    Strength of the weight gradient term. The recommended value is about 0.1.
    -   member double bpMomentScale
    Strength of the momentum term (the difference between weights on the 2 previous iterations).
    This parameter provides some inertia to smooth the random fluctuations of the weights. It
    can vary from 0 (the feature is disabled) to 1 and beyond. The value 0.1 or so is good
    enough
    The RPROP algorithm parameters (see @cite RPROP93 for details):
    -   member double prDW0
    Initial value \f$\Delta_0\f$ of update-values \f$\Delta_{ij}\f$.
    -   member double rpDWPlus
    Increase factor \f$\eta^+\f$. It must be \>1.
    -   member double rpDWMinus
    Decrease factor \f$\eta^-\f$. It must be \<1.
    -   member double rpDWMin
    Update-values lower limit \f$\Delta_{min}\f$. It must be positive.
    -   member double rpDWMax
    Update-values upper limit \f$\Delta_{max}\f$. It must be \>1.
     */
    struct CV_EXPORTS_W_MAP Params
    {
        Params();
        /** @brief Construct the parameter structure

        @param layerSizes Integer vector specifying the number of neurons in each layer including the
        input and output layers.
        @param activateFunc Parameter specifying the activation function for each neuron: one of
        ANN_MLP::IDENTITY, ANN_MLP::SIGMOID_SYM, and ANN_MLP::GAUSSIAN.
        @param fparam1 The first parameter of the activation function, \f$\alpha\f$. See the formulas in the
        introduction section.
        @param fparam2 The second parameter of the activation function, \f$\beta\f$. See the formulas in the
        introduction section.
        @param termCrit Termination criteria of the training algorithm. You can specify the maximum number
        of iterations (maxCount) and/or how much the error could change between the iterations to make the
        algorithm continue (epsilon).
        @param trainMethod Training method of the MLP. Possible values are:
        -   **ANN_MLP_TrainParams::BACKPROP** The back-propagation algorithm.
        -   **ANN_MLP_TrainParams::RPROP** The RPROP algorithm.
        @param param1 Parameter of the training method. It is rp_dw0 for RPROP and bp_dw_scale for
        BACKPROP.
        @param param2 Parameter of the training method. It is rp_dw_min for RPROP and bp_moment_scale
        for BACKPROP.

        By default the RPROP algorithm is used:
        @code
            ANN_MLP_TrainParams::ANN_MLP_TrainParams()
            {
                layerSizes = Mat();
                activateFun = SIGMOID_SYM;
                fparam1 = fparam2 = 0;
                term_crit = TermCriteria( TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 0.01 );
                train_method = RPROP;
                bpDWScale = bpMomentScale = 0.1;
                rpDW0 = 0.1; rpDWPlus = 1.2; rpDWMinus = 0.5;
                rpDWMin = FLT_EPSILON; rpDWMax = 50.;
            }
        @endcode
         */
        Params( const Mat& layerSizes, int activateFunc, double fparam1, double fparam2,
                TermCriteria termCrit, int trainMethod, double param1, double param2=0 );

        enum { BACKPROP=0, RPROP=1 };

        CV_PROP_RW Mat layerSizes;
        CV_PROP_RW int activateFunc;
        CV_PROP_RW double fparam1;
        CV_PROP_RW double fparam2;

        CV_PROP_RW TermCriteria termCrit;
        CV_PROP_RW int trainMethod;

        // backpropagation parameters
        CV_PROP_RW double bpDWScale, bpMomentScale;

        // rprop parameters
        CV_PROP_RW double rpDW0, rpDWPlus, rpDWMinus, rpDWMin, rpDWMax;
    };

    // possible activation functions
    enum { IDENTITY = 0, SIGMOID_SYM = 1, GAUSSIAN = 2 };

    // available training flags
    enum { UPDATE_WEIGHTS = 1, NO_INPUT_SCALE = 2, NO_OUTPUT_SCALE = 4 };

    virtual Mat getWeights(int layerIdx) const = 0;

    /** @brief Sets the new network parameters

    @param p The new parameters

    The existing network, if any, will be destroyed and new empty one will be created. It should be
    re-trained after that.
     */
    virtual void setParams(const Params& p) = 0;

    /** @brief Retrieves the current network parameters
    */
    virtual Params getParams() const = 0;

    /** @brief Creates empty model

    Use StatModel::train to train the model, StatModel::train\<ANN_MLP\>(traindata, params) to create
    and train the model, StatModel::load\<ANN_MLP\>(filename) to load the pre-trained model. Note that
    the train method has optional flags, and the following flags are handled by \`ANN_MLP\`:

    -   **UPDATE_WEIGHTS** Algorithm updates the network weights, rather than computes them from
    scratch. In the latter case the weights are initialized using the Nguyen-Widrow algorithm.
    -   **NO_INPUT_SCALE** Algorithm does not normalize the input vectors. If this flag is not set,
    the training algorithm normalizes each input feature independently, shifting its mean value to
    0 and making the standard deviation equal to 1. If the network is assumed to be updated
    frequently, the new training data could be much different from original one. In this case, you
    should take care of proper normalization.
    -   **NO_OUTPUT_SCALE** Algorithm does not normalize the output vectors. If the flag is not set,
    the training algorithm normalizes each output feature independently, by transforming it to the
    certain range depending on the used activation function.
     */
    static Ptr<ANN_MLP> create(const Params& params=Params());
};

//! @} ml_neural

/****************************************************************************************\
*                           Logistic Regression                                          *
\****************************************************************************************/

//! @addtogroup ml_lr
//! @{

/** @brief Implements Logistic Regression classifier.
 */
class CV_EXPORTS LogisticRegression : public StatModel
{
public:
    class CV_EXPORTS Params
    {
    public:
        /** @brief The constructors

        @param learning_rate Specifies the learning rate.
        @param iters Specifies the number of iterations.
        @param method Specifies the kind of training method used. It should be set to either
        LogisticRegression::BATCH or LogisticRegression::MINI_BATCH. If using
        LogisticRegression::MINI_BATCH, set LogisticRegression::Params.mini_batch_size to a positive
        integer.
        @param normalization Specifies the kind of regularization to be applied.
        LogisticRegression::REG_L1 or LogisticRegression::REG_L2 (L1 norm or L2 norm). To use this, set
        LogisticRegression::Params.regularized to a integer greater than zero.
        @param reg To enable or disable regularization. Set to positive integer (greater than zero) to
        enable and to 0 to disable.
        @param batch_size Specifies the number of training samples taken in each step of Mini-Batch
        Gradient Descent. Will only be used if using LogisticRegression::MINI_BATCH training algorithm.
        It has to take values less than the total number of training samples.

        By initializing this structure, one can set all the parameters required for Logistic Regression
        classifier.
         */
        Params(double learning_rate = 0.001,
               int iters = 1000,
               int method = LogisticRegression::BATCH,
               int normalization = LogisticRegression::REG_L2,
               int reg = 1,
               int batch_size = 1);
        double alpha;
        int num_iters;
        int norm;
        int regularized;
        int train_method;
        int mini_batch_size;
        TermCriteria term_crit;
    };

    enum { REG_L1 = 0, REG_L2 = 1};
    enum { BATCH = 0, MINI_BATCH = 1};

    /** @brief This function writes the trained LogisticRegression clasifier to disk.
    */
    virtual void write( FileStorage &fs ) const = 0;
    /** @brief This function reads the trained LogisticRegression clasifier from disk.
    */
    virtual void read( const FileNode &fn ) = 0;

    /** @brief Trains the Logistic Regression classifier and returns true if successful.

    @param trainData Instance of ml::TrainData class holding learning data.
    @param flags Not used.
     */
    virtual bool train( const Ptr<TrainData>& trainData, int flags=0 ) = 0;
    /** @brief Predicts responses for input samples and returns a float type.

    @param samples The input data for the prediction algorithm. Matrix [m x n], where each row
    contains variables (features) of one object being classified. Should have data type CV_32F.
    @param results Predicted labels as a column matrix of type CV_32S.
    @param flags Not used.
     */
    virtual float predict( InputArray samples, OutputArray results=noArray(), int flags=0 ) const = 0;
    virtual void clear() = 0;

    /** @brief This function returns the trained paramters arranged across rows.

    For a two class classifcation problem, it returns a row matrix.
    It returns learnt paramters of the Logistic Regression as a matrix of type CV_32F.
     */
    virtual Mat get_learnt_thetas() const = 0;

    /** @brief Creates empty model.

    @param params The training parameters for the classifier of type LogisticRegression::Params.

    Creates Logistic Regression model with parameters given.
     */
    static Ptr<LogisticRegression> create( const Params& params = Params() );
};

//! @} ml_lr

/****************************************************************************************\
*                           Auxilary functions declarations                              *
\****************************************************************************************/

/** Generates `sample` from multivariate normal distribution, where `mean` - is an
   average row vector, `cov` - symmetric covariation matrix */
CV_EXPORTS void randMVNormal( InputArray mean, InputArray cov, int nsamples, OutputArray samples);

/** Generates sample from gaussian mixture distribution */
CV_EXPORTS void randGaussMixture( InputArray means, InputArray covs, InputArray weights,
                                  int nsamples, OutputArray samples, OutputArray sampClasses );

/** creates test set */
CV_EXPORTS void createConcentricSpheresTestSet( int nsamples, int nfeatures, int nclasses,
                                                OutputArray samples, OutputArray responses);

//! @} ml

}
}

#endif // __cplusplus
#endif // __OPENCV_ML_HPP__

/* End of file. */
