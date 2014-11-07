Neural Networks
===============

.. highlight:: cpp

ML implements feed-forward artificial neural networks or, more particularly, multi-layer perceptrons (MLP), the most commonly used type of neural networks. MLP consists of the input layer, output layer, and one or more hidden layers. Each layer of MLP includes one or more neurons directionally linked with the neurons from the previous and the next layer. The example below represents a 3-layer perceptron with three inputs, two outputs, and the hidden layer including five neurons:

.. image:: pics/mlp.png

All the neurons in MLP are similar. Each of them has several input links (it takes the output values from several neurons in the previous layer as input) and several output links (it passes the response to several neurons in the next layer). The values retrieved from the previous layer are summed up with certain weights, individual for each neuron, plus the bias term. The sum is transformed using the activation function
:math:`f` that may be also different for different neurons.

.. image:: pics/neuron_model.png

In other words, given the outputs
:math:`x_j` of the layer
:math:`n` , the outputs
:math:`y_i` of the layer
:math:`n+1` are computed as:

.. math::

    u_i =  \sum _j (w^{n+1}_{i,j}*x_j) + w^{n+1}_{i,bias}

.. math::

    y_i = f(u_i)

Different activation functions may be used. ML implements three standard functions:

*
    Identity function ( ``ANN_MLP::IDENTITY``     ):
    :math:`f(x)=x`
*
    Symmetrical sigmoid ( ``ANN_MLP::SIGMOID_SYM``     ):
    :math:`f(x)=\beta*(1-e^{-\alpha x})/(1+e^{-\alpha x}`     ), which is the default choice for MLP. The standard sigmoid with
    :math:`\beta =1, \alpha =1`     is shown below:

    .. image:: pics/sigmoid_bipolar.png

*
    Gaussian function ( ``ANN_MLP::GAUSSIAN``     ):
    :math:`f(x)=\beta e^{-\alpha x*x}`     , which is not completely supported at the moment.

In ML, all the neurons have the same activation functions, with the same free parameters (
:math:`\alpha, \beta` ) that are specified by user and are not altered by the training algorithms.

So, the whole trained network works as follows:

#. Take the feature vector as input. The vector size is equal to the size of the input layer.

#. Pass values as input to the first hidden layer.

#. Compute outputs of the hidden layer using the weights and the activation functions.

#. Pass outputs further downstream until you compute the output layer.

So, to compute the network, you need to know all the
weights
:math:`w^{n+1)}_{i,j}` . The weights are computed by the training
algorithm. The algorithm takes a training set, multiple input vectors
with the corresponding output vectors, and iteratively adjusts the
weights to enable the network to give the desired response to the
provided input vectors.

The larger the network size (the number of hidden layers and their sizes) is,
the more the potential network flexibility is. The error on the
training set could be made arbitrarily small. But at the same time the
learned network also "learns" the noise present in the training set,
so the error on the test set usually starts increasing after the network
size reaches a limit. Besides, the larger networks are trained much
longer than the smaller ones, so it is reasonable to pre-process the data,
using
:ocv:funcx:`PCA::operator()` or similar technique, and train a smaller network
on only essential features.

Another MLP feature is an inability to handle categorical
data as is. However, there is a workaround. If a certain feature in the
input or output (in case of ``n`` -class classifier for
:math:`n>2` ) layer is categorical and can take
:math:`M>2` different values, it makes sense to represent it as a binary tuple of ``M`` elements, where the ``i`` -th element is 1 if and only if the
feature is equal to the ``i`` -th value out of ``M`` possible. It
increases the size of the input/output layer but speeds up the
training algorithm convergence and at the same time enables "fuzzy" values
of such variables, that is, a tuple of probabilities instead of a fixed value.

ML implements two algorithms for training MLP's. The first algorithm is a classical
random sequential back-propagation algorithm.
The second (default) one is a batch RPROP algorithm.

.. [BackPropWikipedia] http://en.wikipedia.org/wiki/Backpropagation. Wikipedia article about the back-propagation algorithm.

.. [LeCun98] Y. LeCun, L. Bottou, G.B. Orr and K.-R. Muller, *Efficient backprop*, in Neural Networks---Tricks of the Trade, Springer Lecture Notes in Computer Sciences 1524, pp.5-50, 1998.

.. [RPROP93] M. Riedmiller and H. Braun, *A Direct Adaptive Method for Faster Backpropagation Learning: The RPROP Algorithm*, Proc. ICNN, San Francisco (1993).


ANN_MLP::Params
---------------------
.. ocv:class:: ANN_MLP::Params

  Parameters of the MLP and of the training algorithm. You can initialize the structure by a constructor or the individual parameters can be adjusted after the structure is created.

  The network structure:

  .. ocv:member:: Mat layerSizes

     The number of elements in each layer of network. The very first element specifies the number of elements in the input layer. The last element - number of elements in the output layer.

  .. ocv:member:: int activateFunc

     The activation function. Currently the only fully supported activation function is ``ANN_MLP::SIGMOID_SYM``.

  .. ocv:member:: double fparam1

     The first parameter of activation function, 0 by default.

  .. ocv:member:: double fparam2

     The second parameter of the activation function, 0 by default.

     .. note::

         If you are using the default ``ANN_MLP::SIGMOID_SYM`` activation function with the default parameter values fparam1=0 and fparam2=0 then the function used is y = 1.7159*tanh(2/3 * x), so the output will range from [-1.7159, 1.7159], instead of [0,1].

  The back-propagation algorithm parameters:

  .. ocv:member:: double bpDWScale

     Strength of the weight gradient term. The recommended value is about 0.1.

  .. ocv:member:: double bpMomentScale

     Strength of the momentum term (the difference between weights on the 2 previous iterations). This parameter provides some inertia to smooth the random fluctuations of the weights. It can vary from 0 (the feature is disabled) to 1 and beyond. The value 0.1 or so is good enough

  The RPROP algorithm parameters (see [RPROP93]_ for details):

  .. ocv:member:: double prDW0

     Initial value :math:`\Delta_0` of update-values :math:`\Delta_{ij}`.

  .. ocv:member:: double rpDWPlus

     Increase factor :math:`\eta^+`. It must be >1.

  .. ocv:member:: double rpDWMinus

     Decrease factor :math:`\eta^-`. It must be <1.

  .. ocv:member:: double rpDWMin

     Update-values lower limit :math:`\Delta_{min}`. It must be positive.

  .. ocv:member:: double rpDWMax

     Update-values upper limit :math:`\Delta_{max}`. It must be >1.


ANN_MLP::Params::Params
--------------------------------------------
Construct the parameter structure

.. ocv:function:: ANN_MLP::Params()

.. ocv:function:: ANN_MLP::Params::Params( const Mat& layerSizes, int activateFunc, double fparam1, double fparam2, TermCriteria termCrit, int trainMethod, double param1, double param2=0 )

    :param layerSizes: Integer vector specifying the number of neurons in each layer including the input and output layers.

    :param activateFunc: Parameter specifying the activation function for each neuron: one of  ``ANN_MLP::IDENTITY``, ``ANN_MLP::SIGMOID_SYM``, and ``ANN_MLP::GAUSSIAN``.

    :param fparam1: The first parameter of the activation function, :math:`\alpha`. See the formulas in the introduction section.

    :param fparam2: The second parameter of the activation function, :math:`\beta`. See the formulas in the introduction section.

    :param termCrit: Termination criteria of the training algorithm. You can specify the maximum number of iterations (``maxCount``) and/or how much the error could change between the iterations to make the algorithm continue (``epsilon``).

    :param train_method: Training method of the MLP. Possible values are:

        * **ANN_MLP_TrainParams::BACKPROP** The back-propagation algorithm.

        * **ANN_MLP_TrainParams::RPROP** The RPROP algorithm.

    :param param1: Parameter of the training method. It is ``rp_dw0`` for ``RPROP`` and ``bp_dw_scale`` for ``BACKPROP``.

    :param param2: Parameter of the training method. It is ``rp_dw_min`` for ``RPROP`` and ``bp_moment_scale`` for ``BACKPROP``.

By default the RPROP algorithm is used:

::

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

ANN_MLP
---------
.. ocv:class:: ANN_MLP : public StatModel

MLP model.

Unlike many other models in ML that are constructed and trained at once, in the MLP model these steps are separated. First, a network with the specified topology is created using the non-default constructor or the method :ocv:func:`ANN_MLP::create`. All the weights are set to zeros. Then, the network is trained using a set of input and output vectors. The training procedure can be repeated more than once, that is, the weights can be adjusted based on the new training data.


ANN_MLP::create
--------------------
Creates empty model

.. ocv:function:: Ptr<ANN_MLP> ANN_MLP::create(const Params& params=Params())

Use ``StatModel::train`` to train the model, ``StatModel::train<ANN_MLP>(traindata, params)`` to create and train the model, ``StatModel::load<ANN_MLP>(filename)`` to load the pre-trained model. Note that the train method has optional flags, and the following flags are handled by ``ANN_MLP``:

        * **UPDATE_WEIGHTS** Algorithm updates the network weights, rather than computes them from scratch. In the latter case the weights are initialized using the Nguyen-Widrow algorithm.

        * **NO_INPUT_SCALE** Algorithm does not normalize the input vectors. If this flag is not set, the training algorithm normalizes each input feature independently, shifting its mean value to 0 and making the standard deviation equal to 1. If the network is assumed to be updated frequently, the new training data could be much different from original one. In this case, you should take care of proper normalization.

        * **NO_OUTPUT_SCALE** Algorithm does not normalize the output vectors. If the flag is not set, the training algorithm normalizes each output feature independently, by transforming it to the certain range depending on the used activation function.


ANN_MLP::setParams
-------------------
Sets the new network parameters

.. ocv:function:: void ANN_MLP::setParams(const Params& params)

    :param params: The new parameters

The existing network, if any, will be destroyed and new empty one will be created. It should be re-trained after that.

ANN_MLP::getParams
-------------------
Retrieves the current network parameters

.. ocv:function:: Params ANN_MLP::getParams() const
