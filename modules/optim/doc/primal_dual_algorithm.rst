Primal-Dual Algorithm
=======================

.. highlight:: cpp

optim::denoise_TVL1
---------------------------------

Primal-dual algorithm is an algorithm for solving special types of variational
problems (that is, finding a function to minimize some functional)
. As the image denoising, in particular, may be seen as the variational
problem, primal-dual algorithm then can be used to perform denoising and this
is exactly what is implemented.

It should be noted, that this implementation was taken from the July 2013 blog entry [Mordvintsev]_, which also contained
(slightly more general) ready-to-use
source code on Python. Subsequently, that code was rewritten on C++ with the usage of openCV by Vadim Pisarevsky
at the end of July 2013 and finally it was slightly adapted by later authors.

Although the thorough discussion and justification
of the algorithm involved may be found in [ChambolleEtAl]_, it might make sense to skim over it here, following [Mordvintsev]_. To
begin with, we consider the 1-byte gray-level images as the functions from the rectangular domain of pixels
(it may be seen as set :math:`\left\{(x,y)\in\mathbb{N}\times\mathbb{N}\mid 1\leq x\leq n,\;1\leq y\leq m\right\}`
for some :math:`m,\;n\in\mathbb{N}`) into :math:`\{0,1,\dots,255\}`. We shall denote the noised images as :math:`f_i` and with this
view, given some image :math:`x` of the same size, we may measure how bad it is by the formula

.. math::
        \left\|\left\|\nabla x\right\|\right\| + \lambda\sum_i\left\|\left\|x-f_i\right\|\right\|

:math:`\|\|\cdot\|\|` here denotes :math:`L_2`-norm and as you see, the first addend states that we want our image to be smooth
(ideally, having zero gradient, thus being constant) and the second states that we want our result to be close to the observations we've got.
If we treat :math:`x` as a function, this is exactly the functional what we seek to minimize and here the Primal-Dual algorithm comes
into play.

.. ocv:function:: void optim::denoise_TVL1(const std::vector<Mat>& observations,Mat& result, double lambda, int niters)

    :param observations: This array should contain one or more noised versions of the image that is to be restored.

    :param result: Here the denoised image will be stored. There is no need to do pre-allocation of storage space, as it will be automatically allocated, if necessary.

    :param lambda: Corresponds to :math:`\lambda` in the formulas above. As it is enlarged, the smooth (blurred) images are treated more favorably than detailed (but maybe more noised) ones. Roughly speaking, as it becomes smaller, the result will be more blur but more sever outliers will be removed.

    :param niters: Number of iterations that the algorithm will run. Of course, as more iterations as better, but it is hard to quantitatively refine this statement, so just use the default and increase it if the results are poor.


.. [ChambolleEtAl] A. Chambolle, V. Caselles, M. Novaga, D. Cremers and T. Pock, An Introduction to Total Variation for Image Analysis, http://hal.archives-ouvertes.fr/docs/00/43/75/81/PDF/preprint.pdf (pdf)

.. [Mordvintsev] Alexander Mordvintsev, ROF and TV-L1 denoising with Primal-Dual algorithm, http://znah.net/rof-and-tv-l1-denoising-with-primal-dual-algorithm.html (blog entry)



DepthmapDenoiseWeightedHuber
---------------------------------

.. highlight:: cpp

cuda::DepthmapDenoiseWeightedHuber
----------------------------------

In 3D reconstruction, a common way to find the location of points in space is to intersect the rays of one image with another, and considering points where rays of the same color intersect to be more likely. This gives a cost as a function of the depth chosen at each pixel:

.. math::
        \mathbf{C(d)}

(This is implicitly summed over the whole image.)  
The problem is that may different depths will produce reasonable colors when pixels are considered individually so we add a term to the cost penalizing the difference between pixels and add a weighting factor:

.. math::
        f\mathbf{(\nabla\mathbf{d}) + \lambda C(d)}

where :math:`\mathbf{\nabla d}` is the difference between neighbouring pixels.

.. note:: This is a slight abuse of notation since there are actually four neighbors for each pixel, so we evaluate the left term over each pair of neighbors and sum. 

One common choice for :math:`f` is the *Huber norm*, which is 

.. math::
        \left\| \nabla \mathbf{d} \right\|_e= \left\{\begin{matrix} \frac{(\nabla \mathbf{d})^2}{2e} & \mathrm{if} \left | \nabla \mathbf{d} \right | < e \\ \left |\nabla \mathbf{d} \right | &\mathrm{else} \end{matrix}\right. 

This is the same as a metal wire: it starts out acting like a spring, but when stretched too much it deforms plastically.

This gives:

.. math::
        \left\| \nabla \mathbf{d} \right\|_e  + \lambda \mathbf{C(\mathbf{d})}

This problem is intractable to solve so we do a *relaxation*: we repeatedly solve the left and right sides independently, but enforce that the two solutions must be increasingly similar as we go along. We do this by creating a spring force between the two solutions. Remember from physics that a spring's energy is expressed as :math:`\frac{1}{2} k(x_1-x_2)^2`, so we write:

.. math::
        \left\| \nabla \mathbf{d} \right\|_e + \frac{1}{2\theta} (\mathbf{d-a})^2  + \lambda \mathbf{C(\mathbf{a})}


:math:`\frac{1}{\theta}` is the spring constant, :math:`\mathbf{d}` is one solution, and :math:`\mathbf{a}` is the other. We refer to :math:`\theta` as the *stiffness*.

We can also give a hint to the left hand side that certain places are likely to have discontinuities by varying the thickness of the wires:

.. math::
        \mathbf{g}\left\| \nabla \mathbf{d} \right\|_e + \frac{1}{2\theta} (\mathbf{d-a})^2  + \lambda \mathbf{C(\mathbf{a})}

The function :math:`\mathbf{g}` is the weight function.

The right half is literally just a search through all possible values of :math:`\mathbf{C(\mathbf{a})}` for each pixel.

It turns out that solving the left half:

.. math::
        \mathbf{g}\left\| \nabla \mathbf{d} \right\|_e + \frac{1}{2\theta} (\mathbf{d-a}_{fixed})^2

| is quite hard.  
| OpenCV provides a class for this:

.. ocv:class:: cuda::DepthmapDenoiseWeightedHuberCuda : public cv::DepthmapDenoiseWeightedHuber

Refines a depthmap estimate with DTAM's [DTAM]_ variant of Chambolle and Pock's 
primal-dual algorithm [ChambollePock]_ ::
   
    class DepthmapDenoiseWeightedHuber : public cv::Algorithm
    {
    public:
        //! This may be called repeatedly to iteratively refine the internal depthmap
        virtual cv::cuda::GpuMat operator()(InputArray input,
                                            float epsilon,
                                            float theta) = 0;
        
        //! In case you want to do these explicitly, or use a custom g function
        //! gx(x,y) is the weight between pixels (x,y) and (x+1,y) (right neighbor)
        //! gy(x,y) is the weight between pixels (x,y) and (x,y+1) (down neighbor)
        virtual void allocate(int rows, int cols, InputArray gx=GpuMat(),InputArray gy=GpuMat()) = 0;
        virtual void cacheGValues(InputArray visibleLightImage=GpuMat()) = 0;
    };


DepthmapDenoiseWeightedHuber::allocate
--------------------------------------

Use to preallocate memory for the functor or replace the internal :math:`\mathbf{g}` function buffers with custom ones.
        
 .. ocv:function:: allocate(int rows, int cols, InputArray gx=GpuMat(),InputArray gy=GpuMat())

    :param rows, cols: Size of the image to process
    
    :param gx,gy: Optional replacement for the :math:`\mathbf{g}` function that would be 
    
DepthmapDenoiseWeightedHuber::cacheGValues
------------------------------------------

Used to precache the :math:`\mathbf{g}` values or add a ``visibleLightImage`` after object creation.

.. ocv:function:: void cacheGValues(InputArray visibleLightImage=GpuMat())

    :param visibleLightImage: Optional replacement for the ``visibleLightImage`` parameter at creation. 

createDepthmapDenoiseWeightedHuber
----------------------------------

Generates a denoising functor to handle the algorithm state on the GPU.

.. ocv:function::  Ptr<DepthmapDenoiseWeightedHuber> createDepthmapDenoiseWeightedHuber(InputArray visibleLightImage=GpuMat(),  Stream s=Stream::Null())
    
    :param visibleLightImage: This is an optional grayscale image (CV_32FC1 for best performance) of the scene corresponding to the depthmap to be optimized. The algorithm uses the image to construct the :math:`\mathbf{g}` function, to provide hints for the location of edge discontinuities.
    
    :param s:The stream to run on. The functor is fully asyncronous except for memory allocation (always synchronous as of the current cuda release). allocate() and cacheGValues() can be called to 


.. [ChambollePock] Antonin Chambolle and Thomas Pock. "A first-order primal-dual algorithm for convex problems with applications to imaging." 
.. [DTAM] Paul Foster's implementation of algorithm by Richard Newcombe, Steven J. Lovegrove, and Andrew J. Davison. "DTAM: Dense tracking and mapping in real-time."

