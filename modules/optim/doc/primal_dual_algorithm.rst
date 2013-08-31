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
