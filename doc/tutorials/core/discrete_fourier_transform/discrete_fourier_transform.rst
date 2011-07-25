.. _discretFourierTransform:

Discrete Fourier Transform
**************************

Goal
==== 

We'll seek answers for the following questions: 

.. container:: enumeratevisibleitemswithsquare

   + What is a Fourier transform and why use it? 
   + How to do it in OpenCV? 
   + Usage of functions such as: :imgprocfilter:`copyMakeBorder() <copymakeborder>`, :operationsonarrays:`merge() <merge>`, :operationsonarrays:`dft() <dft>`, :operationsonarrays:`getOptimalDFTSize() <getoptimaldftsize>`, :operationsonarrays:`log() <log>` and :operationsonarrays:`normalize() <normalize>` .

Source code
===========

Here's a sample usage of :operationsonarrays:`dft() <dft>` : 

.. literalinclude:: ../../../../samples/cpp/tutorial_code/core/discrete_fourier_transform/discrete_fourier_transform.cpp
   :language: cpp
   :linenos:
   :tab-width: 4
   :lines: 1-3, 5, 19-20, 23-78

Explanation
===========

The Fourier Transform will decompose an image into its sinus and cosines components. In other words, it will transform an image from its spatial domain to its frequency domain. The idea is that any function may be approximated exactly with the sum of infinite sinus and cosines functions. The Fourier Transform is a way how to do this. Mathematically a two dimensional images Fourier transform is: 

.. math::
   
   F(k,l) = \displaystyle\sum\limits_{i=0}^{N-1}\sum\limits_{j=0}^{N-1} f(i,j)e^{-i2\pi(\frac{ki}{N}+\frac{lj}{N})} 
   
   e^{ix} = \cos{x} + i\sin {x}

