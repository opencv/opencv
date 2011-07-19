.. _maskOperations:

Mask operations on matrixes
***************************

Mask operations on matrixes are quite simple. The idea is that we recalculate each pixels value in an image according to a matrix mask. This mask holds values that will just how much influence have neighbor pixel values (and the pixel value itself) to the new pixel value. From a mathematical point of view we make a weighted average, with our specified values. 

Our test case
=============

Let us consider the issue of an image contrast enchancement method. Basically we want to apply for every pixel of the image the following formula: 

.. math::

   I(i,j) = 5*I(i,j) - [ I(i-1,j) + I(i+1,j) + I(i,j-1) + I(i,j+1)] 

	\iff I(i,j)*M, \text{where }
	M = \bordermatrix{ _i\backslash ^j  & -1 &  0 & -1 \cr
						-1 &  0 & -1 &  0 \cr
						 0 & -1 &  5 & -1 \cr
						+1 &  0 & -1 &  0 \cr
					  }

The first notation is by using a formula, while the second is a compacted version of the first by using a mask. You use the mask by puting the center of the mask matrix (in the upper case noted by the zero-zero index) on the pixel you want to calculate and sum up the pixel values multiplicated with the overlapped matrix values. It's the same thing, however in case of large matrices the later notation is a lot easier to look over. 
