# Hit-or-Miss

:::{div} opencv-meta-table

|    |    |
| -: | :- |
| Original author | Lorena García |
| Compatibility | OpenCV >= 3.0 |

:::

## Goal

In this tutorial you will learn how to find a given configuration or pattern in a binary image by using the Hit-or-Miss transform (also known as Hit-and-Miss transform).
This transform is also the basis of more advanced morphological operations such as thinning or pruning.

We will use the OpenCV function **morphologyEx()** .

## Hit-or-Miss theory

Morphological operators process images based on their shape. These operators apply one or more *structuring elements* to an input image to obtain the output image.
The two basic morphological operations are the *erosion* and the *dilation*. The combination of these two operations generate advanced morphological transformations such as *opening*, *closing*, or *top-hat* transform.
To know more about these and other basic morphological operations refer to previous tutorials ([Eroding and Dilating](https://docs.opencv.org/5.x/db/df6/tutorial_erosion_dilatation.html)) and ([More Morphology Transformations](https://docs.opencv.org/5.x/d3/dbe/tutorial_opening_closing_hats.html)).

The Hit-or-Miss transformation is useful to find patterns in binary images. In particular, it finds those pixels whose neighbourhood matches the shape of a first structuring element $B_1$
while not matching the shape of a second structuring element $B_2$ at the same time. Mathematically, the operation applied to an image $A$ can be expressed as follows:

$$

A\circledast B = (A\ominus B_1) \cap (A^c\ominus B_2)

$$

Therefore, the hit-or-miss operation comprises three steps:
1. Erode image $A$ with structuring element $B_1$.
2. Erode the complement of image $A$ ($A^c$) with structuring element $B_2$.
3. AND results from step 1 and step 2.

The structuring elements $B_1$ and $B_2$ can be combined into a single element $B$. Let's see an example:
```{figure} images/hitmiss_kernels.png
:alt: Structuring elements (kernels). Left: kernel to 'hit'. Middle: kernel to 'miss'. Right: final combined kernel

Structuring elements (kernels). Left: kernel to 'hit'. Middle: kernel to 'miss'. Right: final combined kernel
```

In this case, we are looking for a pattern in which the central pixel belongs to the background while the north, south, east, and west pixels belong to the foreground. The rest of pixels in the neighbourhood can be of any kind, we don't care about them. Now, let's apply this kernel to an input image:

```{figure} images/hitmiss_input.png
:alt: Input binary image

Input binary image
```
```{figure} images/hitmiss_output.png
:alt: Output binary image

Output binary image
```

You can see that the pattern is found in just one location within the image.

## Code

The code corresponding to the previous example is shown below.

::::{tab-set}
:::{tab-item} C++
:sync: cpp

You can also download it from
[here](https://raw.githubusercontent.com/opencv/opencv/5.x/samples/cpp/tutorial_code/ImgProc/HitMiss/HitMiss.cpp)

```{doxyinclude} samples/cpp/tutorial_code/ImgProc/HitMiss/HitMiss.cpp
:language: cpp
```

:::
:::{tab-item} Java
:sync: java

You can also download it from
[here](https://raw.githubusercontent.com/opencv/opencv/5.x/samples/java/tutorial_code/ImgProc/HitMiss/HitMiss.java)

```{doxyinclude} samples/java/tutorial_code/ImgProc/HitMiss/HitMiss.java
:language: java
```

:::
:::{tab-item} Python
:sync: python

You can also download it from
[here](https://raw.githubusercontent.com/opencv/opencv/5.x/samples/python/tutorial_code/imgProc/HitMiss/hit_miss.py)

```{doxyinclude} samples/python/tutorial_code/imgProc/HitMiss/hit_miss.py
:language: python
```

:::
::::

As you can see, it is as simple as using the function **morphologyEx()** with the operation type **MORPH_HITMISS** and the chosen kernel.

## Other examples

Here you can find the output results of applying different kernels to the same input image used before:

```{figure} images/hitmiss_example2.png
:alt: Kernel and output result for finding top-right corners

Kernel and output result for finding top-right corners
```
```{figure} images/hitmiss_example3.png
:alt: Kernel and output result for finding left end points

Kernel and output result for finding left end points
```

Now try your own patterns!
