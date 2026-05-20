# Adding (blending) two images using OpenCV

:::{div} opencv-meta-table

|    |    |
| -: | :- |
| Original author | Ana Huamán |
| Compatibility | OpenCV >= 3.0 |

:::

We will learn how to blend two images!
## Goal

In this tutorial you will learn:

-   what is *linear blending* and why it is useful;
-   how to add two images using **addWeighted()**

## Theory

:::{note}
The explanation below belongs to the book [Computer Vision: Algorithms and
 Applications](http://szeliski.org/Book/) by Richard Szeliski
:::
From our previous tutorial, we already know a bit of *Pixel operators*. An interesting dyadic
(two-input) operator is the *linear blend operator*:

$$
g(x) = (1 - \alpha)f_{0}(x) + \alpha f_{1}(x)
$$

By varying $\alpha$ from $0 \rightarrow 1$, this operator can be used to perform a temporal
*cross-dissolve* between two images or videos, as seen in slide shows and film productions (cool,
eh?)

## Source Code

::::{tab-set}
:::{tab-item} C++
:sync: cpp

Download the source code from
[here](https://raw.githubusercontent.com/opencv/opencv/5.x/samples/cpp/tutorial_code/core/AddingImages/AddingImages.cpp).

```{doxyinclude} cpp/tutorial_code/core/AddingImages/AddingImages.cpp
:language: cpp
```

:::
:::{tab-item} Java
:sync: java

Download the source code from
[here](https://raw.githubusercontent.com/opencv/opencv/5.x/samples/java/tutorial_code/core/AddingImages/AddingImages.java).

```{doxyinclude} java/tutorial_code/core/AddingImages/AddingImages.java
:language: java
```

:::
:::{tab-item} Python
:sync: python

Download the source code from
[here](https://raw.githubusercontent.com/opencv/opencv/5.x/samples/python/tutorial_code/core/AddingImages/adding_images.py).

```{doxyinclude} python/tutorial_code/core/AddingImages/adding_images.py
:language: python
```

:::
::::

## Explanation

Since we are going to perform:

$$
g(x) = (1 - \alpha)f_{0}(x) + \alpha f_{1}(x)
$$

We need two source images ($f_{0}(x)$ and $f_{1}(x)$). So, we load them in the usual way:
::::{tab-set}
:::{tab-item} C++
:sync: cpp

```{doxysnippet} cpp/tutorial_code/core/AddingImages/AddingImages.cpp
:tag: load
:language: cpp
```

:::
:::{tab-item} Java
:sync: java

```{doxysnippet} java/tutorial_code/core/AddingImages/AddingImages.java
:tag: load
:language: java
```

:::
:::{tab-item} Python
:sync: python

```{doxysnippet} python/tutorial_code/core/AddingImages/adding_images.py
:tag: load
:language: python
```

:::
::::

We used the following images: [LinuxLogo.jpg](https://raw.githubusercontent.com/opencv/opencv/5.x/samples/data/LinuxLogo.jpg) and [WindowsLogo.jpg](https://raw.githubusercontent.com/opencv/opencv/5.x/samples/data/WindowsLogo.jpg)

:::{warning}
Since we are *adding* *src1* and *src2*, they both have to be of the same size
(width and height) and type.
:::
Now we need to generate the `g(x)` image. For this, the function **addWeighted()** comes quite handy:

::::{tab-set}
:::{tab-item} C++
:sync: cpp

```{doxysnippet} cpp/tutorial_code/core/AddingImages/AddingImages.cpp
:tag: blend_images
:language: cpp
```

:::
:::{tab-item} Java
:sync: java

```{doxysnippet} java/tutorial_code/core/AddingImages/AddingImages.java
:tag: blend_images
:language: java
```

:::
:::{tab-item} Python
:sync: python

```{doxysnippet} python/tutorial_code/core/AddingImages/adding_images.py
:tag: blend_images
:language: python
```

Numpy version of above line (but cv function is around 2x faster):
\code{.py}
    dst = np.uint8(alpha*(img1)+beta*(img2))
\endcode
:::
::::

since **addWeighted()**  produces:
$$
dst = \alpha \cdot src1 + \beta \cdot src2 + \gamma
$$
In this case, `gamma` is the argument $0.0$ in the code above.

Create windows, show the images and wait for the user to end the program.
::::{tab-set}
:::{tab-item} C++
:sync: cpp

```{doxysnippet} cpp/tutorial_code/core/AddingImages/AddingImages.cpp
:tag: display
:language: cpp
```

:::
:::{tab-item} Java
:sync: java

```{doxysnippet} java/tutorial_code/core/AddingImages/AddingImages.java
:tag: display
:language: java
```

:::
:::{tab-item} Python
:sync: python

```{doxysnippet} python/tutorial_code/core/AddingImages/adding_images.py
:tag: display
:language: python
```

:::
::::

## Result

![](images/Adding_Images_Tutorial_Result_Big.jpg)
