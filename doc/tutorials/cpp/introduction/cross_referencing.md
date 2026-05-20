# Cross referencing OpenCV from other Doxygen projects

:::{div} opencv-meta-table

|    |    |
| -: | :- |
| Original author | Sebastian Höffner |
| Compatibility | OpenCV >= 3.3.0 |

:::

:::{warning}
This tutorial can contain obsolete information.
:::
## Cross referencing OpenCV

[Doxygen](http://www.doxygen.nl) is a tool to generate
documentations like the OpenCV documentation you are reading right now.
It is used by a variety of software projects and if you happen to use it
to generate your own documentation, and you are using OpenCV inside your
project, this short tutorial is for you.

Imagine this warning inside your documentation code:

```
/**
 * @warning This functions returns a cv::Mat.
 */
```

Inside your generated documentation this warning will look roughly like this:

:::{warning}
This functions returns a cv::Mat.
:::
While inside the OpenCV documentation the `cv::Mat` is rendered as a link:

:::{warning}
This functions returns a [cv::Mat](https://docs.opencv.org/5.x/d3/d63/classcv_1_1Mat.html).
:::
To generate links to the OpenCV documentation inside your project, you only
have to perform two small steps. First download the file
[opencv.tag](https://docs.opencv.org/5.x/opencv.tag) (right-click and choose "save as...") and place it
somewhere in your project directory, for example as
`docs/doxygen-tags/opencv.tag`.

Open your Doxyfile using your favorite text editor and search for the key
`TAGFILES`. Change it as follows:

```
TAGFILES = ./docs/doxygen-tags/opencv.tag=http://docs.opencv.org/5.0.0-alpha
```

If you had other definitions already, you can append the line using a `\`:

```
TAGFILES = ./docs/doxygen-tags/libstdc++.tag=https://gcc.gnu.org/onlinedocs/libstdc++/latest-doxygen \
           ./docs/doxygen-tags/opencv.tag=http://docs.opencv.org/5.0.0-alpha
```

Doxygen can now use the information from the tag file to link to the OpenCV
documentation. Rebuild your documentation right now!

:::{note}
To allow others to also use a *.tag file to link to your documentation,
set `GENERATE_TAGFILE = html/your_project.tag`. Your documentation will now
contain a `your_project.tag` file in its root directory.
:::
## References

- [Doxygen: Linking to external documentation](http://www.doxygen.nl/manual/external.html)
- [opencv.tag](https://docs.opencv.org/5.x/opencv.tag)
