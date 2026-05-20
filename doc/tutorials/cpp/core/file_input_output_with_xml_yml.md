# File Input and Output using XML / YAML / JSON files

:::{div} opencv-meta-table

|    |    |
| -: | :- |
| Original author | Bernát Gábor |
| Compatibility | OpenCV >= 3.0 |

:::

## Goal

You'll find answers to the following questions:

-   How do you print and read text entries to a file in OpenCV using YAML, XML, or JSON files?
-   How can you perform the same operations for OpenCV data structures?
-   How can this be done for your custom data structures?
-   How do you use OpenCV data structures, such as [cv::FileStorage](https://docs.opencv.org/5.x/da/d56/classcv_1_1FileStorage.html) , [cv::FileNode](https://docs.opencv.org/5.x/de/dd9/classcv_1_1FileNode.html) or [cv::FileNodeIterator](https://docs.opencv.org/5.x/d7/d4e/classcv_1_1FileNodeIterator.html) .

## Source code

::::{tab-set}
:::{tab-item} C++
:sync: cpp

You can [download this from here
](https://github.com/opencv/opencv/tree/5.x/samples/cpp/tutorial_code/core/file_input_output/file_input_output.cpp) or find it in the
`samples/cpp/tutorial_code/core/file_input_output/file_input_output.cpp` of the OpenCV source code
library.

Here's a sample code of how to achieve all the stuff enumerated at the goal list.

```{doxyinclude} cpp/tutorial_code/core/file_input_output/file_input_output.cpp
:language: cpp
```

:::
:::{tab-item} Python
:sync: python

You can [download this from here
](https://github.com/opencv/opencv/tree/5.x/samples/python/tutorial_code/core/file_input_output/file_input_output.py) or find it in the
`samples/python/tutorial_code/core/file_input_output/file_input_output.py` of the OpenCV source code
library.

Here's a sample code of how to achieve all the stuff enumerated at the goal list.

```{doxyinclude} python/tutorial_code/core/file_input_output/file_input_output.py
:language: python
```

:::
::::

## Explanation

Here we talk only about XML, YAML and JSON file inputs. Your output (and its respective input) file may
have only one of these extensions and the structure coming from this. They are two kinds of data
structures you may serialize: *mappings* (like the STL map and the Python dictionary) and *element sequence* (like the STL
vector). The difference between these is that in a map every element has a unique name through what
you may access it. For sequences you need to go through them to query a specific item.

1. **XML/YAML/JSON File Open and Close.** Before you write any content to such file you need to open it
   and at the end to close it. The XML/YAML/JSON data structure in OpenCV is [cv::FileStorage](https://docs.opencv.org/5.x/da/d56/classcv_1_1FileStorage.html) . To
   specify that this structure to which file binds on your hard drive you can use either its
   constructor or the *open()* function of this:
   ::::{tab-set}
   :::{tab-item} C++
   :sync: cpp

   ```{doxysnippet} cpp/tutorial_code/core/file_input_output/file_input_output.cpp
   :tag: open
   :language: cpp
   ```

   :::
   :::{tab-item} Python
   :sync: python

   ```{doxysnippet} python/tutorial_code/core/file_input_output/file_input_output.py
   :tag: open
   :language: python
   ```

   :::
   ::::

   Either one of this you use the second argument is a constant specifying the type of operations
   you'll be able to on them: WRITE, READ or APPEND. The extension specified in the file name also
   determinates the output format that will be used. The output may be even compressed if you
   specify an extension such as *.xml.gz*.

   The file automatically closes when the [cv::FileStorage](https://docs.opencv.org/5.x/da/d56/classcv_1_1FileStorage.html) objects is destroyed. However, you
   may explicitly call for this by using the *release* function:
   ::::{tab-set}
   :::{tab-item} C++
   :sync: cpp

   ```{doxysnippet} cpp/tutorial_code/core/file_input_output/file_input_output.cpp
   :tag: close
   :language: cpp
   ```

   :::
   :::{tab-item} Python
   :sync: python

   ```{doxysnippet} python/tutorial_code/core/file_input_output/file_input_output.py
   :tag: close
   :language: python
   ```

   :::
   ::::

1. **Input and Output of text and numbers.** In C++, the data structure uses the \<\< output
   operator in the STL library. In Python, [cv::FileStorage](https://docs.opencv.org/5.x/da/d56/classcv_1_1FileStorage.html).write() is used instead. For
   outputting any type of data structure we need first to specify its name. We do this by just
   simply pushing the name of this to the stream in C++. In Python, the first parameter for the
   write function is the name. For basic types you may follow this with the print of the value :
   ::::{tab-set}
   :::{tab-item} C++
   :sync: cpp

   ```{doxysnippet} cpp/tutorial_code/core/file_input_output/file_input_output.cpp
   :tag: writeNum
   :language: cpp
   ```

   :::
   :::{tab-item} Python
   :sync: python

   ```{doxysnippet} python/tutorial_code/core/file_input_output/file_input_output.py
   :tag: writeNum
   :language: python
   ```

   :::
   ::::

   Reading in is a simple addressing (via the [] operator) and casting operation or a read via
   the \>\> operator. In Python, we address with getNode() and use real() :
   ::::{tab-set}
   :::{tab-item} C++
   :sync: cpp

   ```{doxysnippet} cpp/tutorial_code/core/file_input_output/file_input_output.cpp
   :tag: readNum
   :language: cpp
   ```

   :::
   :::{tab-item} Python
   :sync: python

   ```{doxysnippet} cpp/tutorial_code/core/file_input_output/file_input_output.cpp
   :tag: readNum
   :language: cpp
   ```

   :::
   ::::

1. **Input/Output of OpenCV Data structures.** Well these behave exactly just as the basic C++
   and Python types:
   ::::{tab-set}
   :::{tab-item} C++
   :sync: cpp

   ```{doxysnippet} cpp/tutorial_code/core/file_input_output/file_input_output.cpp
   :tag: iomati
   :language: cpp
   ```

   ```{doxysnippet} cpp/tutorial_code/core/file_input_output/file_input_output.cpp
   :tag: iomatw
   :language: cpp
   ```

   ```{doxysnippet} cpp/tutorial_code/core/file_input_output/file_input_output.cpp
   :tag: iomat
   :language: cpp
   ```

   :::
   :::{tab-item} Python
   :sync: python

   ```{doxysnippet} python/tutorial_code/core/file_input_output/file_input_output.py
   :tag: iomati
   :language: python
   ```

   ```{doxysnippet} python/tutorial_code/core/file_input_output/file_input_output.py
   :tag: iomatw
   :language: python
   ```

   ```{doxysnippet} python/tutorial_code/core/file_input_output/file_input_output.py
   :tag: iomat
   :language: python
   ```

   :::
   ::::

1. **Input/Output of vectors (arrays) and associative maps.** As I mentioned beforehand, we can
   output maps and sequences (array, vector) too. Again we first print the name of the variable and
   then we have to specify if our output is either a sequence or map.

   For sequence before the first element print the "[" character and after the last one the "]"
   character. With Python, call `FileStorage.startWriteStruct(structure_name, struct_type)`,
   where `struct_type` is `cv2.FileNode_MAP` or `cv2.FileNode_SEQ` to start writing the structure.
   Call `FileStorage.endWriteStruct()` to finish the structure:
   ::::{tab-set}
   :::{tab-item} C++
   :sync: cpp

   ```{doxysnippet} cpp/tutorial_code/core/file_input_output/file_input_output.cpp
   :tag: writeStr
   :language: cpp
   ```

   :::
   :::{tab-item} Python
   :sync: python

   ```{doxysnippet} python/tutorial_code/core/file_input_output/file_input_output.py
   :tag: writeStr
   :language: python
   ```

   :::
   ::::

   For maps the drill is the same however now we use the "{" and "}" delimiter characters:
   ::::{tab-set}
   :::{tab-item} C++
   :sync: cpp

   ```{doxysnippet} cpp/tutorial_code/core/file_input_output/file_input_output.cpp
   :tag: writeMap
   :language: cpp
   ```

   :::
   :::{tab-item} Python
   :sync: python

   ```{doxysnippet} python/tutorial_code/core/file_input_output/file_input_output.py
   :tag: writeMap
   :language: python
   ```

   :::
   ::::

   To read from these we use the [cv::FileNode](https://docs.opencv.org/5.x/de/dd9/classcv_1_1FileNode.html) and the [cv::FileNodeIterator](https://docs.opencv.org/5.x/d7/d4e/classcv_1_1FileNodeIterator.html) data
   structures. The [] operator of the [cv::FileStorage](https://docs.opencv.org/5.x/da/d56/classcv_1_1FileStorage.html) class (or the getNode() function in Python) returns a [cv::FileNode](https://docs.opencv.org/5.x/de/dd9/classcv_1_1FileNode.html) data
   type. If the node is sequential we can use the [cv::FileNodeIterator](https://docs.opencv.org/5.x/d7/d4e/classcv_1_1FileNodeIterator.html) to iterate through the
   items. In Python, the at() function can be used to address elements of the sequence and the
   size() function returns the length of the sequence:
   ::::{tab-set}
   :::{tab-item} C++
   :sync: cpp

   ```{doxysnippet} cpp/tutorial_code/core/file_input_output/file_input_output.cpp
   :tag: readStr
   :language: cpp
   ```

   :::
   :::{tab-item} Python
   :sync: python

   ```{doxysnippet} python/tutorial_code/core/file_input_output/file_input_output.py
   :tag: readStr
   :language: python
   ```

   :::
   ::::

   For maps you can use the [] operator (at() function in Python) again to access the given item (or the \>\> operator too):
   ::::{tab-set}
   :::{tab-item} C++
   :sync: cpp

   ```{doxysnippet} cpp/tutorial_code/core/file_input_output/file_input_output.cpp
   :tag: readMap
   :language: cpp
   ```

   :::
   :::{tab-item} Python
   :sync: python

   ```{doxysnippet} python/tutorial_code/core/file_input_output/file_input_output.py
   :tag: readMap
   :language: python
   ```

   :::
   ::::

1. **Read and write your own data structures.** Suppose you have a data structure such as:
   ::::{tab-set}
   :::{tab-item} C++
   :sync: cpp

   ```cpp
   class MyData
   {
   public:
         MyData() : A(0), X(0), id() {}
   public:   // Data Members
      int A;
      double X;
      string id;
   };
   ```

   :::
   :::{tab-item} Python
   :sync: python

   ```py
   class MyData:
       def __init__(self):
           self.A = self.X = 0
           self.name = ''
   ```

   :::
   ::::

   In C++, it's possible to serialize this through the OpenCV I/O XML/YAML interface (just as
   in case of the OpenCV data structures) by adding a read and a write function inside and outside of your
   class. In Python, you can get close to this by implementing a read and write function inside
   the class. For the inside part:
   ::::{tab-set}
   :::{tab-item} C++
   :sync: cpp

   ```{doxysnippet} cpp/tutorial_code/core/file_input_output/file_input_output.cpp
   :tag: inside
   :language: cpp
   ```

   :::
   :::{tab-item} Python
   :sync: python

   ```{doxysnippet} python/tutorial_code/core/file_input_output/file_input_output.py
   :tag: inside
   :language: python
   ```

   :::
   ::::

   In C++, you need to add the following functions definitions outside the class:

   ```{doxysnippet} cpp/tutorial_code/core/file_input_output/file_input_output.cpp
   :tag: outside
   :language: cpp
   ```

   Here you can observe that in the read section we defined what happens if the user tries to read
   a non-existing node. In this case we just return the default initialization value, however a
   more verbose solution would be to return for instance a minus one value for an object ID.

   Once you added these four functions use the \>\> operator for write and the \<\< operator for
   read (or the defined input/output functions for Python):
   ::::{tab-set}
   :::{tab-item} C++
   :sync: cpp

   ```{doxysnippet} cpp/tutorial_code/core/file_input_output/file_input_output.cpp
   :tag: customIOi
   :language: cpp
   ```

   ```{doxysnippet} cpp/tutorial_code/core/file_input_output/file_input_output.cpp
   :tag: customIOw
   :language: cpp
   ```

   ```{doxysnippet} cpp/tutorial_code/core/file_input_output/file_input_output.cpp
   :tag: customIO
   :language: cpp
   ```

   :::
   :::{tab-item} Python
   :sync: python

   ```{doxysnippet} python/tutorial_code/core/file_input_output/file_input_output.py
   :tag: customIOi
   :language: python
   ```

   ```{doxysnippet} python/tutorial_code/core/file_input_output/file_input_output.py
   :tag: customIOw
   :language: python
   ```

   ```{doxysnippet} python/tutorial_code/core/file_input_output/file_input_output.py
   :tag: customIO
   :language: python
   ```

   :::
   ::::

   Or to try out reading a non-existing read:
   ::::{tab-set}
   :::{tab-item} C++
   :sync: cpp

   ```{doxysnippet} cpp/tutorial_code/core/file_input_output/file_input_output.cpp
   :tag: nonexist
   :language: cpp
   ```

   :::
   :::{tab-item} Python
   :sync: python

   ```{doxysnippet} python/tutorial_code/core/file_input_output/file_input_output.py
   :tag: nonexist
   :language: python
   ```

   :::
   ::::

## Result

Well mostly we just print out the defined numbers. On the screen of your console you could see:

```bash
Write Done.

Reading:
100image1.jpg
Awesomeness
baboon.jpg
Two  2; One  1

R = [1, 0, 0;
  0, 1, 0;
  0, 0, 1]
T = [0; 0; 0]

MyData =
{ id = mydata1234, X = 3.14159, A = 97}

Attempt to read NonExisting (should initialize the data structure with its default).
NonExisting =
{ id = , X = 0, A = 0}

Tip: Open up output.xml with a text editor to see the serialized data.
```

Nevertheless, it's much more interesting what you may see in the output xml file:

```xml
<?xml version="1.0"?>
<opencv_storage>
<iterationNr>100</iterationNr>
<strings>
  image1.jpg Awesomeness baboon.jpg</strings>
<Mapping>
  <One>1</One>
  <Two>2</Two></Mapping>
<R type_id="opencv-matrix">
  <rows>3</rows>
  <cols>3</cols>
  <dt>u</dt>
  <data>
    1 0 0 0 1 0 0 0 1</data></R>
<T type_id="opencv-matrix">
  <rows>3</rows>
  <cols>1</cols>
  <dt>d</dt>
  <data>
    0. 0. 0.</data></T>
<MyData>
  <A>97</A>
  <X>3.1415926535897931e+000</X>
  <id>mydata1234</id></MyData>
</opencv_storage>
```

Or the YAML file:

```yaml
YAML:1.0
iterationNr: 100
strings:
   - "image1.jpg"
   - Awesomeness
   - "baboon.jpg"
Mapping:
   One: 1
   Two: 2
R: !!opencv-matrix
   rows: 3
   cols: 3
   dt: u
   data: [ 1, 0, 0, 0, 1, 0, 0, 0, 1 ]
T: !!opencv-matrix
   rows: 3
   cols: 1
   dt: d
   data: [ 0., 0., 0. ]
MyData:
   A: 97
   X: 3.1415926535897931e+000
   id: mydata1234
```

You may observe a runtime instance of this on the [YouTube
here](https://www.youtube.com/watch?v=A4yqVnByMMM) .

```{raw} html
<div class="responsive-iframe" style="position:relative;padding-bottom:56.25%;height:0;overflow:hidden;max-width:100%;margin:1.5rem 0;">
  <iframe style="position:absolute;top:0;left:0;width:100%;height:100%;border:0;" src="https://www.youtube-nocookie.com/embed/A4yqVnByMMM?rel=0" title="YouTube video" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
```
