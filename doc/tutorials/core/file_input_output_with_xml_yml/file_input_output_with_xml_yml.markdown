File Input and Output using XML and YAML files {#tutorial_file_input_output_with_xml_yml}
==============================================

@tableofcontents

@prev_tutorial{tutorial_discrete_fourier_transform}
@next_tutorial{tutorial_how_to_use_OpenCV_parallel_for_new}

|    |    |
| -: | :- |
| Original author | Bernát Gábor |
| Compatibility | OpenCV >= 3.0 |

Goal
----

You'll find answers for the following questions:

-   How to print and read text entries to a file and OpenCV using YAML or XML files?
-   How to do the same for OpenCV data structures?
-   How to do this for your data structures?
-   Usage of OpenCV data structures such as @ref cv::FileStorage , @ref cv::FileNode or @ref
    cv::FileNodeIterator .

Source code
-----------
@add_toggle_cpp
You can [download this from here
](https://github.com/opencv/opencv/tree/master/samples/cpp/tutorial_code/core/file_input_output/file_input_output.cpp) or find it in the
`samples/cpp/tutorial_code/core/file_input_output/file_input_output.cpp` of the OpenCV source code
library.

Here's a sample code of how to achieve all the stuff enumerated at the goal list.

@include cpp/tutorial_code/core/file_input_output/file_input_output.cpp
@end_toggle

@add_toggle_python
You can [download this from here
](https://github.com/opencv/opencv/tree/master/samples/python/tutorial_code/core/file_input_output/file_input_output.py) or find it in the
`samples/python/tutorial_code/core/file_input_output/file_input_output.py` of the OpenCV source code
library.

Here's a sample code of how to achieve all the stuff enumerated at the goal list.

@include python/tutorial_code/core/file_input_output/file_input_output.py
@end_toggle

Explanation
-----------

Here we talk only about XML and YAML file inputs. Your output (and its respective input) file may
have only one of these extensions and the structure coming from this. They are two kinds of data
structures you may serialize: *mappings* (like the STL map and the Python dictionary) and *element sequence* (like the STL
vector). The difference between these is that in a map every element has a unique name through what
you may access it. For sequences you need to go through them to query a specific item.

-#  **XML/YAML File Open and Close.** Before you write any content to such file you need to open it
    and at the end to close it. The XML/YAML data structure in OpenCV is @ref cv::FileStorage . To
    specify that this structure to which file binds on your hard drive you can use either its
    constructor or the *open()* function of this:
    @add_toggle_cpp
    @snippet cpp/tutorial_code/core/file_input_output/file_input_output.cpp open
    @end_toggle
    @add_toggle_python
    @snippet python/tutorial_code/core/file_input_output/file_input_output.py open
    @end_toggle
    Either one of this you use the second argument is a constant specifying the type of operations
    you'll be able to on them: WRITE, READ or APPEND. The extension specified in the file name also
    determinates the output format that will be used. The output may be even compressed if you
    specify an extension such as *.xml.gz*.

    The file automatically closes when the @ref cv::FileStorage objects is destroyed. However, you
    may explicitly call for this by using the *release* function:
    @add_toggle_cpp
    @snippet cpp/tutorial_code/core/file_input_output/file_input_output.cpp close
    @end_toggle
    @add_toggle_python
    @snippet python/tutorial_code/core/file_input_output/file_input_output.py close
    @end_toggle
-#  **Input and Output of text and numbers.** In C++, the data structure uses the \<\< output
    operator in the STL library. In Python, @ref cv::FileStorage.write() is used instead. For
    outputting any type of data structure we need first to specify its name. We do this by just
    simply pushing the name of this to the stream in C++. In Python, the first parameter for the
    write function is the name. For basic types you may follow this with the print of the value :
    @add_toggle_cpp
    @snippet cpp/tutorial_code/core/file_input_output/file_input_output.cpp writeNum
    @end_toggle
    @add_toggle_python
    @snippet python/tutorial_code/core/file_input_output/file_input_output.py writeNum
    @end_toggle
    Reading in is a simple addressing (via the [] operator) and casting operation or a read via
    the \>\> operator. In Python, we address with getNode() and use real() :
    @add_toggle_cpp
    @snippet cpp/tutorial_code/core/file_input_output/file_input_output.cpp readNum
    @end_toggle
    @add_toggle_python
    @snippet cpp/tutorial_code/core/file_input_output/file_input_output.cpp readNum
    @end_toggle
-#  **Input/Output of OpenCV Data structures.** Well these behave exactly just as the basic C++
    and Python types:
    @add_toggle_cpp
    @snippet cpp/tutorial_code/core/file_input_output/file_input_output.cpp iomati
    @snippet cpp/tutorial_code/core/file_input_output/file_input_output.cpp iomatw
    @snippet cpp/tutorial_code/core/file_input_output/file_input_output.cpp iomat
    @end_toggle
    @add_toggle_python
    @snippet python/tutorial_code/core/file_input_output/file_input_output.py iomati
    @snippet python/tutorial_code/core/file_input_output/file_input_output.py iomatw
    @snippet python/tutorial_code/core/file_input_output/file_input_output.py iomat
    @end_toggle
-#  **Input/Output of vectors (arrays) and associative maps.** As I mentioned beforehand, we can
    output maps and sequences (array, vector) too. Again we first print the name of the variable and
    then we have to specify if our output is either a sequence or map.

    For sequence before the first element print the "[" character and after the last one the "]"
    character. With Python, call `FileStorage.startWriteStruct(structure_name, struct_type)`,
    where `struct_type` is `cv2.FileNode_MAP` or `cv2.FileNode_SEQ` to start writing the structure.
    Call `FileStorage.endWriteStruct()` to finish the structure:
    @add_toggle_cpp
    @snippet cpp/tutorial_code/core/file_input_output/file_input_output.cpp writeStr
    @end_toggle
    @add_toggle_python
    @snippet python/tutorial_code/core/file_input_output/file_input_output.py writeStr
    @end_toggle
    For maps the drill is the same however now we use the "{" and "}" delimiter characters:
    @add_toggle_cpp
    @snippet cpp/tutorial_code/core/file_input_output/file_input_output.cpp writeMap
    @end_toggle
    @add_toggle_python
    @snippet python/tutorial_code/core/file_input_output/file_input_output.py writeMap
    @end_toggle
    To read from these we use the @ref cv::FileNode and the @ref cv::FileNodeIterator data
    structures. The [] operator of the @ref cv::FileStorage class (or the getNode() function in Python) returns a @ref cv::FileNode data
    type. If the node is sequential we can use the @ref cv::FileNodeIterator to iterate through the
    items. In Python, the at() function can be used to address elements of the sequence and the
    size() function returns the length of the sequence:
    @add_toggle_cpp
    @snippet cpp/tutorial_code/core/file_input_output/file_input_output.cpp readStr
    @end_toggle
    @add_toggle_python
    @snippet python/tutorial_code/core/file_input_output/file_input_output.py readStr
    @end_toggle
    For maps you can use the [] operator (at() function in Python) again to access the given item (or the \>\> operator too):
    @add_toggle_cpp
    @snippet cpp/tutorial_code/core/file_input_output/file_input_output.cpp readMap
    @end_toggle
    @add_toggle_python
    @snippet python/tutorial_code/core/file_input_output/file_input_output.py readMap
    @end_toggle
-#  **Read and write your own data structures.** Suppose you have a data structure such as:
    @add_toggle_cpp
    @code{.cpp}
    class MyData
    {
    public:
          MyData() : A(0), X(0), id() {}
    public:   // Data Members
       int A;
       double X;
       string id;
    };
    @endcode
    @end_toggle
    @add_toggle_python
    @code{.py}
    class MyData:
        def __init__(self):
            self.A = self.X = 0
            self.name = ''
    @endcode
    @end_toggle
    In C++, it's possible to serialize this through the OpenCV I/O XML/YAML interface (just as
    in case of the OpenCV data structures) by adding a read and a write function inside and outside of your
    class. In Python, you can get close to this by implementing a read and write function inside
    the class. For the inside part:
    @add_toggle_cpp
    @snippet cpp/tutorial_code/core/file_input_output/file_input_output.cpp inside
    @end_toggle
    @add_toggle_python
    @snippet python/tutorial_code/core/file_input_output/file_input_output.py inside
    @end_toggle
    @add_toggle_cpp
    In C++, you need to add the following functions definitions outside the class:
    @snippet cpp/tutorial_code/core/file_input_output/file_input_output.cpp outside
    @end_toggle
    Here you can observe that in the read section we defined what happens if the user tries to read
    a non-existing node. In this case we just return the default initialization value, however a
    more verbose solution would be to return for instance a minus one value for an object ID.

    Once you added these four functions use the \>\> operator for write and the \<\< operator for
    read (or the defined input/output functions for Python):
    @add_toggle_cpp
    @snippet cpp/tutorial_code/core/file_input_output/file_input_output.cpp customIOi
    @snippet cpp/tutorial_code/core/file_input_output/file_input_output.cpp customIOw
    @snippet cpp/tutorial_code/core/file_input_output/file_input_output.cpp customIO
    @end_toggle
    @add_toggle_python
    @snippet python/tutorial_code/core/file_input_output/file_input_output.py customIOi
    @snippet python/tutorial_code/core/file_input_output/file_input_output.py customIOw
    @snippet python/tutorial_code/core/file_input_output/file_input_output.py customIO
    @end_toggle
    Or to try out reading a non-existing read:
    @add_toggle_cpp
    @snippet cpp/tutorial_code/core/file_input_output/file_input_output.cpp nonexist
    @end_toggle
    @add_toggle_python
    @snippet python/tutorial_code/core/file_input_output/file_input_output.py nonexist
    @end_toggle

Result
------

Well mostly we just print out the defined numbers. On the screen of your console you could see:
@code{.bash}
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
@endcode
Nevertheless, it's much more interesting what you may see in the output xml file:
@code{.xml}
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
@endcode
Or the YAML file:
@code{.yaml}
%YAML:1.0
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
@endcode
You may observe a runtime instance of this on the [YouTube
here](https://www.youtube.com/watch?v=A4yqVnByMMM) .

@youtube{A4yqVnByMMM}
