Command Line Parser
===================

.. highlight:: cpp

CommandLineParser
-----------------
.. ocv:class:: CommandLineParser

The CommandLineParser class is designed for command line arguments parsing


    .. ocv:function:: CommandLineParser::CommandLineParser( int argc, const char* const argv[], const String& keys )

        :param argc:
        :param argv:
        :param keys:

    .. ocv:function:: template<typename T> T CommandLineParser::get<T>(const String& name, bool space_delete = true)

        :param name:
        :param space_delete:

    .. ocv:function:: template<typename T> T CommandLineParser::get<T>(int index, bool space_delete = true)

        :param index:
        :param space_delete:

    .. ocv:function:: bool CommandLineParser::has(const String& name)

        :param name:

    .. ocv:function:: bool CommandLineParser::check()


    .. ocv:function:: void CommandLineParser::about( const String& message )

        :param message:

    .. ocv:function:: void CommandLineParser::printMessage()

    .. ocv:function:: void CommandLineParser::printErrors()

    .. ocv:function:: String CommandLineParser::getPathToApplication()


The sample below demonstrates how to use CommandLineParser:

::

    CommandLineParser parser(argc, argv, keys);
    parser.about("Application name v1.0.0");

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    int N = parser.get<int>("N");
    double fps = parser.get<double>("fps");
    String path = parser.get<String>("path");

    use_time_stamp = parser.has("timestamp");

    String img1 = parser.get<String>(0);
    String img2 = parser.get<String>(1);

    int repeat = parser.get<int>(2);

    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }

Syntax:

::

    const String keys =
        "{help h usage ? |      | print this message   }"
        "{@image1        |      | image1 for compare   }"
        "{@image2        |      | image2 for compare   }"
        "{@repeat        |1     | number               }"
        "{path           |.     | path to file         }"
        "{fps            | -1.0 | fps for output video }"
        "{N count        |100   | count of objects     }"
        "{ts timestamp   |      | use time stamp       }"
        ;

Use:

::

    # ./app -N=200 1.png 2.jpg 19 -ts

    # ./app -fps=aaa
    ERRORS:
    Exception: can not convert: [aaa] to [double]
