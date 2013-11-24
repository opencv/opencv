package org.opencv.engine;

/**
* Class provides Java interface to OpenCV Engine Service. Is synchronious with native OpenCVEngine class.
*/
interface OpenCVEngineInterface
{
    /**
    * @return Return service version
    */
    int getEngineVersion();

    /**
    * Find installed OpenCV library
    * @param OpenCV version
    * @return Returns path to OpenCV native libs or empty string if OpenCV was not found
    */
    String getLibPathByVersion(String version);

    /**
    * Try to install defined version of OpenCV from Google Play (Android Market).
    * @param OpenCV version
    * @return Returns true if installation was successful or OpenCV package has been already installed
    */
    boolean installVersion(String version);

    /**
    * Return list of libraries in loading order seporated by ";" symbol
    * @param OpenCV version
    * @return Returns OpenCV libraries names seporated by symbol ";" in loading order
    */
    String getLibraryList(String version);
}
