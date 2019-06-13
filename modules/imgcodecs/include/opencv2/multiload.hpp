// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 PLANET artificial intelligence GmbH

#ifndef OPENCV_IMGCODECS_MULTILOAD_HPP
#define OPENCV_IMGCODECS_MULTILOAD_HPP

#include "opencv2/imgcodecs.hpp"

namespace cv
{
    class BaseImageDecoder;

//! @addtogroup imgcodecs
//! @{

    /** @brief Load pages of multi-images one by one.
     *
     * Objects of this class keep data to access single pages of multi-images one by one.
     * This enables handling multi-images with big numbers of pages, which can not
     * be handled using cv::imreadmulti.
     *
     * Set an image source using cv::MultiLoad::read or cv::MultiLoad::decode first and
     * access the pages of the multi-image using the other methods afterwards.
     *
     * @sa cv::imreadmulti
     */
    class CV_EXPORTS MultiLoad
    {
    public:
        //! Initialize loader.
        MultiLoad();

        //! @see cv::MultiLoad::clear
        ~MultiLoad();

        //! Clear temporary files, buffers, decoders etc.
        void clear();

        /** @brief Start reading images from a file.
         *
         * Reads basic informations from an image-file for further processing.
         *
         * @param filename Name of file to be loaded.
         * @param flags flags to control drivers, recognized are cv::IMREAD_UNCHANGED and cv::IMREAD_LOAD_GDAL, default cv::IMREAD_UNCHANGED
         * @sa cv:imread
         */
        bool read(const String &filename, int flags = IMREAD_UNCHANGED);

        /** @brief Start decoding images from a buffer.
         *
         * Reads basic information from an image-buffer for further processing.
         *
         * @param buf Input array or vector of bytes.
         * @param flags The same flags as in cv::MultiLoad::read
         * @sa cv:imdecode
         */
        bool decode(InputArray buf, int flags = IMREAD_UNCHANGED);

        //! No images available.
        bool empty() const;

        //! Total number of images.
        std::size_t size() const;

        //! Will cv::MultiLoad::next return a valid image.
        bool hasNext() const;

        /** @brief Return the next page.
         *
         * Returns the pages of a multi-image sequentially.
         *
         * @param flags The same flags as in cv::imread except drivers, see cv::ImreadModes.
         * @param dst The optional output placeholder for the loaded/decoded matrix, see cv::imdecode.
         * @return the next page of a multi-image of an empty matrix
         * @sa cv:imread
         */
        Mat next(int flags = IMREAD_COLOR, Mat *dst = 0);

        /** @brief Return an arbitrary page.
         *
         * Returns a selected page of a multi-image. The next call of cv::MultiLoad::next will return
         * the page following this.
         *
         * @param idx index of page to load (0 ... size - 1)
         * @param flags The same flags as in cv::imread except drivers, see cv::ImreadModes.
         * @param dst The optional output placeholder for the loaded/decoded matrix, see cv::imdecode.
         * @return the next page of a multi-image of an empty matrix
         * @sa cv:imread
         */
        Mat at(int idx, int flags = IMREAD_COLOR, Mat *dst = 0);

    private:
        String m_file;
        Mat m_buf;
        String m_tempfile;
        Ptr <BaseImageDecoder> m_decoder;
        bool m_has_next;

        //! no copy
        MultiLoad(const MultiLoad &);

        //! no assign
        MultiLoad &operator=(const MultiLoad &);

        //! filename XOR buf
        //! @see cv::MultiLoad::read
        //! @see cv::MultiLoad::decode
        bool load(const String *filename, const _InputArray *buf, int flags);
    };

//! @} imgcodecs

} // cv

#endif /*OPENCV_IMGCODECS_MULTILOAD_HPP*/
