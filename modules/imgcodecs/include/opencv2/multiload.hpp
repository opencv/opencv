// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 PLANET artificial intelligence GmbH

#ifndef OPENCV_IMGCODECS_MULTILOAD_HPP
#define OPENCV_IMGCODECS_MULTILOAD_HPP

#include "opencv2/imgcodecs.hpp"
#include <map>

namespace cv
{
    class BaseImageDecoder;

//! @addtogroup imgcodecs
//! @{

    /** @brief Load pages of multi-images one by one.
     *
     * Objects of this class keep data to access single pages of multi-images one by one.
     * This enables handling multi-images with big numbers of pages, which can not
     * be handled using imreadmulti().
     *
     * Set an image source using read() or decode() first and
     * access the pages of the multi-image using the other methods afterwards.
     *
     * @sa imreadmulti(), imread(), imdecode()
     */
    class CV_EXPORTS MultiLoad
    {
    public:
        class iterator {
        public:
            Mat operator*() const { return **m_parent; }
            bool operator==(const iterator &other) const;
            bool operator!=(const iterator &other) const { return !(*this == other); }
            void operator++() { m_parent->next(); }
        private:
            friend class MultiLoad;
            MultiLoad *m_parent;
            iterator(MultiLoad *parent) : m_parent(parent) {}
        };

        /** @brief Initialize loader.
         *
         * @param default_flags flags to use if not overwritten
         */
        MultiLoad(int default_flags = IMREAD_COLOR);

        /// @sa clear()
        ~MultiLoad();

        /// iterate over pages: begin
        iterator begin() { return iterator(this); }

        /// iterate over pages: end
        const iterator& end() const;

        //! Clear temporary files, buffers, decoders etc.
        void clear();

        /** @brief Start reading images from a file.
         *
         * Reads basic informations from an image-file for further processing.
         *
         * @param filename Name of file to be loaded.
         * @param flags flags to control drivers (e.g. cv::IMREAD_LOAD_GDAL)
         * @sa imread()
         */
        bool read(const String &filename, int flags);

        /** @brief Start reading images from a file using default-flags.
         *
         * Reads basic informations from an image-file for further processing.
         *
         * @param filename Name of file to be loaded.
         * @sa read(const String&, int)
         */
        bool read(const String &filename) { return read(filename, m_default_flags); }

        /** @brief Start decoding images from a buffer.
         *
         * Reads basic information from an image-buffer for further processing.
         *
         * @param buf Input array or vector of bytes.
         * @param flags The same flags as in read()
         * @sa imdecode()
         */
        bool decode(InputArray buf, int flags);

        /** @brief Start decoding images from a buffer using default-flags.
         *
         * Reads basic information from an image-buffer for further processing.
         *
         * @param buf Input array or vector of bytes.
         * @sa decode(InputArray,int)
         */
        bool decode(InputArray buf) { return decode(buf, m_default_flags); }

        /// Total number of images.
        std::size_t size() const;

        /// No images available.
        bool empty() const { return size() == 0; }

        /// Will current() return a valid image.
        bool valid() const;

        /// @copydoc valid()
        operator bool() const { return valid(); }

        /** @brief Return an arbitrary page.
         *
         * Returns a selected page of a multi-image.
         *
         * @param idx index of page to load (0 ... size() - 1)
         * @param flags The same flags as in imread() except drivers, see cv::ImreadModes.
         * @param properties additional properties like "dpi-x", "dpi-y", "document-name", "page-name", "page-number"  or TIFF/EXIF-tags in string-form.
         * @param dst The optional output placeholder for the loaded/decoded matrix, see imdecode().
         * @return the requested page of a multi-image or an empty matrix
         * @sa imread()
         */
        Mat at(int idx, int flags, std::map<String, String> *properties = 0, Mat *dst = 0) const;

        /** @brief Return an arbitrary page.
         *
         * Returns a selected page of a multi-image using the default-flags.
         *
         * @param idx index of page to load (0 ... size() - 1)
         * @param properties additional properties like "dpi-x", "dpi-y", "document-name", "page-name", "page-number"  or TIFF/EXIF-tags in string-form.
         * @param dst The optional output placeholder for the loaded/decoded matrix, see imdecode().
         * @return the requested page of a multi-image or an empty matrix
         * @sa at(int,int,Mat*) const
         */
        Mat at(int idx, std::map<String, String> *properties = 0, Mat *dst = 0) const { return at(idx, m_default_flags, properties, dst); }

        /** @brief Return an arbitrary page.
         *
         * Returns a selected page of a multi-image using the default-flags.
         *
         * @param idx index of page to load (0 ... size() - 1)
         * @return the requested page of a multi-image or an empty matrix
         * @sa at(int,Mat*) const
         */
        Mat operator[](int idx) const { return at(idx); }

        /** @brief Return the current page.
         *
         * Return the page set by advance() or next().
         *
         * @param flags The same flags as in imread() except drivers, see cv::ImreadModes.
         * @param properties additional properties like "dpi-x", "dpi-y", "document-name", "page-name", "page-number"  or TIFF/EXIF-tags in string-form.
         * @param dst The optional output placeholder for the loaded/decoded matrix, see imdecode().
         * @return the current page of a multi-image or an empty matrix
         * @sa imread()
         */
        Mat current(int flags, std::map<String, String> *properties = 0, Mat *dst = 0) const;

        /** @brief Return the current page.
         *
         * Return the page set by advance() or next() using default-flags.
         *
         * @param properties additional properties like "dpi-x", "dpi-y", "document-name", "page-name", "page-number"  or TIFF/EXIF-tags in string-form.
         * @param dst The optional output placeholder for the loaded/decoded matrix, see imdecode().
         * @return the current page of a multi-image or an empty matrix
         * @sa current(int,Mat*) const
         */
        Mat current(std::map<String, String> *properties = 0, Mat *dst = 0) const { return current(m_default_flags, properties, dst); }

        /** @brief Return the current page.
         *
         * Return the page set by advance() or next() using default-flags.
         *
         * @return the current page of a multi-image or an empty matrix
         * @sa current(Mat*) const
         */
        Mat operator*() const { return current(); }

        /**
         * @brief Move forward to the next page.
         *
         * Make the next page of a multi-image the current one.
         *
         * @return page available
         */
        bool next();

        /// @copydoc next()
        bool operator++() { return next(); }

        /// @copydoc next()
        bool operator++(int) { return next(); }

    private:
        String m_file;
        Mat m_buf;
        String m_tempfile;
        Ptr <BaseImageDecoder> m_decoder;
        int m_default_flags;
        bool m_has_current;

        /// no copy because of @c m_tempfile
        MultiLoad(const MultiLoad &);

        /// no assign because of @c m_tempfile
        MultiLoad &operator=(const MultiLoad &);

        /// called by read() and decode(): @c filename XOR @c buf
        /// @param filename name of file
        /// @param buf array with data
        /// @param flags flags to control drivers (e.g. cv::IMREAD_LOAD_GDAL)
        /// @see read()
        /// @see decode()
        bool load(const String *filename, const _InputArray *buf, int flags);
    };

//! @} imgcodecs

} // cv

#endif /*OPENCV_IMGCODECS_MULTILOAD_HPP*/
