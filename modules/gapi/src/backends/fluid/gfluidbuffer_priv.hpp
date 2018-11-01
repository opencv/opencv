// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_FLUID_BUFFER_PRIV_HPP
#define OPENCV_GAPI_FLUID_BUFFER_PRIV_HPP

#include <vector>

#include "opencv2/gapi/fluid/gfluidbuffer.hpp"
#include "opencv2/gapi/own/exports.hpp" // GAPI_EXPORTS

namespace cv {
namespace gapi {
namespace fluid {

class BufferStorageWithBorder;

class BorderHandler
{
protected:
    int m_border_size;

public:
    BorderHandler(int border_size);
    virtual ~BorderHandler() = default;
    virtual const uint8_t* inLineB(int log_idx, const BufferStorageWithBorder &data, int desc_height) const = 0;

    // Fills border pixels after buffer allocation (if possible (for const border))
    inline virtual void fillCompileTimeBorder(BufferStorageWithBorder &) { /* nothing */ }

    // Fills required border lines
    inline virtual void updateBorderPixels(BufferStorageWithBorder& /*data*/, int /*startLine*/, int /*lpi*/) const { /* nothing */ }

    inline int borderSize() const { return m_border_size; }
    inline virtual std::size_t size() const { return 0; }
};

template<int BorderType>
class BorderHandlerT : public BorderHandler
{
    std::function<void(uint8_t*,int,int,int)> m_fill_border_row;
public:
    BorderHandlerT(int border_size, int data_type);
    virtual void updateBorderPixels(BufferStorageWithBorder& data, int startLine, int lpi) const override;
    virtual const uint8_t* inLineB(int log_idx, const BufferStorageWithBorder &data, int desc_height) const override;
};

template<>
class BorderHandlerT<cv::BORDER_CONSTANT> : public BorderHandler
{
    cv::gapi::own::Scalar m_border_value;
    cv::gapi::own::Mat m_const_border;

public:
    BorderHandlerT(int border_size, cv::gapi::own::Scalar border_value);
    virtual const uint8_t* inLineB(int log_idx, const BufferStorageWithBorder &data, int desc_height) const override;
    virtual void fillCompileTimeBorder(BufferStorageWithBorder &) override;
    virtual std::size_t size() const override;
};

class BufferStorage
{
protected:
    cv::gapi::own::Mat m_data;

public:
    virtual void copyTo(BufferStorageWithBorder &dst, int startLine, int nLines) const = 0;

    virtual ~BufferStorage() = default;

    virtual const uint8_t* ptr(int idx) const = 0;
    virtual       uint8_t* ptr(int idx) = 0;

    inline bool empty() const { return m_data.empty(); }

    inline const cv::gapi::own::Mat& data() const { return m_data; }
    inline       cv::gapi::own::Mat& data()       { return m_data; }

    inline int rows() const { return m_data.rows; }
    inline int cols() const { return m_data.cols; }
    inline int type() const { return m_data.type(); }

    virtual const uint8_t* inLineB(int log_idx, int desc_height) const = 0;

    // FIXME? remember parent and remove src parameter?
    virtual void updateBeforeRead(int startLine, int nLines, const BufferStorage& src) = 0;
    virtual void updateAfterWrite(int startLine, int nLines) = 0;

    virtual int physIdx(int logIdx) const = 0;

    virtual size_t size() const = 0;
};

class BufferStorageWithoutBorder final : public BufferStorage
{
    bool m_is_virtual = true;
    cv::gapi::own::Rect m_roi;

public:
    virtual void copyTo(BufferStorageWithBorder &dst, int startLine, int nLines) const override;

    inline virtual const uint8_t* ptr(int idx) const override
    {
        GAPI_DbgAssert((m_is_virtual && m_roi == cv::gapi::own::Rect{}) || (!m_is_virtual && m_roi != cv::gapi::own::Rect{}));
        return m_data.ptr(physIdx(idx), 0);
    }
    inline virtual uint8_t* ptr(int idx) override
    {
        GAPI_DbgAssert((m_is_virtual && m_roi == cv::gapi::own::Rect{}) || (!m_is_virtual && m_roi != cv::gapi::own::Rect{}));
        return m_data.ptr(physIdx(idx), 0);
    }

    inline void attach(const cv::gapi::own::Mat& _data, cv::gapi::own::Rect _roi)
    {
        m_data = _data(_roi);
        m_roi = _roi;
        m_is_virtual = false;
    }

    void create(int capacity, int desc_width, int type);

    inline virtual const uint8_t* inLineB(int log_idx, int desc_height) const override;

    virtual void updateBeforeRead(int startLine, int nLines, const BufferStorage& src) override;
    virtual void updateAfterWrite(int startLine, int nLines) override;

    inline virtual int physIdx(int logIdx) const override { return (logIdx - m_roi.y) % m_data.rows; }

    virtual size_t size() const override;
};

class BufferStorageWithBorder final: public BufferStorage
{
    std::unique_ptr<BorderHandler> m_borderHandler;

public:
    inline int borderSize() const { return m_borderHandler->borderSize(); }

    virtual void copyTo(BufferStorageWithBorder &dst, int startLine, int nLines) const override;

    inline virtual const uint8_t* ptr(int idx) const override
    {
        return m_data.ptr(physIdx(idx), borderSize());
    }
    inline virtual uint8_t* ptr(int idx) override
    {
        return m_data.ptr(physIdx(idx), borderSize());
    }

    void init(int depth, int border_size, Border border);
    void create(int capacity, int desc_width, int dtype);

    virtual const uint8_t* inLineB(int log_idx, int desc_height) const override;

    virtual void updateBeforeRead(int startLine, int nLines, const BufferStorage &src) override;
    virtual void updateAfterWrite(int startLine, int nLines) override;

    inline virtual int physIdx(int logIdx) const override { return logIdx % m_data.rows; }

    virtual size_t size() const override;
};

// FIXME: GAPI_EXPORTS is used here only to access internal methods
// like readDone/writeDone in low-level tests
class GAPI_EXPORTS View::Priv
{
    friend class View;
protected:
    const Buffer *m_p           = nullptr; // FIXME replace with weak_ptr
    int           m_read_caret  = -1;
    int           m_lines_next_iter = -1;
    int m_border_size = -1;

public:
    virtual ~Priv() = default;
    // API used by actors/backend

    virtual void allocate(int lineConsumption, BorderOpt border) = 0;
    virtual void prepareToRead() = 0;

    void readDone(int linesRead, int linesForNextIteration);
    void reset(int linesForFirstIteration);

    virtual std::size_t size() const = 0;

    // Does the view have enough unread lines for next iteration
    bool ready() const;

    // API used (indirectly) by user code
    virtual const uint8_t* InLineB(int index) const = 0;
};

class ViewPrivWithoutOwnBorder final : public View::Priv
{
public:
    // API used by actors/backend
    ViewPrivWithoutOwnBorder(const Buffer *p, int borderSize);

    inline virtual void allocate(int, BorderOpt) override { /* nothing */ }
    inline virtual void prepareToRead() override { /* nothing */ }

    inline virtual std::size_t size() const override { return 0; }

    // API used (indirectly) by user code
    virtual const uint8_t* InLineB(int index) const override;
};

class ViewPrivWithOwnBorder final : public View::Priv
{
    BufferStorageWithBorder m_own_storage;

public:
    // API used by actors/backend
    ViewPrivWithOwnBorder(const Buffer *p, int borderSize);

    inline virtual void allocate(int lineConsumption, BorderOpt border) override;
    virtual void prepareToRead() override;
    virtual std::size_t size() const override;

    // API used (indirectly) by user code
    virtual const uint8_t* InLineB(int index) const override;
};

void debugBufferPriv(const Buffer& buffer, std::ostream &os);

// FIXME: GAPI_EXPORTS is used here only to access internal methods
// like readDone/writeDone in low-level tests
class GAPI_EXPORTS Buffer::Priv
{
    int m_writer_lpi       =  1;

    cv::GMatDesc m_desc    = cv::GMatDesc{-1,-1,{-1,-1}};
    bool m_is_input        = false;

    int m_write_caret      = -1;

    std::vector<View> m_views;

    std::unique_ptr<BufferStorage> m_storage;

    // Coordinate starting from which this buffer is assumed
    // to be read (with border not being taken into account)
    int m_readStart;
    cv::gapi::own::Rect m_roi;

    friend void debugBufferPriv(const Buffer& p, std::ostream &os);

public:
    Priv() = default;
    Priv(int read_start, cv::gapi::own::Rect roi);

    inline const BufferStorage& storage() const { return *m_storage.get(); }

    // API used by actors/backend
    void init(const cv::GMatDesc &desc,
              int writer_lpi,
              int readStart,
              cv::gapi::own::Rect roi);

    void allocate(BorderOpt border, int border_size, int line_consumption, int skew);
    void bindTo(const cv::gapi::own::Mat &data, bool is_input);

    inline void addView(const View& view) { m_views.push_back(view); }

    inline const GMatDesc& meta() const { return m_desc; }

    bool full() const;
    void writeDone();
    void reset();
    int size() const;

    int linesReady() const;

    inline int y() const { return m_write_caret; }

    inline int writer_lpi()     const { return m_writer_lpi; }

    // API used (indirectly) by user code
    uint8_t* OutLineB(int index = 0);
    int lpi() const;

    inline int readStart()   const { return m_readStart; }
    inline int writeStart()  const { return m_roi.y; }
    inline int writeEnd()    const { return m_roi.y + m_roi.height; }
    inline int outputLines() const { return m_roi.height; }
};

} // namespace cv::gapi::fluid
} // namespace cv::gapi
} // namespace cv

#endif // OPENCV_GAPI_FLUID_BUFFER_PRIV_HPP
