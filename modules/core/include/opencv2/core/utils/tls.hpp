// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_UTILS_TLS_HPP
#define OPENCV_UTILS_TLS_HPP

#ifndef OPENCV_CORE_UTILITY_H
#error "tls.hpp must be included after opencv2/core/utility.hpp or opencv2/core.hpp"
#endif

namespace cv {

//! @addtogroup core_utils
//! @{

namespace details { class TlsStorage; }

/** TLS container base implementation
 *
 * Don't use directly.
 *
 * @sa TLSData, TLSDataAccumulator templates
 */
class CV_EXPORTS TLSDataContainer
{
protected:
    TLSDataContainer();
    virtual ~TLSDataContainer();

    /// @deprecated use detachData() instead
    void  gatherData(std::vector<void*> &data) const;
    /// get TLS data and detach all data from threads (similar to cleanup() call)
    void  detachData(std::vector<void*>& data);

    void* getData() const;
    void  release();

protected:
    virtual void* createDataInstance() const = 0;
    virtual void  deleteDataInstance(void* pData) const = 0;

private:
    int key_;

    friend class cv::details::TlsStorage;  // core/src/system.cpp

public:
    void cleanup(); //!< Release created TLS data container objects. It is similar to release() call, but it keeps TLS container valid.

private:
    // Disable copy/assign (noncopyable pattern)
    TLSDataContainer(TLSDataContainer &) = delete;
    TLSDataContainer& operator =(const TLSDataContainer &) = delete;
};


/** @brief Simple TLS data class
 *
 * @sa TLSDataAccumulator
 */
template <typename T>
class TLSData : protected TLSDataContainer
{
public:
    inline TLSData() {}
    inline ~TLSData() { release(); }

    inline T* get() const   { return (T*)getData(); }  //!< Get data associated with key
    inline T& getRef() const { T* ptr = (T*)getData(); CV_DbgAssert(ptr); return *ptr; }  //!< Get data associated with key

    /// Release associated thread data
    inline void cleanup()
    {
        TLSDataContainer::cleanup();
    }

protected:
    /// Wrapper to allocate data by template
    virtual void* createDataInstance() const CV_OVERRIDE { return new T; }
    /// Wrapper to release data by template
    virtual void  deleteDataInstance(void* pData) const CV_OVERRIDE { delete (T*)pData; }
};


/// TLS data accumulator with gathering methods
template <typename T>
class TLSDataAccumulator : public TLSData<T>
{
    mutable cv::Mutex mutex;
    mutable std::vector<T*> dataFromTerminatedThreads;
    std::vector<T*> detachedData;
    bool cleanupMode;
public:
    TLSDataAccumulator() : cleanupMode(false) {}
    ~TLSDataAccumulator()
    {
        release();
    }

    /** @brief Get data from all threads
     * @deprecated replaced by detachData()
     *
     * Lifetime of vector data is valid until next detachData()/cleanup()/release() calls
     *
     * @param[out] data result buffer (should be empty)
     */
    void gather(std::vector<T*> &data) const
    {
        CV_Assert(cleanupMode == false);  // state is not valid
        CV_Assert(data.empty());
        {
            std::vector<void*> &dataVoid = reinterpret_cast<std::vector<void*>&>(data);
            TLSDataContainer::gatherData(dataVoid);
        }
        {
            AutoLock lock(mutex);
            data.reserve(data.size() + dataFromTerminatedThreads.size());
            for (typename std::vector<T*>::const_iterator i = dataFromTerminatedThreads.begin(); i != dataFromTerminatedThreads.end(); ++i)
            {
                data.push_back((T*)*i);
            }
        }
    }

    /** @brief Get and detach data from all threads
     *
     * Call cleanupDetachedData() when returned vector is not needed anymore.
     *
     * @return Vector with associated data. Content is preserved (including lifetime of attached data pointers) until next detachData()/cleanupDetachedData()/cleanup()/release() calls
     */
    std::vector<T*>& detachData()
    {
        CV_Assert(cleanupMode == false);  // state is not valid
        std::vector<void*> dataVoid;
        {
            TLSDataContainer::detachData(dataVoid);
        }
        {
            AutoLock lock(mutex);
            detachedData.reserve(dataVoid.size() + dataFromTerminatedThreads.size());
            for (typename std::vector<T*>::const_iterator i = dataFromTerminatedThreads.begin(); i != dataFromTerminatedThreads.end(); ++i)
            {
                detachedData.push_back((T*)*i);
            }
            dataFromTerminatedThreads.clear();
            for (typename std::vector<void*>::const_iterator i = dataVoid.begin(); i != dataVoid.end(); ++i)
            {
                detachedData.push_back((T*)(void*)*i);
            }
        }
        dataVoid.clear();
        return detachedData;
    }

    /// Release associated thread data returned by detachData() call
    void cleanupDetachedData()
    {
        AutoLock lock(mutex);
        cleanupMode = true;
        _cleanupDetachedData();
        cleanupMode = false;
    }

    /// Release associated thread data
    void cleanup()
    {
        cleanupMode = true;
        TLSDataContainer::cleanup();

        AutoLock lock(mutex);
        _cleanupDetachedData();
        _cleanupTerminatedData();
        cleanupMode = false;
    }

    /// Release associated thread data and free TLS key
    void release()
    {
        cleanupMode = true;
        TLSDataContainer::release();
        {
            AutoLock lock(mutex);
            _cleanupDetachedData();
            _cleanupTerminatedData();
        }
    }

protected:
    // synchronized
    void _cleanupDetachedData()
    {
        for (typename std::vector<T*>::iterator i = detachedData.begin(); i != detachedData.end(); ++i)
        {
            deleteDataInstance((T*)*i);
        }
        detachedData.clear();
    }

    // synchronized
    void _cleanupTerminatedData()
    {
        for (typename std::vector<T*>::iterator i = dataFromTerminatedThreads.begin(); i != dataFromTerminatedThreads.end(); ++i)
        {
            deleteDataInstance((T*)*i);
        }
        dataFromTerminatedThreads.clear();
    }

protected:
    virtual void* createDataInstance() const CV_OVERRIDE
    {
        // Note: we can collect all allocated data here, but this would require raced mutex locks
        return new T;
    }
    virtual void  deleteDataInstance(void* pData) const CV_OVERRIDE
    {
        if (cleanupMode)
        {
            delete (T*)pData;
        }
        else
        {
            AutoLock lock(mutex);
            dataFromTerminatedThreads.push_back((T*)pData);
        }
    }
};


//! @}

} // namespace

#endif // OPENCV_UTILS_TLS_HPP
