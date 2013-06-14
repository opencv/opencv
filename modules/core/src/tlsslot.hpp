/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, NVIDIA CORPORATION, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __OPENCV_CORE_TLSSLOT_HPP__
#define __OPENCV_CORE_TLSSLOT_HPP__

#ifdef _WIN32

#include <list>
#include <utility>

namespace cv
{

#ifdef WINCE
#   define TLS_OUT_OF_INDEXES ((DWORD)0xFFFFFFFF)
#endif

template <typename _ValueType>
class TlsSlot
{
    typedef std::pair<HANDLE, _ValueType *> TlsListEntry;
    typedef std::list<TlsListEntry> TlsList;

public:
    TlsSlot()
    {
        hMutex = CreateMutex(NULL, FALSE, NULL);
        CV_Assert(hMutex != NULL);
        tlsKey = TlsAlloc();
        if (tlsKey == TLS_OUT_OF_INDEXES)
            CloseHandle(hMutex);
        CV_Assert(tlsKey != TLS_OUT_OF_INDEXES);
    }

    ~TlsSlot()
    {
        // Safe cleanup of the (static) TlsSlot object. The threads may still be active.
        while (true)
        {
            WaitForSingleObject(hMutex, INFINITE);
            clearTlsList();
            if (tlsList.empty())
                break;
            ReleaseMutex(hMutex);
            WaitForSingleObject(tlsList.back().first, INFINITE);
        }
        TlsFree(tlsKey);
        CloseHandle(hMutex);
    }

    _ValueType &data()
    {
        DWORD waitResult = WaitForSingleObject(hMutex, INFINITE);
        CV_Assert(waitResult != WAIT_FAILED); // Has the TlsSlot object been destructed?
        _ValueType *pData = (_ValueType *)TlsGetValue(tlsKey);
        if (!pData)
        {
            HANDLE hThread = NULL;
            try
            {
                pData = new _ValueType;
                TlsSetValue(tlsKey, pData);
                DuplicateHandle(GetCurrentProcess(), GetCurrentThread(),
                                GetCurrentProcess(), &hThread, 0, FALSE,
                                DUPLICATE_SAME_ACCESS); // Get the real thread handle.
                clearTlsList(); // Remove unused tlsList entries.
                tlsList.push_back(TlsListEntry(hThread, pData));
            }
            catch(...)
            {
                delete pData;
                if (hThread != NULL)
                    CloseHandle(hThread);
                ReleaseMutex(hMutex);
                throw;
            }
        }
        ReleaseMutex(hMutex);
        return *pData;
    }

private:
    TlsSlot(TlsSlot const &); // Not implemented.
    TlsSlot &operator =(TlsSlot const &); // Not implemented.

    void clearTlsList()
    {
        TlsList::iterator it = tlsList.begin();
        while (it != tlsList.end())
        {
            DWORD exitCode = 0;
            GetExitCodeThread(it->first, &exitCode);
            if (exitCode == STILL_ACTIVE)
                ++it;
            else
            {
                CloseHandle(it->first);
                delete it->second;
                it = tlsList.erase(it);
            }
        }
    }

    HANDLE hMutex;
    DWORD tlsKey;
    TlsList tlsList;
};

}

#endif /*_WIN32*/

#endif /*__OPENCV_CORE_TLSSLOT_HPP__*/
