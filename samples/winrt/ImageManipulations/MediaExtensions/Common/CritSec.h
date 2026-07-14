#pragma once

//////////////////////////////////////////////////////////////////////////
//  CritSec
//  Description: Wraps a critical section.
//////////////////////////////////////////////////////////////////////////

class CritSec
{
public:
    CRITICAL_SECTION m_criticalSection;
public:
    CritSec()
    {
        InitializeCriticalSectionEx(&m_criticalSection, 100, 0);
    }

    ~CritSec()
    {
        DeleteCriticalSection(&m_criticalSection);
    }

    _Acquires_lock_(m_criticalSection)
    void Lock()
    {
        EnterCriticalSection(&m_criticalSection);
    }

    _Releases_lock_(m_criticalSection)
    void Unlock()
    {
        LeaveCriticalSection(&m_criticalSection);
    }
};


//////////////////////////////////////////////////////////////////////////
//  AutoLock
//  Description: Provides automatic locking and unlocking of a
//               of a critical section.
//
//  Note: The AutoLock object must go out of scope before the CritSec.
//////////////////////////////////////////////////////////////////////////

class AutoLock
{
private:
    CritSec *m_pCriticalSection;
public:
    _Acquires_lock_(m_pCriticalSection)
    AutoLock(CritSec& crit)
    {
        m_pCriticalSection = &crit;
        m_pCriticalSection->Lock();
    }

    _Releases_lock_(m_pCriticalSection)
    ~AutoLock()
    {
        m_pCriticalSection->Unlock();
    }
};
