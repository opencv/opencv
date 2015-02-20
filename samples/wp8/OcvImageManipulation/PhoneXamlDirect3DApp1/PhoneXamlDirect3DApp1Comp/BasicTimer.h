#pragma once

#include <wrl.h>

// Helper class for basic timing.
ref class BasicTimer sealed
{
public:
    // Initializes internal timer values.
    BasicTimer()
    {
        if (!QueryPerformanceFrequency(&m_frequency))
        {
            throw ref new Platform::FailureException();
        }
        Reset();
    }

    // Reset the timer to initial values.
    void Reset()
    {
        Update();
        m_startTime = m_currentTime;
        m_total = 0.0f;
        m_delta = 1.0f / 60.0f;
    }

    // Update the timer's internal values.
    void Update()
    {
        if (!QueryPerformanceCounter(&m_currentTime))
        {
            throw ref new Platform::FailureException();
        }

        m_total = static_cast<float>(
            static_cast<double>(m_currentTime.QuadPart - m_startTime.QuadPart) /
            static_cast<double>(m_frequency.QuadPart)
            );

        if (m_lastTime.QuadPart == m_startTime.QuadPart)
        {
            // If the timer was just reset, report a time delta equivalent to 60Hz frame time.
            m_delta = 1.0f / 60.0f;
        }
        else
        {
            m_delta = static_cast<float>(
                static_cast<double>(m_currentTime.QuadPart - m_lastTime.QuadPart) /
                static_cast<double>(m_frequency.QuadPart)
                );
        }

        m_lastTime = m_currentTime;
    }

    // Duration in seconds between the last call to Reset() and the last call to Update().
    property float Total
    {
        float get() { return m_total; }
    }

    // Duration in seconds between the previous two calls to Update().
    property float Delta
    {
        float get() { return m_delta; }
    }

private:
    LARGE_INTEGER m_frequency;
    LARGE_INTEGER m_currentTime;
    LARGE_INTEGER m_startTime;
    LARGE_INTEGER m_lastTime;
    float m_total;
    float m_delta;
};
