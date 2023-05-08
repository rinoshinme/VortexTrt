#pragma once
#include <chrono>
#include <vector>


namespace vortex
{
    // single spot timer
    class Timer
    {
    private:
        std::chrono::time_point m_Start;
        
    public:
        Timer() {}

        void Start()
        {
            m_Start = std::chrono::steady_clock::now();
        }

        float Stop()
        {
            auto end = std::chrono::steady_clock::now();
            auto duration = end - m_Start;
            return duration.count();
        }
    };

    // multiple step timer
    class StopWatch
    {
    private:
        std::vector<float> m_Steps;
        std::chrono::time_point m_Prev;

    public:
        StopWatch() {}
        void Start()
        {
            m_Prev = std::chrono::steady_clock::now();
            m_Steps.clear();
        }

        void Step()
        {
            auto current = std::chrono::steady_clock::now();
            auto duration = current - m_Prev;
            m_Steps.push_back(duration.count());
            m_Prev = current;
        }

        float Stop()
        {
            auto current = std::chrono::steady_clock::now();
            auto duration = current - m_Prev;
            m_Steps.push_back(duration.count());
            // get mean duration
            float sum = 0;
            for (float v : m_Steps)
                sum += v;
            float avg = sum / m_Steps.size();
            m_Steps.clear();
            return avg;
        }

    };
}
