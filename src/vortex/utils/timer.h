#pragma once
#include <chrono>


namespace vortex
{
    class Timer
    {
    private:
        float m_Start;
        
    public:
        Timer();
        void Tick();
        float Tock();

    };
}
