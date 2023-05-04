#pragma once

#include <chrono>

namespace vortex
{
    class Timer
    {
    private:

    public:
        Timer();

        void Tick();
        float Tock();
        
        // elapsed time from creation
        float Elapsed();
    };

}
