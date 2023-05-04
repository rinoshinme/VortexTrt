#pragma once

#include <iostream>
#include "NvInfer.h"

namespace vortex
{
    class Logger : public nvinfer1::ILogger
    {
    public:
        void log(Severity severity, const char* msg) noexcept override
        {
            // suppress info-level message
            if (severity != Severity::kINFO)
            {
                std::cout << msg << std::endl;
            }
        }
    };
}
