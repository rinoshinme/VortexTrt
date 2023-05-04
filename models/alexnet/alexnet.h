#pragma once
#include "logging.h"

namespace vortex
{
    class AlexNet
    {
    private:
        Logger m_Logger;
        uint32_t m_Width;
        uint32_t m_Height;
        uint32_t m_Channels;
        uint32_t m_MaxBatchSize;
        std::string m_InputBlobName;
        std::string m_OutputBlobName;

        // trt related
        IBuilder* m_Builder;
        IBuilderConfig* m_Config;
        INetworkDefinition* m_Network;

    public:
        AlexNet(const std::string& weightsPath);

        void Serialize(const std::string& enginePath);
    private:
        void Build();

    };

}
