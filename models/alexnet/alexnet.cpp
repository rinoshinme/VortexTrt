#include "alexnet.h"


namespace vortex
{
    AlexNet::AlexNet()
    {
        m_Builder = createInferBuilder(m_Logger);
        m_Config = m_Builder->createBuilderConfig();
    }

    void AlexNet::Build()
    {
        // build network
        m_Network = m_Builder->createNetworkV2(0U);
        // create input tensor
        ITensor* data = network->addInput()
        
    }

}
