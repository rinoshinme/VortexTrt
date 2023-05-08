#include "semantic_decoder.h"


namespace vortex
{
    SemanticDecoder::SemanticDecoder()
    {
        m_Width = 512;
        m_Height = 512;
        m_NumClasses = 2;
    }

    void SemanticDecoder::Forward(const std::vector<float>& data, std::vector<unsigned char>& output)
    {
        // assert(false);
    }
}
