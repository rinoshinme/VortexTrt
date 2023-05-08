#pragma once

#include <vector>

namespace vortex
{
    class SemanticDecoder
    {
    private:
        int m_Width;
        int m_Height;
        int m_NumClasses;
        
    public:
        SemanticDecoder();
        void Forward(const std::vector<float>& data, std::vector<unsigned char>& output);
    };
}
