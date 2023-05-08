#pragma once

#include <vector>

namespace vortex
{
    class SoftmaxProb
    {
    private:
        int m_NumClasses;

    public:
        SoftmaxProb(int num_classes);

        void Forward(const std::vector<float>& data, std::vector<float>& probs);
    };
}
