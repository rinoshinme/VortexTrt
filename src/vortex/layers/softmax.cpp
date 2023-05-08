#include "softmax_prob.h"
#include <cmath>

namespace vortex
{
    SoftmaxProb::SoftmaxProb(int num_classes)
    {
        m_NumClasses = num_classes;
    }

    void SoftmaxProb::Forward(const std::vector<float>& data, std::vector<float>& probs)
    {
        // calculate softmax
        probs.resize(data.size());
        float exp_sum = 0.0;
        for (size_t i = 0; i < data.size(); ++i)
        {
            float val = std::exp(data[i]);
            probs.push_back(val);
            exp_sum += val;
        }

        for (size_t i = 0; i < data.size(); ++i)
        {
            probs[i] = probs[i] / exp_sum;
        }
    }

}
