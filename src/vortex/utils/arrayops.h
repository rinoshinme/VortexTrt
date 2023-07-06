#pragma once

#include <cstdint>
#include <utility>

namespace vortex
{
    // array operations
    template<typename T>
    std::pair<uint32_t, T> maxValue(const T* array, uint32_t start, uint32_t end)
    {
        T max_value = array[start];
        uint32_t max_index = start;

        for (uint32_t idx = start; idx < end; ++idx)
        {
            if (array[idx] > max_value)
            {
                max_value = array[idx];
                max_index = idx;
            }
        }
        return std::make_pair(max_index, max_value);
    }

    template<typename T>
    std::pair<uint32_t, T> minValue(const T* array, uint32_t start, uint32_t end)
    {
        T min_value = array[start];
        uint32_t min_index = start;

        for (uint32_t idx = start; idx < end; ++idx)
        {
            if (array[idx] < min_value)
            {
                min_value = array[idx];
                min_index = idx;
            }
        }
        return std::make_pair(min_index, min_value);
    }

}
