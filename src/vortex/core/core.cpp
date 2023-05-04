#include "core.h"

namespace vortex
{
    std::ostream& operator<<(std::ostream& os, const Rect& rect)
    {
        os << "[" << rect.x << ", " << rect.y << ", " << rect.width << ", " << rect.height << "]";
        return os;
    }

    std::map<std::string, Weights> loadWeights(const std::string& filpath)
    {
        std::cout << "Loading weights from " << filepath << std::endl;

        std::map<std::string, Weights> weightMap;

        std::ifstream input(filepath);
        if (!input.is_open())
            return weightMap;
        
        // read number of weight blobs
        int32_t count;
        input >> count;

        while (count--)
        {
            Weights wt{DataType::kFloat, nullptr, 0};
            uint32_t size;
            // read name and size of blob
            std::string name;
            input >> name >> std::dec >> size;
            // load blob
            uint32_t* val = new uint32_t[size * sizeof(uint32_t)];
            for (uint32_t x = 0; x < size; ++x)
                input >> std::hex >> val[x];
            wt.values = val;
            wt.count = size;
            weightMap[name] = wt;
        }
        return weightMap;
    }

}
