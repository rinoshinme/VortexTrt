#include "fileops.h"
#include <fstream>
#include <iostream>

namespace vortex
{
    bool loadBinaryContent(const std::string& file_path, std::vector<unsigned char>& data)
    {
        std::ifstream file(file_path, std::ios::binary);
        if (file.good())
        {
            file.seekg(0, file.end);
            size_t file_size = file.tellg();
            file.seekg(0, file.beg);
            data.resize(file_size);
            char* ptr = reinterpret_cast<char*>(data.data());
            file.read(ptr, file_size);
            file.close();
            return true;
        }
        return false;
    }

    std::map<std::string, nvinfer1::Weights> loadWeights(const std::string& filepath)
    {
        std::cout << "Loading weights from " << filepath << std::endl;

        std::map<std::string, nvinfer1::Weights> weightMap;

        std::ifstream input(filepath);
        if (!input.is_open())
            return weightMap;
        
        // read number of weight blobs
        int32_t count;
        input >> count;

        while (count--)
        {
            nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, nullptr, 0};
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
