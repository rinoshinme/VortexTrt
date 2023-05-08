#pragma once

#include <string>
#include <vector>
#include <map>
#include "NvInfer.h"


namespace vortex
{
    bool loadBinaryContent(const std::string& file_path, std::vector<unsigned char>& data);
    // void saveBinaryContent(const std::string& file_path, std::vector<float>& data);
    std::map<std::string, nvinfer1::Weights> loadWeights(const std::string& filpath);

    void saveFloats(const std::string& file_path, const std::vector<float>& data);
    void loadFloats(const std::string& file_path, std::vector<float>& data);

}
