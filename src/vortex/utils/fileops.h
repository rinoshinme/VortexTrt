#pragma once

#include <string>
#include <vector>


namespace vortex
{
    bool loadBinaryContent(const std::string& file_path, std::vector<unsigned char>& data);
}