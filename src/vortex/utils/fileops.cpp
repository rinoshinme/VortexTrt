#include "fileops.h"
#include <fstream>

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
}
