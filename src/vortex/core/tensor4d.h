#pragma once

#include <vector>

namespace vortex
{
    // commonly used tensor in NCHW format.
    template<typename T>
    class Tensor4D
    {
    private:
        std::vector<T> m_Data;
        uint32_t m_Batch;
        uint32_t m_Channels;
        uint32_t m_Height;
        uint32_t m_Width;

    public:
        Tensor4D(uint32_t n, uint32_t c, uint32_t h, uint32_t w, T* data = nullptr)
            : m_Batch(n), m_Channels(c), m_Height(h), m_Width(w)
        {
            uint32_t size = n * c * h * w;
            m_Data.resize(size);
            if (data != nullptr)
                memcpy(m_Data.data(), data, size * sizeof(T));
        }

        T* Ptr() { m_Data.data(); }

        T* Ptr(uint32_t n) 
        { 
            return m_Data.data() + n * m_Channels * m_Height * m_Width;
        }

        T* Ptr(uint32_t n, uint32_t c)
        {
            return m_Data.data() + n * ImageSize() + c * ImageArea();
        }

        T* Ptr(uint32_t n, uint32_t c, uint32_t h)
        {
            return m_Data.data() + n * ImageSize() + c * ImageArea() + h * ImageWidth();
        }

        uint32_t Size() const { return m_Batch * m_Width * m_Height * m_Channels; }
        uint32_t ImageSize() const { return m_Width * m_Height * m_Channels; }
        uint32_t ImageArea() const { return m_Width * m_Height; }
        uint32_t ImageWidth() const { return m_Width; }
        uint32_t ImageHeight() const { return m_Height; }
        uint32_t BatchSize() const { return m_Batch; }
        uint32_t Channels() const { return m_Channels; }
    };

    typedef Tensor4D<float> Tensor4Df;
}
