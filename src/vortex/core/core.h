#pragma once

#include <iostream>

namespace vortex
{
    struct Rect
    {
        uint32_t x;
        uint32_t y;
        uint32_t width;
        uint32_t height;

        friend std::ostream& operator<<(std::ostream& os, const Rect& rect);
    };

    template<typename T>
    struct DetBox
    {
        T left;
        T right;
        T top;
        T bottom;
        float score;
        uint32_t class_id;

        T Area() const { return (right - left) * (bottom - top); }
    };

    typedef DetBox<float> DetBoxF;

    // box overlap
    float BoxIoU(const DetBoxF& box1, const DetBoxF& box2);
    bool DetBoxLess(const DetBoxF& box1, const DetBoxF& box2);
}
