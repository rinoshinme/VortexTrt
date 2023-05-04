#pragma once

#include <opencv2/core/core.hpp>

#include "vortex/core/core.h"

namespace vortex
{
    struct AffineMatrix
    {
        /*
         *  a11 a12 a13
         *  a21 a22 a23
         *    0   0   1
         */ 

        float m[2][3];

        AffineMatrix() {}
        // create matrix from src/dst rectangle region
        AffineMatrix(const Rect& source, const Rect& target);
        // create inverse transform.
        AffineMatrix inverse() const;
        // apply pixel position
        __device__ void apply(float x, float y, float* px, float* py) const
        {
            *px = m[0][0] * x + m[0][1] * y + m[0][2];
            *py = m[1][0] * x + m[1][1] * y + m[1][2];
        }

        std::string toString() const
        {
            char buf[1024];
            sprintf(buf, "((%f,%f,,%f)\n(%f,%f,%f))", m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2]);
            return buf;
        }
    };

    std::ostream& operator<<(std::ostream& os, const AffineMatrix& mat);

}
