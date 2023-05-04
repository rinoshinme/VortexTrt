#include "affine.h"


namespace vortex
{
    AffineMatrix::AffineMatrix(const Rect& source, const Rect& target)
    {
        m[0][1] = 0.0f;
        m[1][0] = 0.0f;
        
        float scalex = target.width * 1.0f / source.width;
        float scaley = target.height * 1.0f / source.height;
        m[0][0] = scalex;
        m[1][1] = scaley;

        m[0][2] = target.x - scalex * source.x;
        m[1][2] = target.y - scaley * source.y;
    }

    AffineMatrix AffineMatrix::inverse() const 
    {
        AffineMatrix result;
        float det = m[0][0] * m[1][1] - m[0][1] * m[1][0];

        result.m[0][0] = m[1][1] / det;
        result.m[0][1] = -m[0][1] / det;
        result.m[0][2] = (m[0][1] * m[1][2] - m[1][1] * m[0][2]) / det;
        result.m[1][0] = -m[1][0] / det;
        result.m[1][1] = m[0][0] / det;
        result.m[1][2] = (m[1][0] * m[0][2] - m[0][0] * m[1][2]) / det;
        return result;
    }

    std::ostream& operator<<(std::ostream& os, const AffineMatrix& mat)
    {
        os << mat.toString();
        return os;
    }
}
