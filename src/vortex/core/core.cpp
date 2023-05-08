#include "core.h"
#include <fstream>

namespace vortex
{
    std::ostream& operator<<(std::ostream& os, const Rect& rect)
    {
        os << "[" << rect.x << ", " << rect.y << ", " << rect.width << ", " << rect.height << "]";
        return os;
    }

    float BoxIoU(const DetBoxF& box1, const DetBoxF& box2)
    {
        float x1 = std::max(box1.left, box2.left);
        float x2 = std::min(box1.right, box2.right);
        float y1 = std::max(box1.top, box2.top);
        float y2 = std::max(box1.bottom, box2.bottom);
        float i = std::max(x2 - x1, 0.0f) * std::max(y2 - y1, 0.0f);
        float u = box1.Area() + box2.Area() - i;
        return i / u;
    }

    bool DetBoxLess(const DetBoxF& box1, const DetBoxF& box2)
    {
        return box1.score < box2.score;
    }
}
