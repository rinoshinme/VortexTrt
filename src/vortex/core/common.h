#pragma once

#include <memory>


namespace vortex
{
    struct TrtInferDeleter
    {
        template<typename T>
        void operator()(T* obj)
        {
            if (obj) obj->destroy();
        }
    };

    template<typename T>
    // using TrtUniquePtr = std::unique_ptr<T, TrtInferDeleter>;
    using TrtUniquePtr = std::unique_ptr<T>;

    template<typename T>
    // using TrtSharedPtr = std::shared_ptr<T, TrtInferDeleter>;
    using TrtSharedPtr = std::shared_ptr<T>;
}
