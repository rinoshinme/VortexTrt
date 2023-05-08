/*
 * basic infer engine functions
*/

#include <iostream>

namespace vortex
{
    class BaseInferEngine
    {
    private:

    public:
        virtual ~BaseInferEngine();
        virtual void LoadEngine(const std::string& engine_path);
        virtual void SetIOBlobs();
    };
}

