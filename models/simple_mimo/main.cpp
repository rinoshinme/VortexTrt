#include <string>
#include "vortex/engine/mimo_infer_engine.h"
#include "vortex/utils/fileops.h"

using namespace vortex;


class MimoSample : public MimoInferEngine
{
private:

public:
    MimoSample(const std::string& engine_path)
    {
        std::vector<BlobInfo> input_info;
        input_info.push_back({"x1", 224, 224, 3});
        input_info.push_back({"x2", 224, 224, 3});
        std::vector<BlobInfo> output_info;
        output_info.push_back({"output", 1000, 1, 1});
        output_info.push_back({"feature", 512, 1, 1});

        this->LoadEngine(engine_path, input_info, output_info);
    }

    /*
     * test sample:
     *  load input from file
     */
    void Forward()
    {
        const std::string input_path("./input.txt");
        std::vector<float> input_data;
        loadFloats(input_path, input_data);

        m_InputBlobs[0]->CopyFrom(input_data.data());
        m_InputBlobs[1]->CopyFrom(input_data.data());

        this->InternalInfer();

        std::vector<float> out_data;
        out_data.resize(m_OutputBlobs[0]->count);
        m_OutputBlobs[0]->CopyTo(out_data.data());
        std::vector<float> feature_data;
        feature_data.resize(m_OutputBlobs[1]->count);
        m_OutputBlobs[1]->CopyTo(feature_data.data());
        const std::string out_text("./output.txt");
        saveFloats(out_text, out_data);
        const std::string feature_text("./feature.txt");
        saveFloats(feature_text, feature_data);
    }
};


int main()
{
    const std::string engine_path("../mimo.engine");
    MimoSample sample(engine_path);
    sample.Forward();

    return 0;
}
