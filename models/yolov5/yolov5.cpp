#include "yolov5.h"
#include <algorithm>
#include <set>
#include <opencv2/imgproc/imgproc.hpp>
#include "vortex/utils/fileops.h"
#include "vortex/utils/arrayops.h"


namespace vortex
{
    void sortBoxesByScore(std::vector<DetBox>& boxes)
    {
        std::sort(boxes.begin(), boxes.end(), 
            [](DetBox& b1, DetBox& b2) {
                return b1.score < b2.score;
            }
        );
    }

    float BoxOverlap(DetBox& b1, DetBox& b2)
    {
        float xmin = std::max(b1.xmin, b2.xmin);
        float xmax = std::min(b1.xmax, b2.xmax);
        float ymin = std::max(b1.ymin, b2.ymin);
        float ymax = std::min(b1.ymax, b2.ymax);

        float i = std::max(xmax - xmin, 0.0f) * std::max(ymax - ymin, 0.0f);
        float u = b1.Area() + b2.Area() - i;
		return i / u;
    }

    Yolov5::Yolov5(const std::string& engine_path, float conf_thresh, float iou_thresh)
        : m_ConfThresh(conf_thresh), m_IouThresh(iou_thresh)
    {
        // assume single image inference.
        BlobInfo input_info = {
            "images", 1, 3, 640, 640
        };
        BlobInfo output_info = {
            "output", 1, 25200, 85, 1
        };
        this->LoadEngine(engine_path, input_info, output_info);

        m_Decoder = std::make_shared<YoloDecoder>(0.5, 0.5, 80);
    }

    void Yolov5::Preprocess(cv::Mat& image)
    {
        // preprocess image, feed into m_InputBlob
        cv::Mat temp;
        cv::resize(image, temp, cv::Size(640, 640));
        
        uint32_t height = m_InputInfo.height;
        uint32_t width = m_InputInfo.width;
        
        uint32_t image_area = width * height;
        float* red_plane = m_InputBlob->dataCpu;
        float* green_plane = m_InputBlob->dataCpu + image_area;
        float* blue_plane = m_InputBlob->dataCpu + image_area * 2;

        for (uint32_t y = 0; y < height; ++y)
        {
            unsigned char* line = temp.ptr<unsigned char>(y);
            for (uint32_t x = 0; x < width; ++x)
            {
                unsigned char red = line[x * 3 + 2];
                unsigned char green = line[x * 3 + 1];
                unsigned char blue = line[x * 3];
                uint32_t index = y * width + x;
                red_plane[index] = red / 255.0;
                green_plane[index] = green / 255.0;
                blue_plane[index] = blue / 255.0;
            }
        }
    }

    void Yolov5::Postprocess(std::vector<DetBox>& results)
    {
        // NCHW
        // 1x25200x85x1
        uint32_t num_anchors = m_OutputInfo.channels;
        uint32_t num_elements = m_OutputInfo.height;
        std::vector<DetBox> boxes;
        for (uint32_t i = 0; i < num_anchors; ++i)
        {
            float* anchor_data = m_OutputBlob->dataCpu + i * num_elements;
            if (anchor_data[4] < m_ConfThresh)
                continue;
            
            DetBox box;
            auto m = maxValue(anchor_data, 5, num_elements);
            box.class_index = m.first - 5;
            box.xmin = anchor_data[0] - anchor_data[2] / 2;
            box.ymin = anchor_data[1] - anchor_data[3] / 2;
            box.xmax = anchor_data[0] + anchor_data[2] / 2;
            box.ymax = anchor_data[1] + anchor_data[3] / 2;
            box.score = anchor_data[4];
            
            boxes.push_back(box);
        }

        // do nms
        results.clear();
        std::vector<uint32_t> indices = Nms(boxes);
        for (auto k : indices)
            results.push_back(boxes[k]);
    }

    void Yolov5::Detect(cv::Mat& image, std::vector<DetBox>& boxes)
    {
        Preprocess(image);
        InternalInfer();
        Postprocess(boxes);
    }

    void Yolov5::Infer(const std::vector<float>& inputs, std::vector<float>& outputs)
    {
        // copy data
        m_InputBlob->CopyFrom(inputs.data());

        this->InternalInfer();

        // collect data
        outputs.resize(m_OutputBlob->count);
        m_OutputBlob->CopyTo(outputs.data());
    }

    std::vector<uint32_t> Yolov5::Nms(std::vector<DetBox>& boxes)
    {
        sortBoxesByScore(boxes);
        std::vector<uint32_t> indices;
        std::vector<uint32_t> living(boxes.size(), 1);

        for (uint32_t pos = 0; pos < boxes.size(); ++pos)
        {
            if (living[pos] == 0) continue;
            indices.push_back(pos);
            // calculate overlap with later boxes
            for (uint32_t k = pos + 1; k < boxes.size(); ++k)
            {
                float overlap = BoxOverlap(boxes[pos], boxes[k]);
                if (overlap > m_IouThresh)
                    living[k] = 0;
            }
        }

        return indices;
    }
}
