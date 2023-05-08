#include "yolo_decode.h"
#include <set>
#include <iostream>
#include <algorithm>


namespace vortex
{
	bool YoloBoxGreater(const YoloBox& box1, const YoloBox& box2)
	{
		return box1.confidence > box2.confidence;
	}

	float YoloBoxIou(const YoloBox& box1, const YoloBox& box2)
	{
		float x1 = std::max(box1.left, box2.left);
		float x2 = std::min(box1.right, box2.right);
		float y1 = std::max(box1.top, box2.top);
		float y2 = std::max(box1.bottom, box2.bottom);
		float i = std::max(x2 - x1, 0.0f) * std::max(y2 - y1, 0.0f);

		float area1 = (box1.right - box1.left) * (box1.bottom - box1.top);
		float area2 = (box2.right - box2.left) * (box2.bottom - box2.top);
		float u = area1 + area2 - i;
		return i / u;
	}

	YoloDecoder::YoloDecoder(float conf_thresh, float nms_thresh, int num_classes)
	{
		m_ConfThresh = conf_thresh;
		m_NmsThresh = nms_thresh;
		m_NumClasses = num_classes;
	}


	std::vector<YoloBox> YoloDecoder::DecodeCpu(const std::vector<float>& pred, int image_width, int image_height)
	{
		// get all possible class_indices
		int box_size = m_NumClasses + 5;
		std::vector<YoloBox> boxes;
		std::set<int> class_indices;
		for (size_t i = 0; i < pred.size(); i += box_size)
		{
			float score = pred[i + 4];
			if (score <= m_ConfThresh) continue;

			float cx = pred[i];
			float cy = pred[i + 1];
			float w = pred[i + 2];
			float h = pred[i + 3];
			std::pair<int, float> max_score = Argmax(pred.data(), i + 5, i + box_size);
			
			YoloBox box;
#if 1
			box.left = cx - w / 2;
			box.top = cy - h / 2;
			box.right = cx + w / 2;
			box.bottom = cy + h / 2;
#else
			box.left = cx;
			box.top = cy;
			box.right = w;
			box.bottom = h;

#endif
			// box.confidence = score * max_score.second;
			box.confidence = score;
			box.label = max_score.first;
			boxes.push_back(box);

			class_indices.insert(max_score.first);
		}

		// sort boxes from high to low
		std::sort(boxes.begin(), boxes.end(), YoloBoxGreater);
		// do per-class nms
		std::vector<YoloBox> results;
		for (auto class_id : class_indices)
		{
			std::set<size_t> keep = PerClassNms(boxes, class_id);
			for (auto keep_idx : keep)
				results.push_back(boxes[keep_idx]);
		}

		// resize boxes
		for (auto& box : results)
		{
			box.left = box.left / 640 * image_width;
			box.right = box.right / 640 * image_width;
			box.top = box.top / 640 * image_height;
			box.bottom = box.bottom / 640 * image_height;
		}
		
		return results;
	}

	std::set<size_t> YoloDecoder::PerClassNms(const std::vector<YoloBox>& boxes, int class_id)
	{
		std::set<size_t> keep_indices;
		std::vector<YoloBox> class_boxes;
		for (size_t idx = 0; idx < boxes.size(); ++idx)
		{
			if (boxes[idx].label == class_id)
			{
				keep_indices.insert(idx);
				class_boxes.push_back(boxes[idx]);
			}
		}

		size_t current = 0;
		while (current < boxes.size())
		{
			if (keep_indices.find(current) != keep_indices.end())
			{
				// calculate iou with later boxes
				for (size_t i = current + 1; i < boxes.size(); ++i)
				{
					if (keep_indices.find(i) == keep_indices.end())
						continue;
					float overlap = YoloBoxIou(boxes[current], boxes[i]);
					if (overlap > m_NmsThresh)
						keep_indices.erase(i);
				}
			}
			current += 1;
		}
		return keep_indices;
	}

	std::pair<int, float> YoloDecoder::Argmax(const float* pred, int start, int end)
	{
		int max_index = start;
		float max_value = pred[start];
		for (int i = start; i < end; ++i)
		{
			if (pred[i] > max_value)
			{
				max_value = pred[i];
				max_index = i;
			}
		}
		return std::make_pair(max_index - start, max_value);
	}
}
