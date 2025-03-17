#include <stdexcept>
#include <ranges>

#include "expected.hpp"
#include "logger.hpp"
#include "image_preprocessor.hpp"

namespace captioning
{
    /*
    ImagePreprocessor::ImagePreprocessor(std::vector<int64_t> input_shape) : input_shape_(std::move(input_shape))
    {
        if (input_shape_.size() != 4)
        {
            throw std::invalid_argument("Input shape must have 4 dimensions (N, C, H, W)");
        }
    }
    */

    ImagePreprocessor::ImagePreprocessor(std::vector<int64_t> input_shape) noexcept : input_shape_(std::move(input_shape))
    {
        if (input_shape_.size() != 4)
        {
            auto logger = Logger::get_logger();
            logger->error("Invalid input shape: expected 4 dimensions (N, C, H, W)");
            input_shape_ = {1, 3, 224, 224}; // Set a default shape or handle it appropriately
        }
    }

    tl::expected<Ort::Value, std::string> ImagePreprocessor::preprocess(const std::string &image_path) const noexcept
    {
        auto logger = Logger::get_logger();

        cv::Mat image = cv::imread(image_path);
        if (image.empty())
        {
            logger->error("Failed to load image: {}", image_path);
            return tl::unexpected("Failed to load image: " + image_path);
        }

        // Resize image
        cv::resize(image, image, cv::Size(input_shape_[3], input_shape_[2]));

        // Convert BGR to RGB
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

        // Normalize image
        image = normalize_image(image);

        // Convert HWC to CHW
        std::vector<float> input_data = hwc_to_chw(image);

        // Create ONNX tensor
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_data.data(), input_data.size(), input_shape_.data(), input_shape_.size());

        logger->info("Image preprocessed successfully: {}", image_path);
        return input_tensor;
    }

    cv::Mat ImagePreprocessor::normalize_image(const cv::Mat &image) const noexcept
    {
        cv::Mat normalized_image;
        image.convertTo(normalized_image, CV_32F, 1.0 / 255.0);
        return normalized_image;
    }

    std::vector<float> ImagePreprocessor::hwc_to_chw(const cv::Mat &image) const noexcept
    {
        std::vector<float> input_data(input_shape_[1] * input_shape_[2] * input_shape_[3]);
        for (int c = 0; c < input_shape_[1]; ++c)
        {
            for (int h = 0; h < input_shape_[2]; ++h)
            {
                for (int w = 0; w < input_shape_[3]; ++w)
                {
                    input_data[c * input_shape_[2] * input_shape_[3] + h * input_shape_[3] + w] =
                        image.at<cv::Vec3f>(h, w)[c];
                }
            }
        }
        return input_data;
    }

}