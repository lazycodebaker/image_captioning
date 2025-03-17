#ifndef IMAGE_PREPROCESSOR_HPP
#define IMAGE_PREPROCESSOR_HPP

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <span>

#include "expected.hpp"

namespace captioning
{

    class ImagePreprocessor
    {
    public:
        explicit ImagePreprocessor(std::vector<int64_t> input_shape) noexcept;

        [[nodiscard]] tl::expected<Ort::Value, std::string> preprocess(const std::string &image_path) const noexcept;

    private:
        std::vector<int64_t> input_shape_;

        [[nodiscard]] cv::Mat normalize_image(const cv::Mat &image) const noexcept;

        [[nodiscard]] std::vector<float> hwc_to_chw(const cv::Mat &image) const noexcept;
    };
}

#endif