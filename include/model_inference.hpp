#ifndef MODEL_INFERENCE_HPP
#define MODEL_INFERENCE_HPP

#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <memory>
#include <ranges>
#include <span>

#include "expected.hpp"
#include "vocabulary.hpp"

namespace captioning
{
    class ModelInference
    {
    public:
        static tl::expected<ModelInference, std::string> create(const std::string &model_path, std::vector<int64_t> input_shape) noexcept;

        [[nodiscard]] tl::expected<std::vector<int>, std::string> run(Ort::Value &input_tensor, int max_length, int beam_width, const Vocabulary &vocab) noexcept;

    private:
        std::unique_ptr<Ort::Session> session_;
        std::string input_name_;
        std::string output_name_;
        std::vector<int64_t> input_shape_;

        struct BeamState
        {
            std::vector<int> sequence;
            float score;
            bool finished;

            bool operator<(const BeamState &other) const
            {
                return score < other.score;
            }

            auto operator<=>(const BeamState &) const = default;
        };

        ModelInference(std::unique_ptr<Ort::Session> session, std::string input_name, std::string output_name, std::vector<int64_t> input_shape) noexcept;

        [[nodiscard]] std::vector<int> beam_search(Ort::Value &input_tensor, int max_length, int beam_width, const Vocabulary &vocab) noexcept;

        // [[nodiscard]] std::vector<float> run_single_step(Ort::Value &input_tensor, std::span<const int> current_sequence) noexcept; // C++ 20
        // [[nodiscard]] std::vector<float> run_single_step(Ort::Value &input_tensor, const std::vector<int> &current_sequence) noexcept; // C++17 ( for span fallback )
        [[nodiscard]] std::vector<float> run_single_step(Ort::Value &input_tensor, const std::vector<int> &current_sequence) noexcept;
    };
}

#endif