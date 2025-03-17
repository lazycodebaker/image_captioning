#include <algorithm>
#include <queue>
#include <stdexcept>
#include <ranges>

#include "model_inference.hpp"
#include "expected.hpp"
#include "logger.hpp"

namespace captioning
{
    tl::expected<ModelInference, std::string> ModelInference::create(const std::string &model_path, std::vector<int64_t> input_shape) noexcept
    {
        auto logger = Logger::get_logger();

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "image_captioning");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        std::unique_ptr<Ort::Session> session;
        try
        {
            session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
        }
        catch (const Ort::Exception &ex)
        {
            logger->error("Failed to load ONNX model: {}", ex.what());
            return tl::unexpected("Failed to load ONNX model: " + std::string(ex.what()));
        }

        Ort::AllocatorWithDefaultOptions allocator;
        Ort::AllocatedStringPtr input_name_ptr = session->GetInputNameAllocated(0, allocator);
        std::string input_name = input_name_ptr.get();

        Ort::AllocatedStringPtr output_name_ptr = session->GetOutputNameAllocated(0, allocator);
        std::string output_name = output_name_ptr.get();

        logger->info("Model loaded successfully: {}", model_path);
        return ModelInference(std::move(session), std::move(input_name), std::move(output_name), std::move(input_shape));
    }

    ModelInference::ModelInference(std::unique_ptr<Ort::Session> session, std::string input_name, std::string output_name, std::vector<int64_t> input_shape) noexcept
        : session_(std::move(session)), input_name_(std::move(input_name)), output_name_(std::move(output_name)), input_shape_(std::move(input_shape)) {}

    tl::expected<std::vector<int>, std::string> ModelInference::run(Ort::Value &input_tensor, int max_length, int beam_width, const Vocabulary &vocab) noexcept
    {
        auto logger = Logger::get_logger();
        try
        {
            auto token_ids = beam_search(input_tensor, max_length, beam_width, vocab);
            if (token_ids.empty())
            {
                logger->warn("No valid caption generated");
                return tl::unexpected("No valid caption generated");
            }
            return token_ids;
        }
        catch (const std::exception &ex)
        {
            logger->error("Failed to run model inference: {}", ex.what());
            return tl::unexpected("Failed to run model inference: " + std::string(ex.what()));
        }
    }

    std::vector<int> ModelInference::beam_search(Ort::Value &input_tensor, int max_length, int beam_width, const Vocabulary &vocab) noexcept
    {
        auto logger = Logger::get_logger();

        auto beam_compare = [](const BeamState &a, const BeamState &b)
        { return a.score < b.score; };
        std::priority_queue<BeamState, std::vector<BeamState>, decltype(beam_compare)> beam(beam_compare);

        // Initialize beam with start token
        BeamState initial_state{{vocab.token_to_id(vocab.get_start_token())}, 0.0f, false};
        beam.push(initial_state);

        std::vector<BeamState> finished_sequences;

        for (int step = 0; step < max_length; ++step)
        {
            std::vector<BeamState> candidates;

            while (!beam.empty())
            {
                BeamState state = beam.top();
                beam.pop();

                if (state.finished)
                {
                    finished_sequences.push_back(state);
                    continue;
                }

                std::vector<float> scores = run_single_step(input_tensor, state.sequence);
                // auto sorted_scores = std::ranges::zip_view(std::views::iota(0, static_cast<int>(scores.size())), scores) | std::ranges::to<std::vector<std::pair<int, float>>>();

                std::vector<std::pair<int, float>> sorted_scores;
                for (size_t i = 0; i < scores.size(); ++i)
                {
                    sorted_scores.emplace_back(i, scores[i]);
                };

                std::ranges::sort(sorted_scores, std::ranges::greater{}, &std::pair<int, float>::second);

                for (const auto &[token_id, score] : sorted_scores | std::views::take(beam_width))
                {
                    std::vector<int> new_sequence = state.sequence;
                    new_sequence.push_back(token_id);
                    float new_score = state.score + score;
                    bool finished = token_id == vocab.token_to_id(vocab.get_end_token());
                    candidates.push_back(BeamState{std::move(new_sequence), new_score, finished});
                }
            }

            // Keep top beam_width candidates
            std::ranges::sort(candidates, std::ranges::greater{}, &BeamState::score);
            for (const auto &candidate : candidates | std::views::take(beam_width))
            {
                beam.push(candidate);
            }

            if (beam.empty())
                break;
        }

        while (!beam.empty())
        {
            finished_sequences.push_back(beam.top());
            beam.pop();
        }

        if (finished_sequences.empty())
        {
            logger->warn("No valid caption generated");
            return {};
        }

        std::ranges::sort(finished_sequences, std::ranges::greater{}, &BeamState::score);
        return finished_sequences.front().sequence;
    }

    // std::vector<float> ModelInference::run_single_step(Ort::Value &input_tensor, std::span<const int> current_sequence) noexcept
    std::vector<float> ModelInference::run_single_step(Ort::Value &input_tensor, const std::vector<int> &current_sequence) noexcept
    {
        // Placeholder for running the model on a single step. In a real model, this would involve:
        // 1. Feeding the image tensor and current sequence to the model.
        // 2. Getting the softmax scores for the next token.
        // For simplicity, we assume the model handles this internally.

        const char *input_names[] = {input_name_.c_str()};
        const char *output_names[] = {output_name_.c_str()};

        std::vector<Ort::Value> output_tensors = session_->Run(
            Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

        float *output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        return std::vector<float>(output_data, output_data + output_shape[1]);
    }

} // namespace captioning