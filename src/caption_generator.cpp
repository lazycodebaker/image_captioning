#include <stdexcept>
#include <ranges>
#include <format>

#include "logger.hpp"
#include "expected.hpp"
#include "caption_generator.hpp"

namespace captioning
{
    tl::expected<CaptionGenerator, std::string> CaptionGenerator::create(const Config &config) noexcept
    {
        auto logger = Logger::get_logger();

        try
        {
            std::unique_ptr<captioning::ImagePreprocessor>  preprocessor = std::make_unique<ImagePreprocessor>(config.input_shape());
            tl::expected<captioning::Vocabulary, std::string> vocab = Vocabulary::from_file(config.vocab_path());
            if (!vocab)
            {
                return tl::unexpected(vocab.error());
            }
            tl::expected<captioning::ModelInference, std::string> model = ModelInference::create(config.model_path(), config.input_shape());
            if (!model)
            {
                return tl::unexpected(model.error());
            }

            logger->info("CaptionGenerator initialized successfully");
            return CaptionGenerator(config, std::move(preprocessor), std::make_unique<Vocabulary>(std::move(*vocab)), std::make_unique<ModelInference>(std::move(*model)));
        }
        catch (const std::exception &ex)
        {
            logger->error("Failed to initialize CaptionGenerator: {}", ex.what());
            return tl::unexpected("Failed to initialize CaptionGenerator: " + std::string(ex.what()));
        }
    }

    CaptionGenerator::CaptionGenerator(Config config, std::unique_ptr<ImagePreprocessor> preprocessor, std::unique_ptr<Vocabulary> vocab, std::unique_ptr<ModelInference> model) noexcept
        : config_(std::move(config)), preprocessor_(std::move(preprocessor)), vocab_(std::move(vocab)), model_(std::move(model)) {}

    tl::expected<std::string, std::string> CaptionGenerator::generate(const std::string &image_path) noexcept
    {
        auto logger = Logger::get_logger();

        try
        {
            auto input_tensor = preprocessor_->preprocess(image_path);
            if (!input_tensor)
            {
                return tl::unexpected(input_tensor.error());
            }

            auto token_ids = model_->run(*input_tensor, config_.max_caption_length(), config_.beam_width(), *vocab_);
            if (!token_ids)
            {
                return tl::unexpected(token_ids.error());
            }

            std::string caption = decode_caption(*token_ids);
            logger->info("Generated caption for {}: {}", image_path, caption);
            return caption;
        }
        catch (const std::exception &ex)
        {
            logger->error("Failed to generate caption for {}: {}", image_path, ex.what());
            return tl::unexpected("Failed to generate caption for " + image_path + ": " + ex.what());
        }
    }

    std::string CaptionGenerator::decode_caption(std::span<const int> token_ids) const noexcept
    {
        std::string caption;
        for (int id : token_ids)
        {
            std::string token = vocab_->id_to_token(id);
            if (token == vocab_->get_start_token() || token == vocab_->get_end_token() || token == "<unk>")
            {
                continue;
            }
            caption += token + " ";
        }
        return caption;
    }
}