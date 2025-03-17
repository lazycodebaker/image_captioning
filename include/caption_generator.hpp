#ifndef CAPTION_GENERATOR_HPP
#define CAPTION_GENERATOR_HPP

#include <string>
#include <memory>

#include "config.hpp"
#include "image_preprocessor.hpp"
#include "vocabulary.hpp"
#include "model_inference.hpp"

namespace captioning
{
    class CaptionGenerator
    {
    public:
        static tl::expected<CaptionGenerator, std::string> create(const Config &config) noexcept;

        [[nodiscard]] tl::expected<std::string, std::string> generate(const std::string &image_path) noexcept;

    private:
        Config config_;
        std::unique_ptr<ImagePreprocessor> preprocessor_;
        std::unique_ptr<Vocabulary> vocab_;
        std::unique_ptr<ModelInference> model_;

        CaptionGenerator(Config config, std::unique_ptr<ImagePreprocessor> preprocessor, std::unique_ptr<Vocabulary> vocab, std::unique_ptr<ModelInference> model) noexcept;

        [[nodiscard]] std::string decode_caption(std::span<const int> token_ids) const noexcept;
    };
}

#endif