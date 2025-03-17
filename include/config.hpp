#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <string>
#include <vector>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <iomanip>
#include <iostream>

#include "expected.hpp"

namespace captioning
{

    class Config
    {
    public:
        static tl::expected<Config, std::string> from_file(const std::filesystem::path &config_path) noexcept;

        [[nodiscard]] std::string model_path() const noexcept { return model_path_; }
        [[nodiscard]] std::string vocab_path() const noexcept { return vocab_path_; }
        [[nodiscard]] std::vector<int64_t> input_shape() const noexcept { return input_shape_; }
        [[nodiscard]] int max_caption_length() const noexcept { return max_caption_length_; }
        [[nodiscard]] int beam_width() const noexcept { return beam_width_; }

    private:
        std::string model_path_;
        std::string vocab_path_;
        std::vector<int64_t> input_shape_;
        int max_caption_length_ = 20;
        int beam_width_ = 5;

        Config() = default;

        [[nodiscard]] std::string validate() const noexcept;
    };
}

#endif