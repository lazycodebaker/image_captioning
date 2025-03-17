#include <fstream>
#include <stdexcept>
#include <ranges>

#include "config.hpp"
#include "logger.hpp"
#include "expected.hpp"

namespace captioning
{
    tl::expected<Config, std::string> Config::from_file(const std::filesystem::path &config_path) noexcept
    {
        auto logger = Logger::get_logger();

        std::ifstream config_file(config_path);
        if (!config_file.is_open())
        {
            logger->error("Failed to open config file: {}", config_path.string());
            return tl::unexpected("Failed to open config file: " + config_path.string());
        }

        try
        {
            nlohmann::json config_json;
            config_file >> config_json;

            Config config;
            config.model_path_ = config_json.at("model_path").get<std::string>();
            config.vocab_path_ = config_json.at("vocab_path").get<std::string>();
            config.input_shape_ = config_json.at("input_shape").get<std::vector<int64_t>>();
            config.max_caption_length_ = config_json.value("max_caption_length", 20);
            config.beam_width_ = config_json.value("beam_width", 5);

            if (auto error = config.validate(); !error.empty())
            {
                logger->error("Invalid configuration: {}", error);
                return tl::unexpected(error);
            }

            logger->info("Configuration loaded successfully from {}", config_path.string());
            return config;
        }
        catch (const std::exception &ex)
        {
            logger->error("Failed to parse config file {}: {}", config_path.string(), ex.what());
            return tl::unexpected("Failed to parse config file: " + std::string(ex.what()));
        }
    }

    std::string Config::validate() const noexcept
    {
        if (model_path_.empty())
        {
            return "Model path cannot be empty";
        }
        if (vocab_path_.empty())
        {
            return "Vocabulary path cannot be empty";
        }
        if (input_shape_.size() != 4)
        {
            return "Input shape must have 4 dimensions (N, C, H, W)";
        }
        if (max_caption_length_ <= 0)
        {
            return "Max caption length must be positive";
        }
        if (beam_width_ <= 0)
        {
            return "Beam width must be positive";
        }
        return {};
    }
}