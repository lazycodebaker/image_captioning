#include <fstream>
#include <nlohmann/json.hpp>

#include "vocabulary.hpp"
#include "logger.hpp"
#include "expected.hpp"

namespace captioning
{
    tl::expected<Vocabulary, std::string> Vocabulary::from_file(const std::filesystem::path &vocab_path) noexcept
    {
        Vocabulary vocab;
        if (auto error = vocab.load_from_file(vocab_path); !error.empty())
        {
            return tl::unexpected(error);
        }
        return vocab;
    }

    std::string Vocabulary::load_from_file(const std::filesystem::path &vocab_path) noexcept
    {
        auto logger = Logger::get_logger();

        std::ifstream vocab_file(vocab_path);
        if (!vocab_file.is_open())
        {
            logger->error("Failed to open vocabulary file: {}", vocab_path.string());
            return "Failed to open vocabulary file: " + vocab_path.string();
        }

        try
        {
            nlohmann::json vocab_json;
            vocab_file >> vocab_json;

            id_to_token_ = vocab_json.get<std::vector<std::string>>();
            for (int i = 0; i < static_cast<int>(id_to_token_.size()); ++i)
            {
                token_to_id_[id_to_token_[i]] = i;
            }

            logger->info("Vocabulary loaded successfully with {} tokens", id_to_token_.size());
            return {};
        }
        catch (const std::exception &ex)
        {
            logger->error("Failed to parse vocabulary file {}: {}", vocab_path.string(), ex.what());
            return "Failed to parse vocabulary file: " + std::string(ex.what());
        }
    }

    std::string Vocabulary::id_to_token(int id) const noexcept
    {
        if (id < 0 || id >= static_cast<int>(id_to_token_.size()))
        {
            return "<unk>";
        }
        return id_to_token_[id];
    }

    int Vocabulary::token_to_id(std::string_view token) const noexcept
    {
        auto it = token_to_id_.find(std::string(token));
        if (it == token_to_id_.end())
        {
            return token_to_id_.at("<unk>");
        }
        return it->second;
    }
}