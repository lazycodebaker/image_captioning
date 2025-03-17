#ifndef VOCABULARY_HPP
#define VOCABULARY_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>

#include "expected.hpp"

namespace captioning
{

    class Vocabulary
    {
    public:
        static tl::expected<Vocabulary, std::string> from_file(const std::filesystem::path &vocab_path) noexcept;

        [[nodiscard]] std::string id_to_token(int id) const noexcept;

        [[nodiscard]] int token_to_id(std::string_view token) const noexcept;

        [[nodiscard]] size_t size() const noexcept { return id_to_token_.size(); }

        [[nodiscard]] std::string get_start_token() const noexcept { return "<start>"; }

        [[nodiscard]] std::string get_end_token() const noexcept { return "<end>"; }

    private:
        std::vector<std::string> id_to_token_;
        std::unordered_map<std::string, int, std::hash<std::string_view>> token_to_id_;

        Vocabulary() = default;

        [[nodiscard]] std::string load_from_file(const std::filesystem::path &vocab_path) noexcept;
    };
}

#endif