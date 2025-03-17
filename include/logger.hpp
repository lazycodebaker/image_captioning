#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <spdlog/spdlog.h>
#include <memory>

namespace captioning
{
    class Logger
    {
    public:
        static void init(const std::string &log_file = "captioning.log") noexcept;

        [[nodiscard]] static std::shared_ptr<spdlog::logger> get_logger() noexcept;

    private:
        static std::shared_ptr<spdlog::logger> logger_;
    };
}

#endif
