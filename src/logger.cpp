#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <iostream>

#include "logger.hpp"

namespace captioning
{
    std::shared_ptr<spdlog::logger> Logger::logger_ = nullptr;

    void Logger::init(const std::string &log_file) noexcept
    {
        try
        {
            std::shared_ptr<spdlog::sinks::stdout_color_sink_mt> console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            console_sink->set_level(spdlog::level::info);

            std::shared_ptr<spdlog::sinks::rotating_file_sink_mt> file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(log_file, 1024 * 1024 * 5, 3);
            file_sink->set_level(spdlog::level::debug);

            logger_ = std::make_shared<spdlog::logger>("captioning", spdlog::sinks_init_list{console_sink, file_sink});
            logger_->set_level(spdlog::level::debug);
            spdlog::set_default_logger(logger_);
        }
        catch (const spdlog::spdlog_ex &ex)
        {
            std::cerr << "Logger initialization failed: " << ex.what() << std::endl;
        }
    }

    std::shared_ptr<spdlog::logger> Logger::get_logger() noexcept
    {
        if (!logger_)
        {
            init();
        }
        return logger_;
    }

}