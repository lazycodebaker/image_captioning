#include <iostream>
#include <filesystem>
#include <format>

#include "config.hpp"
#include "caption_generator.hpp"
#include "logger.hpp"

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cerr << std::format("Usage: {} <config_path> <image_path>", argv[0]) << std::endl;
        return 1;
    }

    try
    {
        // Initialize logger
        captioning::Logger::init("captioning.log");

        // Load configuration
        std::filesystem::path config_path = argv[1];
        auto config = captioning::Config::from_file(config_path);
        if (!config)
        {
            std::cerr << std::format("Error: {}", config.error()) << std::endl;
            return 1;
        }

        // Initialize caption generator
        auto generator = captioning::CaptionGenerator::create(*config);
        if (!generator)
        {
            std::cerr << std::format("Error: {}", generator.error()) << std::endl;
            return 1;
        }

        // Generate caption
        std::string image_path = argv[2];
        auto caption = generator->generate(image_path);
        if (!caption)
        {
            std::cerr << std::format("Error: {}", caption.error()) << std::endl;
            return 1;
        }

        std::cout << std::format("Generated Caption: {}", *caption) << std::endl;
    }
    catch (const std::exception &ex)
    {
        std::cerr << std::format("Error: {}", ex.what()) << std::endl;
        return 1;
    }
    return EXIT_SUCCESS;
}

// https://github.com/goodwillyoga/Flickr8k_dataset?tab=readme-ov-file

// STILL UNDER WORK -- NOT MADE COMPLETELY ... ( WORKING ON IT )