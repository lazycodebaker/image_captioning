 

# Image Captioning System

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![C++ Version](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)

A robust and extensible C++ application for generating captions from images using deep learning models (ONNX format) and beam search decoding. This project leverages modern C++ features, OpenCV for image preprocessing, and the ONNX Runtime for model inference.

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features
- **Image Preprocessing**: Resizes, normalizes, and converts images to the required tensor format using OpenCV.
- **Model Inference**: Utilizes ONNX Runtime to run deep learning models for caption generation.
- **Beam Search**: Implements beam search decoding to generate high-quality captions.
- **Configurable**: Supports JSON-based configuration for model paths, vocabulary, and hyperparameters.
- **Logging**: Integrated logging with `spdlog` for debugging and monitoring.
- **Error Handling**: Uses `tl::expected` for robust error management.

## Prerequisites
Before building and running the project, ensure you have the following installed:
- **C++20-compatible compiler** (e.g., GCC 11+, Clang 13+, MSVC 2019+)
- **CMake** (version 3.20 or higher)
- **OpenCV** (version 4.x)
- **ONNX Runtime** (version 1.10 or higher)
- **spdlog** (for logging)
- **nlohmann/json** (JSON parsing library)
- **tl::expected** (for error handling)

Optional:
- A pre-trained image captioning model in ONNX format (e.g., based on the [Flickr8k dataset](https://github.com/goodwillyoga/Flickr8k_dataset)).
- A vocabulary file in JSON format.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/image-captioning-system.git
   cd image-captioning-system
   ```

2. **Install Dependencies**:
   - Install OpenCV, ONNX Runtime, and other dependencies via your package manager (e.g., `apt`, `brew`, or `vcpkg`) or build from source.
   - Example for Ubuntu:
     ```bash
     sudo apt update
     sudo apt install libopencv-dev
     ```

3. **Build the Project**:
   ```bash
   mkdir build && cd build
   cmake ..
   make
   ```

4. **Prepare Model and Vocabulary**:
   - Place your ONNX model file (e.g., `model.onnx`) and vocabulary file (e.g., `vocab.json`) in a directory of your choice.
   - Update the configuration file (see [Configuration](#configuration)) with the correct paths.

## Usage
Run the application from the command line with the following syntax:
```bash
./image_captioning <config_path> <image_path>
```
- `<config_path>`: Path to the JSON configuration file.
- `<image_path>`: Path to the input image file (e.g., `.jpg`, `.png`).

### Example
```bash
./image_captioning config.json sample.jpg
```
**Output**:
```
Generated Caption: A dog runs across a grassy field.
```

### Configuration
The application uses a JSON configuration file to specify model settings. Example `config.json`:
```json
{
  "model_path": "path/to/model.onnx",
  "vocab_path": "path/to/vocab.json",
  "input_shape": [1, 3, 224, 224],
  "max_caption_length": 20,
  "beam_width": 5
}
```
- `model_path`: Path to the ONNX model file.
- `vocab_path`: Path to the vocabulary JSON file.
- `input_shape`: Model input shape in `[N, C, H, W]` format (batch size, channels, height, width).
- `max_caption_length`: Maximum length of the generated caption (default: 20).
- `beam_width`: Beam width for beam search decoding (default: 5).

## Project Structure
```
image-captioning-system/
├── include/
│   ├── config.hpp          # Configuration parsing and validation
│   ├── caption_generator.hpp # Caption generation logic
│   ├── logger.hpp          # Logging utilities
│   ├── image_preprocessor.hpp # Image preprocessing
│   ├── model_inference.hpp # Model inference with ONNX Runtime
│   ├── vocabulary.hpp      # Vocabulary management
│   └── expected.hpp        # Error handling with tl::expected
├── src/
│   ├── main.cpp            # Entry point
│   ├── config.cpp          # Config implementation
│   ├── caption_generator.cpp # Caption generator implementation
│   ├── logger.cpp          # Logger implementation
│   ├── image_preprocessor.cpp # Image preprocessor implementation
│   ├── model_inference.cpp # Model inference implementation
│   └── vocabulary.cpp      # Vocabulary implementation
├── CMakeLists.txt          # CMake build configuration
├── README.md               # Project documentation
└── config.json             # Example configuration file
```

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/my-feature`).
3. Commit your changes (`git commit -m "Add my feature"`).
4. Push to the branch (`git push origin feature/my-feature`).
5. Open a pull request.

Please ensure your code follows the C++20 standard and includes appropriate tests.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Inspired by the [Flickr8k dataset](https://github.com/goodwillyoga/Flickr8k_dataset).
- Built with the support of open-source libraries like OpenCV, ONNX Runtime, and spdlog.
