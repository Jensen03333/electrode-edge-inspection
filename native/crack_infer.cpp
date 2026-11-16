// Windows x64, ONNX Runtime 1.20.1: PGM -> NCHW float -> native crack logits.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
constexpr int width = 320;
constexpr int height = 240;

std::string pgm_token(std::istream& input) {
    while (input >> std::ws && input.peek() == '#') {
        input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    std::string token;
    if (!(input >> token)) throw std::runtime_error("Truncated PGM header");
    return token;
}

std::vector<float> read_frame(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open input frame");
    if (pgm_token(input) != "P5" || pgm_token(input) != "320" ||
        pgm_token(input) != "240" || pgm_token(input) != "255") {
        throw std::runtime_error("Input must be an 8-bit P5 PGM of size 320x240");
    }
    const int separator = input.get();
    if (!std::isspace(separator)) throw std::runtime_error("Invalid PGM raster separator");
    if (separator == '\r' && input.peek() == '\n') input.get();
    std::vector<unsigned char> pixels(width * height);
    input.read(reinterpret_cast<char*>(pixels.data()), pixels.size());
    if (input.gcount() != static_cast<std::streamsize>(pixels.size()) ||
        input.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("Unexpected PGM raster size");
    }
    std::vector<float> tensor(pixels.size());
    std::transform(pixels.begin(), pixels.end(), tensor.begin(),
                   [](unsigned char value) { return static_cast<float>(value) / 255.0f; });
    return tensor;
}

void run(const fs::path& model, const fs::path& frame, const fs::path& destination) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "crack-infer");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    Ort::Session session(env, model.c_str(), options);

    auto pixels = read_frame(frame);
    const std::array<int64_t, 4> shape{1, 1, height, width};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input = Ort::Value::CreateTensor<float>(memory, pixels.data(), pixels.size(), shape.data(), shape.size());
    const char* inputs[] = {"image"};
    const char* outputs[] = {"logits"};
    auto result = session.Run(Ort::RunOptions{nullptr}, inputs, &input, 1, outputs, 1);
    auto output_info = result.front().GetTensorTypeAndShapeInfo();
    if (output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        output_info.GetShape() != std::vector<int64_t>(shape.begin(), shape.end())) {
        throw std::runtime_error("Expected one float logit per input pixel");
    }
    std::ofstream output(destination, std::ios::binary);
    const auto* logits = result.front().GetTensorData<float>();
    output.write(reinterpret_cast<const char*>(logits), pixels.size() * sizeof(float));
    output.close();
    if (!output) throw std::runtime_error("Failed to write logits");
}

int wmain(int argc, wchar_t** argv) {
    if (argc != 5) {
        std::cerr << "Usage: crack_infer.exe onnxruntime.dll model.onnx frame.pgm logits.f32\n";
        return 2;
    }
    const auto dll_path = fs::absolute(argv[1]);
    HMODULE library = LoadLibraryExW(dll_path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!library) {
        std::cerr << "Could not load ONNX Runtime DLL, Windows error " << GetLastError() << '\n';
        return 1;
    }
    int exit_code = 0;
    try {
        auto get_api = reinterpret_cast<decltype(&OrtGetApiBase)>(GetProcAddress(library, "OrtGetApiBase"));
        if (!get_api) throw std::runtime_error("DLL has no OrtGetApiBase symbol");
        const OrtApi* api = get_api()->GetApi(ORT_API_VERSION);
        if (!api) throw std::runtime_error("DLL does not support the compiled ONNX Runtime API version");
        Ort::InitApi(api);
        run(argv[2], argv[3], argv[4]);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        exit_code = 1;
    }
    FreeLibrary(library);
    return exit_code;
}
