#include <string>
#include <iostream>
#include <filesystem>
#include <regex>
#include <fstream>
#include <opencv2/opencv.hpp>

std::string extractFileName(const std::string& filePath) {
    const std::regex PICTURE_FILENAME_REGEX("[^/]*\\.png");
    std::smatch match;

    if (std::regex_search(filePath, match, PICTURE_FILENAME_REGEX))
        return match[0];
    else
        return "";
}

void saveHuMoments(std::ofstream& outputFile, const std::string& picturePath) {
    cv::Mat image = cv::imread(picturePath, cv::IMREAD_GRAYSCALE);
    cv::Moments moments = cv::moments(image, false);
    double huMoments[7];
    cv::HuMoments(moments, huMoments);

    for (double huMoment : huMoments) {
        outputFile << huMoment << ";";
    }

    outputFile << std::endl;
}

int main() {
    const std::string PICTURES_PATH("../01-hu-moments/pictures");
    const std::regex PICTURE_REGEX(".*\\.png");

    std::ofstream outputFile;
    outputFile.open("results.csv");
    outputFile << "File name;h0;h1;h2;h3;h4;h5;h6" << std::endl;

    for (const auto& entry : std::filesystem::directory_iterator(PICTURES_PATH)) {
        std::string filePath = entry.path().string();
        if (std::regex_match(filePath, PICTURE_REGEX)) {
            std::string fileName = extractFileName(filePath);
            outputFile << fileName << ";";
            saveHuMoments(outputFile, filePath);
        }
    }

    outputFile.close();

    return 0;
}