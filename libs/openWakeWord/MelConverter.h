#ifndef openwakeword_h
#define openwakeword_h

#include <cstdio>
#include <filesystem>
#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <mutex>
#include <condition_variable>
#include <stdexcept>
#include <thread>

#include <onnxruntime_cxx_api.h>

using namespace std;
using namespace filesystem;

#define VERBOSE
//#define TIME_PROFILE

#ifdef TIME_PROFILE
using clock_time = std::chrono::system_clock;
using sec = std::chrono::duration<double>;
#endif

/**
 * @brief Compute the product over all the elements of a vector
 * @tparam T
 * @param v: input vector
 * @return the product
 */
template <typename T>
size_t vectorProduct(const std::vector<T>& v) {
  return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

#ifdef VERBOSE
/**
 * @brief Operator overloading for printing vectors
 * @tparam T
 * @param os
 * @param v
 * @return std::ostream&
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
  os << "[";
  for (int i = 0; i < v.size(); ++i) {
    os << v[i];
    if (i != v.size() - 1) {
      os << ", ";
    }
  }
  os << "]";
  return os;
}
#endif

const string instanceName = "openWakeWord";

const size_t numMels = 32;
const size_t embWindowSize = 76; // 775 ms
const size_t embStepSize = 8;    // 80 ms
const size_t embFeatures = 96;
const size_t wwFeatures = 16;

class MelConverter {
 public:
  /**
   * @brief Constructor
   * @param modelFilepath: path to the .onnx file
   */
  MelConverter(const std::string& MelFilepath  //Audio to Mel onnx model path
            //    const std::string& FeaturesFilepath, //Mel to Features onnx model path
            //    const std::string& OutputFilepath   //Features to Output onnx model path
                );

  /**
   * @brief Perform inference on a single image
   * @param imageFilepath: path to the image
   * @return the index of the predicted class
   */
  void Inference(vector<float> &samplesIn,
                 vector<float> &melsOut);

 private:
  // ORT Environment
  std::shared_ptr<Ort::Env> mEnv;

  // Session
  std::shared_ptr<Ort::Session> mSession;

  // Inputs
  vector<const char *> mInputName;
  std::vector<int64_t> mInputDims;

  // Outputs
  vector<const char *> mOutputName;
  std::vector<int64_t> mOutputDims;

  /**
   * @brief Create a tensor from an input image
   * @param img: the input image
   * @param inputTensorValues: the output tensor
   */
//   void CreateTensorFromImage(const cv::Mat& img,
//                              std::vector<float>& inputTensorValues);
};


#endif  //openwakeword_h