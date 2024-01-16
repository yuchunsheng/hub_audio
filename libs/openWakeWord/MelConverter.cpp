#include "MelConverter.h"

MelConverter::MelConverter(const std::string& MelFilepath){
  
  std::string instanceName{"Audio To Mel inference"};
  
  mEnv = std::make_shared<Ort::Env>(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                                  instanceName.c_str());
  mEnv->DisableTelemetryEvents();
  // Set up options for session
  Ort::SessionOptions sessionOptions;

  /**************** Create allocator ******************/
  // Allocator is used to get model information
  Ort::AllocatorWithDefaultOptions allocator;
  
  // Create session by loading the onnx model
  mSession = std::make_shared<Ort::Session>(*mEnv, MelFilepath.c_str(),sessionOptions);
  /**************** Input info ******************/
  // Get the number of input nodes
  auto temp_input_name0 = mSession->GetInputNameAllocated(0, allocator);
  mInputName.push_back(temp_input_name0.get());

  // Get the type of the input
  // 0 means the first input of the model
  Ort::TypeInfo inputTypeInfo = mSession->GetInputTypeInfo(0);
  auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
  ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
  // Get the shape of the input
  mInputDims = inputTensorInfo.GetShape();

  /**************** Output info ******************/
  // Get the number of output nodes
  size_t numOutputNodes = mSession->GetOutputCount();
  
  // Get the name of the output
  // 0 means the first output of the model
  // The example only has one output, so use 0 here
  auto temp_output_name0  = mSession->GetOutputNameAllocated(0, allocator);
  mOutputName.push_back(temp_output_name0.get());
  
  // Get the type of the output
  // 0 means the first output of the model
  Ort::TypeInfo outputTypeInfo = mSession->GetOutputTypeInfo(0);
  auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
  ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();

  // Get the shape of the output
  mOutputDims = outputTensorInfo.GetShape();

}

void MelConverter::Inference(vector<float> &samplesIn, vector<float> &melsOut){
    
  // Compute the product of all input dimension
  size_t inputTensorSize = vectorProduct(mInputDims); 
  std::vector<float> inputTensorValues(inputTensorSize);
  // copy the input sample to the input tensor and clear the input vector
  copy(samplesIn.begin(), samplesIn.end(), back_inserter(inputTensorValues));
  samplesIn.clear();
  // Assign memory for input tensor
  // inputTensors will be used by the Session Run for inference
  std::vector<Ort::Value> inputTensors;
  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
      OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  inputTensors.push_back(Ort::Value::CreateTensor<float>(
      memoryInfo, inputTensorValues.data(), inputTensorSize, mInputDims.data(),
      mInputDims.size()));

  // Create output tensor (including size and value)
  size_t outputTensorSize = vectorProduct(mOutputDims);
  std::vector<float> outputTensorValues(outputTensorSize);

  // Assign memory for output tensors
  // outputTensors will be used by the Session Run for inference
  std::vector<Ort::Value> outputTensors;
  outputTensors.push_back(Ort::Value::CreateTensor<float>(
      memoryInfo, outputTensorValues.data(), outputTensorSize,
      mOutputDims.data(), mOutputDims.size()));
  
  // 1 means number of inputs and outputs
  // InputTensors and OutputTensors, and inputNames and
  // outputNames are used in Session Run
  std::vector<const char*> inputNames{mInputName};
  std::vector<const char*> outputNames{mOutputName};
  mSession->Run(Ort::RunOptions{nullptr}, inputNames.data(),
                inputTensors.data(), 1, outputNames.data(),
                outputTensors.data(), 1);
  // (1, 1, frames, mels = 32)
  const auto &melOut = outputTensors.front();
  const auto melInfo = melOut.GetTensorTypeAndShapeInfo();
  const auto melShape = melInfo.GetShape();

  const float *melData = melOut.GetTensorData<float>();
  size_t melCount =
      accumulate(melShape.begin(), melShape.end(), 1, multiplies<>());

  for (size_t i = 0; i < melCount; i++) {
          // Scale mels for Google speech embedding model
          melsOut.push_back((melData[i] / 10.0f) + 2.0f);
  }

  const size_t chunkSamples = 1280; // 80 ms
  size_t frameSize = 4 * chunkSamples;

  inputTensorValues.erase(inputTensorValues.begin(),
                        inputTensorValues.begin() + frameSize);
}

