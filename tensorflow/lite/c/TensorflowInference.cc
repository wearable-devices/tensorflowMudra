/* Wearable Devices - Tensorflow library */

#include <cstdlib>
#include <unordered_set>
#include <string>
#include <fstream>
#include <map>
#include <strstream>

#include "TensorflowInference.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/mutable_op_resolver.h"
#include "tensorflow/lite/delegates/coreml/coreml_delegate.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"

//#include "tensorflow/lite/delegates/flex/delegate.h"

using namespace std;

using namespace Mudra::Computation;

class ComputationalModel;

map<const char*, unique_ptr<ComputationalModel>> g_model;

class ComputationalModel
{
    // Logging
    shared_ptr<Logger> m_logger;

    std::unique_ptr<tflite::FlatBufferModel> m_model;
    std::unique_ptr<tflite::Interpreter> m_interpreter;

    TfLiteDelegate* m_coreMl_delegate;
    string m_WeightsFileName;
    vector<int> m_outputSizes;
    int batches;
    void InitInterpreter(const char* modelFileName, int num_threads) {
        // Load the model
        m_model = tflite::FlatBufferModel::BuildFromFile(modelFileName);

        if (!m_model) {
            ErrorMessage(m_logger) << "\nCould not create TensorFlow Graph: " << modelFileName;
        }
        else
        {
            DebugMessage(m_logger) << "\nGraph " << modelFileName << " read successfully! ";
        }

        // Build the interpreter
        tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::InterpreterBuilder(*m_model, resolver)(&m_interpreter);

        if (!m_interpreter) {
            // Handle the error
            ErrorMessage(m_logger) << "\nCould not build interpreter";
            return;
        } else {
            DebugMessage(m_logger) << "\nBuilt interpreter successfully! ";
        }
        
        if (num_threads != 1) {
            m_interpreter->SetNumThreads(num_threads);
        }
    }

    void InitCoreMlDelegate(int coreMLVersion) {
        TfLiteCoreMlDelegateOptions coreMlDelegateOptions;
        
        if (coreMLVersion >= 1) {
            coreMlDelegateOptions.coreml_version = coreMLVersion;
            
            m_coreMl_delegate = TfLiteCoreMlDelegateCreate(&coreMlDelegateOptions);
            
            TfLiteStatus delegate_status = m_interpreter->ModifyGraphWithDelegate(m_coreMl_delegate);
            if (delegate_status == kTfLiteOk) {
                DebugMessage(m_logger) << "\nTensorflow init coreMl successfully";
            } else {
                ErrorMessage(m_logger) << "\nTensorflow cant init coreMl, delegate status = " << delegate_status;
            }
        }
    }

/*    void InitFlexDelegate() {
        // Add Flex delegate for TF ops (necessary for custom ops often used in training)
        auto* delegate = TfLiteFlexDelegateCreate();
        TfLiteStatus delegate_status = interpreter->ModifyGraphWithDelegate(delegate);
        if (delegate_status == kTfLiteOk) {
            DebugMessage(m_logger) << "\nTensorflow add Flex delegate successfully";
        } else {
            ErrorMessage(m_logger) << "\nTensorflow add Flex delegate failed, delegate status = " << delegate_status;
        }
    } */

    void InitInputs(const vector<vector<int>> & inputDims) {
        DebugMessage(m_logger) << "\nModel inputs size = " << m_interpreter->inputs().size() << ":";
        if (inputDims.size() != m_interpreter->inputs().size()) {
            ErrorMessage(m_logger) << "\nWrong dims sizes";
        }

        for (unsigned i = 0; i < m_interpreter->inputs().size(); i++)  {
            int size = m_interpreter->tensor(m_interpreter->inputs()[i])->bytes / sizeof(float);
            DebugMessage(m_logger) << ", " << size;

            m_interpreter->ResizeInputTensor(m_interpreter->inputs()[i], inputDims[i]);
        }
    }
    
    void InitRunnerInputsWithLabels(tflite::impl::SignatureRunner* runner, const map<string, vector<int>>& inputDims) {
       
        for (const auto& input : inputDims) {

            if (runner->ResizeInputTensor(input.first.c_str(), input.second) != kTfLiteOk) {
                ErrorMessage(m_logger) << "Failed to resize input tensor with name ";
                return;
            }
            
            DebugMessage(m_logger) << "\nResizeInputTensor " << input.first;
            for (int dim : input.second) {
                DebugMessage(m_logger) << dim;
            }
        }
        
        // Allocate tensors for the restore signature runner
        if (runner->AllocateTensors() != kTfLiteOk) {
            ErrorMessage(m_logger) << "Failed to allocate tensors for restore signature runner";
        }
        else DebugMessage(m_logger) << "\nAllocateTensors for runner succeeded";
    }
    
    void AllocateTensors() {
        if (m_interpreter->AllocateTensors() != kTfLiteOk) ErrorMessage(m_logger) << "\nAllocateTensors failed";
        else DebugMessage(m_logger) << "\nAllocateTensors succeeded";
    }

    tflite::impl::SignatureRunner* GetRunner(const string &runnerName)
    {
        // Get the signature runner for 'train'
        tflite::impl::SignatureRunner* runner = m_interpreter->GetSignatureRunner(runnerName.c_str());
        if (!runner) {
            ErrorMessage(m_logger) << "Failed to get signature runner for " << runnerName;
            return nullptr;
        }
        
        DebugMessage(m_logger) << "Get signature runner for for " << runnerName << " succeeded";
        return runner;
    }

public:
    ComputationalModel(
        const char* modelFileName,
        const vector<vector<int>> & inputDims,
        int num_threads,
        int loggerSeverity,
        int coreMLVersion) :
        m_logger(make_shared<Logger>("Mudra", (Logger::Severity)loggerSeverity))
    {
        DebugMessage(m_logger) << "\nStart TensorFlow 2.16 with coreML support init function on " << modelFileName;
        DebugMessage(m_logger) << "\nnumOfThreads = " << num_threads;
        DebugMessage(m_logger) << "\ncoreMLVersion = " << coreMLVersion;

        InitInterpreter(modelFileName, num_threads);
        InitCoreMlDelegate(coreMLVersion);
        InitInputs(inputDims);
        AllocateTensors();

        m_outputSizes.resize(m_interpreter->outputs().size());
        DebugMessage(m_logger) << "\nModel outputSizes " << m_interpreter->outputs().size() << ":";
        for (unsigned i = 0; i < m_interpreter->outputs().size(); i++)
        {
            m_outputSizes[i] = m_interpreter->tensor(m_interpreter->outputs()[i])->bytes / sizeof(float);
            DebugMessage(m_logger) << ", " << m_outputSizes[i];
        }
    }
    
    ComputationalModel(
        const char* modelFileName,
        const char* weightsFileName,
        int num_threads,
        int loggerSeverity) :
        m_logger(make_shared<Logger>("Mudra", (Logger::Severity)loggerSeverity))
    {
        DebugMessage(m_logger) << "\nStart TensorFlow 2.16 with on device training support init function on " << modelFileName << ", weights file : " << weightsFileName;
        DebugMessage(m_logger) << "\nnumOfThreads = " << num_threads;
        m_WeightsFileName = weightsFileName;

        InitInterpreter(modelFileName, num_threads);
        Restore();
        AllocateTensors();
        
    }
    
    ~ComputationalModel() {
        TfLiteCoreMlDelegateDelete(m_coreMl_delegate);
    }

    void Run(const vector<vector<float>>& inputs, std::vector<std::vector<float>>& outputs)
    {
        DebugMessage(m_logger) << "\nRun time input size " << inputs.size() << ":";
        
        for (unsigned i = 0; i < inputs.size(); i++)
        {
            DebugMessage(m_logger) << "\ninputs[" << i << "]=" << inputs[i].size();
            
            float* tensorInput = m_interpreter->typed_input_tensor<float>(i);
            
            DebugMessage(m_logger) << "\nBefore copying " << i;
            std::copy(inputs[i].begin(), inputs[i].end(), tensorInput);
            DebugMessage(m_logger) << "\nAfter copying " << i;
        }
        
        if (m_interpreter->Invoke() != kTfLiteOk) {
            ErrorMessage(m_logger) << "\nInvoke failed";
        }
        else
        {
            DebugMessage(m_logger) << "\nInvoke successfully";
        }
        
        outputs.resize(m_outputSizes.size());
        for (unsigned i = 0; i < outputs.size(); i++)
        {
            float* tensorOutput = m_interpreter->typed_output_tensor<float>(i);
            
            outputs[i].assign(tensorOutput, tensorOutput + m_outputSizes[i]);
            DebugMessage(m_logger) << "outputs[" << i << "]=" << outputs[i].size();
        }
    }

    template<typename T>
    void CopyDataToTensor(TfLiteTensor* input_tensor,
                                              const vector<T>& input_data) {
      switch (input_tensor->type) {
        case kTfLiteFloat32: {
          float* dest = input_tensor->data.f;
          for (size_t i = 0; i < input_data.size(); i++) {
            dest[i] = static_cast<float>(input_data[i]);
          }
          break;
        }

        case kTfLiteInt32: {
          int32_t* dest = input_tensor->data.i32;
          for (size_t i = 0; i < input_data.size(); i++) {
            dest[i] = static_cast<int32_t>(input_data[i]);
          }
          break;
        }

        case kTfLiteUInt8: {
          uint8_t* dest = input_tensor->data.uint8;
          for (size_t i = 0; i < input_data.size(); i++) {
            dest[i] = static_cast<uint8_t>(input_data[i]);
          }
          break;
        }

        case kTfLiteInt8: {
          int8_t* dest = input_tensor->data.int8;
          for (size_t i = 0; i < input_data.size(); i++) {
            dest[i] = static_cast<int8_t>(input_data[i]);
          }
          break;
        }

        case kTfLiteInt64: {
          int64_t* dest = input_tensor->data.i64;
          for (size_t i = 0; i < input_data.size(); i++) {
            dest[i] = static_cast<int64_t>(input_data[i]);
          }
          break;
        }

        case kTfLiteBool: {
          bool* dest = input_tensor->data.b;
          for (size_t i = 0; i < input_data.size(); i++) {
            dest[i] = static_cast<bool>(input_data[i]);
          }
          break;
        }

        case kTfLiteInt16: {
          int16_t* dest = input_tensor->data.i16;
          for (size_t i = 0; i < input_data.size(); i++) {
            dest[i] = static_cast<int16_t>(input_data[i]);
          }
          break;
        }

        case kTfLiteFloat64: {
          double* dest = input_tensor->data.f64;
          for (size_t i = 0; i < input_data.size(); i++) {
            dest[i] = static_cast<double>(input_data[i]);
          }
          break;
        }

        case kTfLiteUInt16: {
          uint16_t* dest = input_tensor->data.ui16;
          for (size_t i = 0; i < input_data.size(); i++) {
            dest[i] = static_cast<uint16_t>(input_data[i]);
          }
          break;
        }

        case kTfLiteResource:
        case kTfLiteVariant:
          ErrorMessage(m_logger)
              << "Unsupported tensor type: " << input_tensor->type;
          break;

        default:
          ErrorMessage(m_logger)
              << "Unknown tensor type: " << input_tensor->type;
          break;
      }
    }

    template<typename T>
    vector<void*> InvokeSignatureRunner(const char* name, const map<string, vector<int>>& inputDims,const map<string, vector<T>>& inputs, const vector<string>& outputNames){
       
        tflite::impl::SignatureRunner* runner = GetRunner(name);
        InitRunnerInputsWithLabels(runner,inputDims);
        for (const auto& input : inputs) {
            TfLiteTensor* input_tensor = runner->input_tensor(input.first.c_str());

            // Prepare input tensors
            CopyDataToTensor(input_tensor, input.second);
        }
        // Invoke the runner
        if (runner->Invoke() != kTfLiteOk) {
            ErrorMessage(m_logger) << "\nInvoke failed";
        } else{
            DebugMessage(m_logger) << "\nInvoke successfully";
        }

        vector<void*> outputs;
        for (const auto& outputName : outputNames) {
            outputs.push_back(runner->output_tensor(outputName.c_str())->data.data);
        }
        return outputs;
    }

    vector<void*> InvokeSignatureRunner(const char* name, const map<string, vector<int>>& inputDims, const map<string, vector<string>>& inputs, const vector<string>& outputNames) 
    {
        tflite::impl::SignatureRunner* runner = GetRunner(name);
        InitRunnerInputsWithLabels(runner, inputDims);

        // Prepare input tensors
        for (const auto& input : inputs) {
            TfLiteTensor* input_tensor = runner->input_tensor(input.first.c_str());
            
            tflite::DynamicBuffer buffer;
            for (int i = 0; i < input.second.size(); i++) {
                buffer.AddString(input.second[i].c_str(), input.second[i].size());
            }
            buffer.WriteToTensor(input_tensor, /*new_shape=*/nullptr);
        }

        // Invoke the runner
        if (runner->Invoke() != kTfLiteOk) {
            ErrorMessage(m_logger) << "\nInvoke failed";
        } else {
            DebugMessage(m_logger) << "\nInvoke successfully";
        }

        vector<void*> outputs;
        for (const auto& outputName : outputNames) {
            outputs.push_back(runner->output_tensor(outputName.c_str())->data.data);
        }
        return outputs;
    }
    
    void Train(const map<string, vector<float>>& inputs,const map<string, vector<int>>& inputDims)
    {


       vector<void*> loss = InvokeSignatureRunner<float>("train",inputDims,inputs,{"loss"});
     
        float* floatLoss = (float*) loss[0];
        DebugMessage(m_logger) << "output: " << floatLoss[0];
      
        Save();
    }

    void Save()
    {
        InvokeSignatureRunner("save", {{"checkpoint_path",{1}}}, {{"checkpoint_path",{m_WeightsFileName}}}, {});
        // Get the signature runner for 'save'
       
    }

    void Restore() {

        InvokeSignatureRunner("restore", {{"checkpoint_path",{1}}}, {{"checkpoint_path",{m_WeightsFileName}}}, {});
    }
};

void InitTensorflowModel(
    const char* modelFileName,
    const vector<vector<int>>& inputDims,
    int loggerSeverity,
    int numOfThreads,
    int coreMLVersion)
{
    g_model[modelFileName] = make_unique<ComputationalModel>(modelFileName, inputDims, numOfThreads, loggerSeverity, coreMLVersion);
}

void InitTensorflowTrainingModel(
    const char* modelFileName,
    const char* weightsFileName,
    int loggerSeverity,
    int numOfThreads)
{
    g_model[modelFileName] = make_unique<ComputationalModel>(modelFileName, weightsFileName, numOfThreads, loggerSeverity);
}

//Rough implementation of training model until We know sizes

void RunTensorflowModel(
    const char* graphFileName,
    const vector<vector<float>>& inputs,
    vector<vector<float>>& outputs)
{
    g_model[graphFileName]->Run(inputs, outputs);
}
 
void DeleteTensorflowModel(const char* graphFileName)
{
    g_model[graphFileName].reset();
}

void TrainTensorflowModel(
    const char* modelFileName,
    const map<string, vector<int>>& inputDims,
    const map<string, vector<float>>& inputs)
{
    g_model[modelFileName]->Train(inputs,inputDims);
}
