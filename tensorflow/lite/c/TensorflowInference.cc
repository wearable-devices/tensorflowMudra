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
    string m_savedWeightsFileName;
    vector<int> m_outputSizes;

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
                ErrorMessage(m_logger) << "Failed to resize input tensor";
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
        const map<string, vector<int>>& inputDims,
        int num_threads,
        int loggerSeverity) :
        m_logger(make_shared<Logger>("Mudra", (Logger::Severity)loggerSeverity))
    {
        DebugMessage(m_logger) << "\nStart TensorFlow 2.16 with on device training support init function on " << modelFileName << ", weights file : " << weightsFileName;
        DebugMessage(m_logger) << "\nnumOfThreads = " << num_threads;
        m_savedWeightsFileName = weightsFileName;

        InitInterpreter(modelFileName, num_threads);
        tflite::impl::SignatureRunner* runner = GetRunner("train");
        InitRunnerInputsWithLabels(runner, inputDims);
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
    
	void Train(const map<string, vector<float>>& inputs)
	{
        DebugMessage(m_logger) << "\nTrain" << inputs.size() << ":";

        tflite::impl::SignatureRunner* runner = GetRunner("train");
        
		// Prepare input tensors
		for (const auto& input : inputs) {
            DebugMessage(m_logger) << "\nCopying " << input.first;
			TfLiteTensor* input_tensor = runner->input_tensor(input.first.c_str());
			std::copy(input.second.begin(), input.second.end(), input_tensor->data.f);
		}

		// Invoke the runner
		if (runner->Invoke() != kTfLiteOk) {
            ErrorMessage(m_logger) << "\nInvoke failed";
		} else{
            DebugMessage(m_logger) << "\nInvoke successfully";
        }

//		const TfLiteTensor* output_tensor = runner->output_tensor(0);  // TODO: Confirm output tensor index/name with Leeor
		//std::vector<float> loss =  output_tensor;
		//std::cout << "Loss: " << loss << std::endl;

		// TODO: Implement SaveWeights function and call it here
		// SaveWeights(output);
	}

    void Save()
    {
        tflite::impl::SignatureRunner* saver = GetRunner("train");
        
        if (saver->Invoke() != kTfLiteOk) {
            ErrorMessage(m_logger) << "Failed to invoke save signature runner" << std::endl;
            return;
        }

        // Get the output string
        const TfLiteTensor* output_tensor = saver->output_tensor(0);
        if (!output_tensor) {
            ErrorMessage(m_logger) << "Failed to get output tensor from save signature runner" << std::endl;
            return;
        }

        // Convert the output tensor to a string
        std::string saved_model_path(reinterpret_cast<const char*>(output_tensor->data.raw), output_tensor->bytes);
        
        DebugMessage(m_logger) << "Model saved to: " << saved_model_path << std::endl;

        // Get the home directory for iOS
        const char* home_dir = getenv("HOME");
        if (!home_dir) {
            ErrorMessage(m_logger) << "Failed to get HOME environment variable" << std::endl;
            return;
        }

        // Construct the full path for the .ckpt file
        std::string full_path = std::string(home_dir) + "/" + m_savedWeightsFileName + ".ckpt";

        // Save the output data to a .ckpt file
        std::ofstream outFile(full_path, std::ios::binary);
        if (!outFile) {
            ErrorMessage(m_logger) << "Failed to open file for writing saved model data: " << full_path << std::endl;
            return;
        }

        // Write the size of the data
        size_t dataSize = output_tensor->bytes;
        outFile.write(reinterpret_cast<const char*>(&dataSize), sizeof(size_t));

        // Write the actual data
        outFile.write(reinterpret_cast<const char*>(output_tensor->data.raw), dataSize);
        if (!outFile) {
            ErrorMessage(m_logger) << "Failed to write saved model data to file: " << full_path << std::endl;
        } else {
            DebugMessage(m_logger) << "Saved model data written to: " << full_path << std::endl;
        }

        outFile.close();
    }


    void Restore() {
        tflite::impl::SignatureRunner* restore = GetRunner("restore");
        
        // Get the input tensor for the checkpoint path
        TfLiteTensor* input_tensor = restore->input_tensor("checkpoint_path");
        if (!input_tensor) {
            ErrorMessage(m_logger) << "Failed to get input tensor" << std::endl;
            return;
        }

        // Set the checkpoint path
        std::string checkpoint_path = m_savedWeightsFileName + ".ckpt";
        tflite::DynamicBuffer buffer;
        buffer.AddString(checkpoint_path.c_str(), checkpoint_path.length());
        buffer.WriteToTensor(input_tensor, /*new_shape=*/nullptr);

        // Invoke the restore operation
        TfLiteStatus status = restore->Invoke();
        if (status != kTfLiteOk) {
            ErrorMessage(m_logger) << "Failed to invoke restore signature runner" << std::endl;
        } else {
            DebugMessage(m_logger) << "Weights restored successfully" << std::endl;
        }
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
    const map<string, vector<int>>& inputDims,
    int loggerSeverity,
    int numOfThreads)
{
    g_model[modelFileName] = make_unique<ComputationalModel>(modelFileName, weightsFileName, inputDims, numOfThreads, loggerSeverity);
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
                          const map<string, vector<float>>& inputs)
{
    g_model[modelFileName]->Train(inputs);
}
