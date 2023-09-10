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

    vector<int> m_outputSizes;

public:

	ComputationalModel(
		const char* modelFileName,
		const vector<vector<int>> & inputDims,
		int num_threads,
		int loggerSeverity,
        int coreMLVersion) :
		m_logger(make_shared<Logger>("Mudra", (Logger::Severity)loggerSeverity))
	{
        DebugMessage(m_logger) << "\nStart TensorFlow 2.14 with coreML support init function on " << modelFileName;
        DebugMessage(m_logger) << "\nnumOfThreads = " << num_threads;
        DebugMessage(m_logger) << "\ncoreMLVersion = " << coreMLVersion;

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

		if (num_threads != 1) {
			m_interpreter->SetNumThreads(num_threads);
		}

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
        
		DebugMessage(m_logger) << "\nModel inputs size = " << m_interpreter->inputs().size() << ":";
		if (inputDims.size() != m_interpreter->inputs().size())
		{
			ErrorMessage(m_logger) << "\nWrong dims sizes";
		}

		for (unsigned i = 0; i < m_interpreter->inputs().size(); i++)
		{
			int size = m_interpreter->tensor(m_interpreter->inputs()[i])->bytes / sizeof(float);
			DebugMessage(m_logger) << ", " << size;

			m_interpreter->ResizeInputTensor(m_interpreter->inputs()[i], inputDims[i]);
		}

		if (m_interpreter->AllocateTensors() != kTfLiteOk) ErrorMessage(m_logger) << "\nAllocateTensors failed";
		else DebugMessage(m_logger) << "\nAllocateTensors succeeded";

		m_outputSizes.resize(m_interpreter->outputs().size());
		DebugMessage(m_logger) << "\nModel outputSizes " << m_interpreter->outputs().size() << ":";
		for (unsigned i = 0; i < m_interpreter->outputs().size(); i++)
		{
			m_outputSizes[i] = m_interpreter->tensor(m_interpreter->outputs()[i])->bytes / sizeof(float);
			DebugMessage(m_logger) << ", " << m_outputSizes[i];
		}
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
};

void InitTensorflowModel(
	const char* graphFileName,
	const vector<vector<int>>& inputDims,
	int loggerSeverity,
	int numOfThreads,
    int coreMLVersion)
{
    g_model[graphFileName] = make_unique<ComputationalModel>(graphFileName, inputDims, numOfThreads, loggerSeverity, coreMLVersion);
}

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
