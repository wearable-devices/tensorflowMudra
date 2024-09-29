#pragma  once

#include <vector>
#include <string>
#include <map> 

#ifdef _WINDOWS
#include "../Computation/Logging.h"
#else
#include "Logging.h"
#endif

extern "C" void InitTensorflowModel(
    const char* graphFileName,
    const vector<vector<int>> & inputDims,
    int loggerSeverity,
    int numOfThreads,
    int coreMLVersion);
        
extern "C" void RunTensorflowModel(
    const char* graphFileName,
    const vector<vector<float>>& inputs,
    vector<vector<float>>& outputs);

extern "C" void InitTensorflowTrainingModel(
    const char* modelFileName,
    const char* weightsFileName,
    const map<string, vector<int>>& inputDims,
    int loggerSeverity,
    int numOfThreads);

extern "C" void DeleteTensorflowModel(const char* graphFileName);

extern "C" void TrainTensorflowModel(
    const char* modelFileName,
    const map<string, vector<float>>& inputs);
        
 
