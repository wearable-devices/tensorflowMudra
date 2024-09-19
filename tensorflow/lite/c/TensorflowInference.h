#pragma  once

#include <vector>
#include <string>


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
    int loggerSeverity,
    int numOfThreads);

extern "C" void DeleteTensorflowModel(const char* graphFileName);


        
 
