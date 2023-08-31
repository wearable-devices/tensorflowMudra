#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <memory>
#include <functional>

#include "Constants.h"

namespace Mudra
{
	namespace Computation
	{
		typedef unsigned char Byte; //uint8_t

		typedef float Quaternion[4];

		typedef std::vector<float> BufferType;

		struct SncPackageData
		{
			unsigned int timeStamp;
			BufferType Snc[NUMBER_OF_SNCS];

			SncPackageData() : timeStamp(0) {}
		};

		struct ImuPackageData
		{
			unsigned int timeStamp;
			BufferType data;
			
			ImuPackageData() : timeStamp(0) {}
		};

		struct ImuQuaternionPackageData
		{
			unsigned int timeStamp;
			std::vector<BufferType> data;
			std::vector<BufferType> eulerData;

			ImuQuaternionPackageData() : timeStamp(0) {}
		};
    
    struct TensorFlowPackageData
    {
        std::vector<BufferType> data;

        TensorFlowPackageData(){}
    };

		enum class  RunMode
		{
			Inference, Calibration, OnlyParsingValues
		};
		enum class ImuType
		{
			None = -1,
			AccNorm = 1,
			Quaternion = 11
		};

		enum class GestureType
		{
			None = 0 ,
			MiddleTap = 1,
			IndexTap = 2,
            ThumbTap = 3,
			ModelLength = 4,
			Twist = 4,
			DoubleIndexTap = 5,
            DoubleMiddleTap = 6,
            SwipeLeft = 7,
            SwipeRight = 8,
            LongPress = 9,
			Length = 10
		};

		enum class HandType
		{
			Left,
			Right,
			Length
		};

		enum class WindowQualityType
		{
			Good,
			Bad,
			NoWindow
		};

		enum class LicenseType
		{
            Main,
			RawData,
			TensorFlowData,
			DoubleTap
		};

		typedef std::function< void(GestureType gestureType)>		OnGestureReadyCallBackType;
		typedef std::function< void(float proportional)>			OnProportionalReadyCallBackType;
		typedef std::function< void(SncPackageData &data)>			OnSncPackageReadyCallbackType;
		typedef std::function< void(ImuPackageData&)>				OnImuPackageReadyCallbackType;
		typedef std::function< void(ImuQuaternionPackageData&)>		OnImuQuaternionPackageReadyCallbackType;
		typedef std::function< void(const std::string&)>			OnCalibrationChangedCallbackType;
		typedef std::function< void(GestureType, const std::vector<SncPackageData> &, WindowQualityType, float width, float midpoint, float energy)> OnGestureCalibrationWindowReadyCallBackType;
		typedef std::function< void(const BufferType &)>	OnAirMousePositionChangedCallbackType;
		typedef std::function< void(std::vector<BufferType> &)>	OnTensorFlowDataReadyCallBackType;
		typedef std::function< void(const std::string &msg)>		OnLoggingMessageCallBackType;
	}
}
