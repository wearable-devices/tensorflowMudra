#pragma once

namespace Mudra
{
	namespace Computation
	{
		// Sensors (SNC, IMU) constants
		static const unsigned NUMBER_OF_SNCS = 3;

		// Queue lengths
		static const unsigned NUM_OF_SNC_PACKAGES_IN_WINDOW = 16;
		static const unsigned NUM_OF_IMU_ACC_NORM_PACKAGES_IN_WINDOW = 8;
		static const unsigned NUM_OF_IMU_QUATERNION_PACKAGES_IN_WINDOW = 16;

		static const unsigned SNC_PACKAGE_SIZE = 18;
		static const unsigned IMU_ACC_NORM_PACKAGE_SIZE = 8;
		static const unsigned IMU_QUATERNION_PACKAGE_SIZE = 4;

		static const unsigned CONFIDENCE_GESTURE_WINDOW = 3;

		static const unsigned MIDDLE_GESTURE_WINDOW = 9;
        static const unsigned TWIST_GESTURE_WINDOW = 15;
		static const unsigned QUICK_DOUBLE_GESTURE_WINDOW = 23;
		static const unsigned SLOW_DOUBLE_GESTURE_WINDOW = 20;

		static const unsigned NUM_SAMPLES_SNC_RESIZED = 50;

		// protocol data size parameters in bytes
		static const unsigned SAMPLE_SIZE = 2;
		static const unsigned TIMESTAMP_SIZE = 4;

		static const unsigned IMU_HEADER_SIZE = 1;

		static const unsigned NUM_OF_QUATERNION_ELEMENTS = 4;


		static const float	  QUATERNION_COEFFICIENT_B0 = 0.291870087908281f;
		static const float	  QUATERNION_COEFFICIENT_B1 = 0.291870087908281f;
		static const float	  QUATERNION_COEFFICIENT_A1 = -0.416259824183439f;

		// Algorithm constants
		static const float    GESTURE_THRESHOLD_HIGH = 0.8f;
		static const float    GESTURE_THRESHOLD_LOW = 0.2f;
		static const float	  NOMINAL_GESTURE_WIDTH = 0.45f;

		static const unsigned NUMBER_OF_CALIBRATION_ITERATIONS = 5;
		static const unsigned K_NEAREST_NEIGHBOUR = 6; // floor(sqrt(num_calibration))

	}
}

