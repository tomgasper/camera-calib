#pragma once
#include <vector>
#include <opencv2/core/types.hpp>
#include <math.h>


namespace MY
{
	class CameraCalibration
	{
	public:
		template<typename T1>
		void CalculateNormMatrix(std::vector<T1>, cv::Mat&);
		cv::Point2f EstRadialDisplacement(cv::Mat&, std::vector<cv::Mat>&, std::vector<std::vector<cv::Point3f>>&, std::vector<std::vector<cv::Point2f>>&);
		cv::Point3f ToRodriguezVec(cv::Mat& R);
		void ComposeParamVec(cv::Mat&, cv::Point2f&, std::vector<cv::Mat>&, cv::Mat&);

		void FindPairs(int C_WIDTH, int C_HEIGHT, std::vector<std::vector<cv::Point3f>>& wPoints, std::vector<std::vector<cv::Point2f>>& iPoints);
		bool FindHomography(std::vector<cv::Point3f>&, std::vector<cv::Point2f>&, std::vector<std::vector<float>>&);
		bool vector_to_mat(std::vector<std::vector<float>>& v, cv::Mat& mat);
		void Build_L(std::vector<cv::Point2f> m, std::vector<cv::Point3f> M, std::vector<std::vector<float>>& L);
		void Build_V(std::vector<std::vector<std::vector<float>>>&, std::vector<std::vector<float>>&);
		void mat_to_vec(cv::Mat& mat, std::vector<std::vector<float>>& v);

		void FindCameraIntrinsics(std::vector<std::vector<std::vector<float>>>&, cv::Mat&);
		void ExtractIntrinsicParams(std::vector<float>& b, cv::Mat& K_out);
		void ExtractViewParams(cv::Mat&, std::vector<std::vector<float>>&, cv::Mat&);
		void RefineParams(cv::Mat& A, cv::Point2f& k, std::vector<cv::Mat>& W_arr, std::vector<std::vector<cv::Point3f>>& worldPoints, std::vector<std::vector<cv::Point2f>>& imagePoints);
	};
}