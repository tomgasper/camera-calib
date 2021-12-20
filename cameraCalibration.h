#pragma once
#include <vector>
#include <opencv2/core/types.hpp>
#include <math.h>
#include <stdexcept>
#include <fstream>


namespace MY
{
	class CameraCalibration
	{
	private:
		cv::Point3f p;
		float x;
		float y;
		float z;
	public:
		bool ReadDataFromFile(std::string modelFile, std::string imageFile, std::vector<std::vector<cv::Point2f>>& i_pts_arr, std::vector<std::vector<cv::Point3f>>& w_pts_arr);
		template<typename T1>
		void CalculateNormMatrix(std::vector<T1>, cv::Mat&);
		cv::Point2d EstRadialDisplacement(cv::Mat&, std::vector<cv::Mat>&, std::vector<std::vector<cv::Point3f>>&, std::vector<std::vector<cv::Point2f>>&);
		void ComposeParamVec(cv::Mat&, cv::Point2d&, std::vector<cv::Mat>&, cv::Mat&);

		void FindPairs(std::string imgs_dir, int C_WIDTH, int C_HEIGHT, std::vector<std::vector<cv::Point3f>>& wPoints, std::vector<std::vector<cv::Point2f>>& iPoints);
		bool FindHomography(std::vector<cv::Point3f>&, std::vector<cv::Point2f>&, std::vector<std::vector<float>>&);
		template<typename T1>
		bool vector_to_mat(std::vector<std::vector<T1>>& v, cv::Mat& mat);
		template<typename T1>
		bool vector_to_mat_double(std::vector<std::vector<T1>>& v, cv::Mat& mat);
		void Build_L(std::vector<cv::Point2f> m, std::vector<cv::Point3f> M, std::vector<std::vector<float>>& L);
		void Build_V(std::vector<std::vector<std::vector<float>>>&, std::vector<std::vector<float>>&);
		template<typename T1>
		void mat_to_vec(cv::Mat& mat, std::vector<std::vector<T1>>& v);

		void FindCameraIntrinsics(std::vector<std::vector<std::vector<float>>>&, cv::Mat&);
		void ExtractIntrinsicParams(std::vector<double>& b, cv::Mat& K_out);
		void ExtractViewParams(cv::Mat&, std::vector<std::vector<float>>&, cv::Mat&);
		void RefineParams(cv::Mat& A, cv::Point2d& k, std::vector<cv::Mat>& W_arr, std::vector<std::vector<cv::Point3f>>& worldPoints, std::vector<std::vector<cv::Point2f>>& imagePoints);
		void EIGEN_RefineParams(cv::Mat& A, cv::Point2d& k, std::vector<cv::Mat>& W_arr, std::vector<std::vector<cv::Point3f>>& worldPoints, std::vector<std::vector<cv::Point2f>>& imagePoints);

	friend std::ifstream& operator>> (std::ifstream& input, cv::Point3f& p)
	{
		char char1, char2, char3, char4;
		input >> char1 >> p.x >> char2 >> p.y >> char3;

		if (input.fail()) throw std::invalid_argument("bad_istream");
		if (char1 == '[' && char2 == ',' && char3 == ']')
			return input;
		else
			throw std::invalid_argument("bad_input_Point");
	}
	};

	
}