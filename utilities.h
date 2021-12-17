#pragma once

#include <opencv2/opencv.hpp>
using namespace cv;


void jacobian_fnc(cv::Mat J, cv::Mat& X, cv::Mat& P);
void error_fnc(Mat& err, cv::Mat& X, cv::Mat& U, cv::Mat& P);
void distort(cv::Mat& k, cv::Mat& hom_point);
cv::Point2d project_point(cv::Mat& a, cv::Mat& w, cv::Point3d world_point);
void unpack_params(cv::Mat& a, cv::Mat& w, cv::Mat& K_out, cv::Mat& Rt_out, cv::Mat& k_out);