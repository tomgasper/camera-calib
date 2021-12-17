#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>
#include <cstdio>
#include "CameraCalibration.h"

using namespace cv;
MY::CameraCalibration cameraCalib;

	// Camera modeled using usual pinhole and dual degenerate conic c*_infinite mapping(Zhang's method)
	// Relationship between point M and m;
	// m = K[R | t]M;
	// or s*m_aug = A[R t]M_aug

	// [R t]-extrinsinc params that relate world coord sys to the camera sys

	// A =  [ a  g  u_0 ]
	//		[ 0  b  v_0 ]
	//		[ 0  0   1  ]

	// u_0, v_0 coords of principle point
	// a b - scale factors in image u and v axes - focal length
	// g(skew) not needed

	// Using homography which means - two images of the same planar surface in space are related in someway
	// we have
	// s*m_aug = H*M_aug where H = A[ r_1 r_2 t ]

	// note that we are dropping Z since its assumed to be 0 in the world coord sys
	// A[ r_1 r_2 r_3 t ] * [ X Y 0 1 ]^T  ---> A[ r_1 r_2 t ]*[X Y 1]^T

	// in this case points on a flat surface(f.e black and white rectangle pattern on a paper) are related to sensor points
	// both are images of the same planar surface in space

	// [ r1 0 ]^T and [ r2 0 ]^T circle points, two particular points at line/intersection of plane z=0 and plane at infinity

	// model plane described in the camera coords sys -> [ r3 r3^T*t ] [ x y z w ] = 0
	// w = 0 for points at infinity and w = 1 otherwise
	// this plane intersect the plane at infinity at a line
	// x_infinity = a*[ r1 0 ]^T + b*[ r2 0 ] = [ a*r_1 + b*r_2 ]  because by definition (x^T_infinity)*(x_infinity) = 0
	//											[		0		]

	// a^2 + b^2 = 0 -> b = +- ai where i^2 = -1

int main(int ac, char* av[])
{
	// Hard coded input data
	const std::string IMGS_DIR = "./data/";
	const int CHECKERBOARD_WIDTH(8);
	const int CHECKERBOARD_HEIGHT(6);

	bool imagePairsProvided = true;
	const std::string MODEL_POINTS_DIR = "./data/model_points.txt";
	const std::string IMAGE_POINTS_DIR = "./data/image_points.txt";

	std::vector<std::vector<Point2f>> imagePoints;
	std::vector<std::vector<Point3f>> worldPoints;
	std::vector<std::vector<std::vector<float>>> H_arr;

	if (imagePairsProvided)
	{
		bool dataRead = cameraCalib.ReadDataFromFile(MODEL_POINTS_DIR, IMAGE_POINTS_DIR, imagePoints, worldPoints);

		if (!dataRead) throw std::invalid_argument("Data not read");
	}
	else
	{
		cameraCalib.FindPairs(IMGS_DIR, CHECKERBOARD_WIDTH, CHECKERBOARD_HEIGHT, worldPoints, imagePoints);
	}

	// Input found pairs to find homography matrix for each view
	for (unsigned int i = 0; i < worldPoints.size(); i++)
	{
		// Loop through all images
		std::vector<std::vector<float>> H;
		std::cout << "FindHomography: " << "-------VIEW " << i << "-------" << std::endl;
		cameraCalib.FindHomography(worldPoints[i], imagePoints[i], H);
		
		H_arr.push_back(H);
	}

	// Look for Intrinsics Parameters
	// (Intrinsics Parameters denote the conversion from 3d camera coordinates to camera sensor)
	// K = [ fx, y, u_0 ]
	//     [ 0, fy, v_0 ]
	//     [ 0,  0,  1  ]
	// where fx, fy - focal length (in pixels) u_0, v_0 - principle point coordinates, y - skew between x and y axes

	cv::Mat K(3,3,CV_64F);
	cameraCalib.FindCameraIntrinsics(H_arr, K);

	// Look for Extrinsinc Parameters for each view
	// ( Extrinsic Parameters denote the conversion form world 3d coordinates system to 3d camera coordinates )
	// ( Note that t is the position of the origin of the world coord system expressed in camera(!) coordinate system )
	// ( Position of camera - C is in world coordinate system C = -R^-1*t = -R^T * t )
	// ( as we are solving 0 = RC * T )
	// W = [               ] 
	//	   [ r_1 r_2 r_3 t ]
	//	   [               ]
	//     [               ] 
	

	std::vector<cv::Mat> W_arr;
	for (unsigned int i = 0; i < H_arr.size(); i++)
	{
		cv::Mat W_i;
		cameraCalib.ExtractViewParams(K, H_arr[i], W_i);

		W_arr.push_back(W_i);
	}
	
	cv::Point2d k;
	k = cameraCalib.EstRadialDisplacement(K, W_arr, worldPoints, imagePoints);

	cameraCalib.RefineParams(K, k, W_arr, worldPoints, imagePoints);

	std::cout << "----------------------------" << std::endl;
	std::cout << "FindHomography: " << "Refined parameters" << std::endl;
	std::cout << "----------------------------" << std::endl;

	std::cout << "FindHomography: " << "K" << std::endl;
	std::cout << K << std::endl;

	std::cout << "FindHomography: " << "k" << std::endl;
	std::cout << k << std::endl;

	for (int i = 0; i < W_arr.size(); i++)
	{
		std::cout << "----------------------------" << std::endl;
		std::cout << "FindHomography: " << "W #" << i << std::endl;
		std::cout <<  W_arr[i] << std::endl;
	}

	
	std::getchar();

	return 0;
}