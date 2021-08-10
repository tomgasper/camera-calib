#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/calib3d.hpp>

#include <vector>

#include "CameraCalibration.h"

using namespace cv;

MY::CameraCalibration cameraCalib;

int main()
{
	// use dual degenerate conic c*_infinite

	// camera modeled using usual pinhole
	
	// relationship between point M and m;
	// m = K[R | t]M;
	// or s*m_aug( = A[R t]M_aug

	// [R t]-extrinsinc params that relate world coord sys to the camera sys

	// A =  [ a  g  u_0 ]
	//		[ 0  b  v_0 ]
	//		[ 0  0   1  ]
	//

	// u_0, v_0 coords of principle point
	// a b - scale factors in image u and v axes - focal length
	// g(skew) not needed

	// note A^-T = (A^-1)^T or (A^T)^-1
	// also note that we are dropping Z since its assumed to be 0 in the world coord sys
	
	// A[ r_1 r_2 r_3 t ] * [ X Y 0 1 ]^T  ---> A[ r_1 r_2 t ]*[X Y 1]^T

	// using homography which means - two images of the same planar surface in space are related in someway
	// s*m_aug = H*M_aug where H = A[ r_1 r_2 t ]

	// [ r1 0 ]^T and [ r2 0 ]^T our circle points, two particular point at line/intersection of plane z =0 and plane at infinity

	// model plane described in the camera coords sys -> [ r3 r3^T*t ] [ x y z w ] = 0
	// w = 0 for points at infinity and w = 1 otherwise
	// this plane intersect the plane at infinity at a line
	// x_infinity = a*[ r1 0 ]^T + b*[ r2 0 ] = [ a*r_1 + b*r_2 ]  because by definition (x^T_infinity)*(x_infinity) = 0
 	//											[		0		]

	// a^2 + b^2 = 0 -> b = +- ai where i^2 = -1

	std::vector<std::vector<Point2f>> imagePoints;
	std::vector<std::vector<Point3f>> worldPoints;

	std::vector<std::vector<std::vector<float>>> H_arr;
	// now searching for real image pairs.. :)

	cameraCalib.FindPairs(6, 8, worldPoints, imagePoints);

	// input these pairs to findHomography fnc
	// loop through all images
	for (unsigned int i = 0; i < worldPoints.size(); i++)
	{
		std::vector<std::vector<float>> H;
		cameraCalib.FindHomography(worldPoints[i], imagePoints[i], H);
		H_arr.push_back(H);
	}

	// 3. Build build N * ( v_12 and (v_11 - v_22)^T ) * b now

	cv::Mat K(3,3,CV_64F);
	cameraCalib.FindCameraIntrinsics(H_arr, K);

	std::vector<cv::Mat> W_arr;
	for (unsigned int i = 0; i < H_arr.size(); i++)
	{
		cv::Mat W_i;
		cameraCalib.ExtractViewParams(K, H_arr[i], W_i);

		W_arr.push_back(W_i);
	}
	
	cv::Point2f k;
	k = cameraCalib.EstRadialDisplacement(K, W_arr, worldPoints, imagePoints);


	// closed form solution
	// B = A^-T*A^-1

	// h_i^T * B * h_j = v_ij^T * b;
	
	// v_ij = [	h_i1*h_j1, h_i1*h_j2 + h_i2*h_j1, h_i2*h_j2, h_i3*h_j1 + h_i1*h_j3, h_i3*h_j2 + h_i2*h_j3, h_i3*h_j3 ]^T

	return 0;
}