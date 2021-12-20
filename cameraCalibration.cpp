#include "CameraCalibration.h"
#include "utilities.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include <opencv2/core/types_c.h>

#include <vector>
#include <fstream>
#include <iostream>
#include <assert.h>
#include <math.h>

#include <Eigen>
#include <unsupported/NonLinearOptimization>

#include "worldPointsData.h"
#include "imagePointsData.h"

using namespace MY;
using namespace cv;

CameraCalibration camCalib;

bool CameraCalibration::ReadDataFromFile(std::string modelFile, std::string imageFile, std::vector<std::vector<cv::Point2f>>& i_pts_arr, std::vector<std::vector<cv::Point3f>>& w_pts_arr)
{
	worldPointsData input_w_pts;
	imagePointsData input_i_pts;
	std::ifstream model_pts_stream, image_pts_stream;

	// read from model/target points file
	model_pts_stream.open(modelFile);
	while (!model_pts_stream.eof())
	{
		try
		{
			model_pts_stream >> input_w_pts;
		}
		catch (const std::invalid_argument& ia) {
			std::cerr << "Model points stream: " << ia.what() << '\n';
		}
	}
	model_pts_stream.close();

	// read from image points file
	image_pts_stream.open(imageFile);
	while (!image_pts_stream.eof())
	{
		try
		{
			image_pts_stream >> input_i_pts;
		}
		catch (const std::invalid_argument& ia) {
			std::cerr << "Image Points stream: " << ia.what() << '\n';
		}
	}
	image_pts_stream.close();


	std::vector<cv::Point3f> w_pts;
	input_w_pts.getData(w_pts);
	std::vector<cv::Point2f> i_pts;
	input_i_pts.getData(i_pts);

	int N_PAIRS = w_pts.size();
	int N_VIEWS = i_pts.size() / w_pts.size();

	// copy data to provided arrays
	for (int i = 0; i < N_VIEWS; i++)
	{
		w_pts_arr.push_back(w_pts);
	}

	
	for (int i = 0; i < N_VIEWS; i++)
	{
		std::vector<cv::Point2f> sub_arr;
		for (int j = 0; j < N_PAIRS; j++)
		{
			sub_arr.push_back(i_pts[(i * N_PAIRS) + j]);
		}
	i_pts_arr.push_back(sub_arr);
	}

	return true;
}

void CameraCalibration::FindPairs(std::string imgs_dir, int C_WIDTH, int C_HEIGHT, std::vector<std::vector<Point3f>>& wPoints, std::vector<std::vector<Point2f>>& iPoints)
{
	int CHECKERBOARD[2]{ C_WIDTH,C_HEIGHT };
	namedWindow("Camera Calibration");

	// now finally calibrate camera

	std::vector<Point3f> initWorldPoints;
	std::vector<std::vector<Point3f>> worldPoints;
	std::vector<std::vector<Point2f>> imgPoints;


	std::vector<String> imgs;

	std::string path = imgs_dir + "\*.jpg";

	// load all string dirs into specified vector list
	glob(path, imgs);

	// define world coords for 3d points
	std::ofstream m_points_stream;
	m_points_stream.open(".\\data\\model_points.txt");

	for (int i = 0; i < CHECKERBOARD[1]; i++)
	{
		for (int j = 0; j < CHECKERBOARD[0]; j++)
		{
			// assumed that the points are at X,Y,Z, where Z = 0 - > X,Y,0
			// [	1	0	0	0	]		[	X'	]
			// [	0	1	0	0	]	*	[	Y'	]
			// [	0	0	1	0	]		[	Z'	]
			//								[	1	]

			initWorldPoints.push_back(Point3f(j, i, 0));

			
			std::stringstream input;

			input << "[" << j << ","  << i << "," << 0 << "]";
			
			m_points_stream << input.str();
			m_points_stream << '\n';
		}
	}

	m_points_stream.close();

	// loop over all imgs
	std::vector<Point2f> corners;
	Size p_size(8, 6);
	Mat frame, gray, resized;
	bool success;

	std::ofstream w_points_stream;
	w_points_stream.open(".\\data\\image_points.txt");


	for (int i = 0; i < imgs.size(); i++)
	{
		frame = imread(imgs[i]);
		resize(frame, resized, cv::Size(), 0.15, 0.15);
		cvtColor(resized, gray, COLOR_BGR2GRAY);

		success = findChessboardCorners(gray, p_size, corners, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE
			| CALIB_CB_FAST_CHECK);

		if (success)
		{
			TermCriteria criteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.001);

			cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1), criteria);
			drawChessboardCorners(gray, Size(CHECKERBOARD[0], CHECKERBOARD[1]), corners, success);

			wPoints.push_back(initWorldPoints);
			iPoints.push_back(corners);

			// write to txt file
			for (int i = 0; i < corners.size(); i++)
			{
				std::stringstream ss;

				ss << "[" << std::setprecision(10) << corners[i].x << "," << std::setprecision(10) << corners[i].y << "]";
				ss << '\n';
				w_points_stream << ss.str();
			}
		}
		imshow("Camera Calibration", gray);
		cv::waitKey(10);
	}

	// cv::Mat cameraMatrix, distCoeffs, R, T;

	// cv::calibrateCamera(wPoints, iPoints, cv::Size(gray.rows, gray.cols), cameraMatrix, distCoeffs, R, T);
	w_points_stream.close();
}
void CameraCalibration::Build_L(std::vector<cv::Point2f> m, std::vector<cv::Point3f> M, std::vector<std::vector<float>>& L)
{
	// # of image pairs = m.size()/M.size()
	for (unsigned int i = 0; i < M.size(); i++)
	{
		std::vector<float> L_row_1{ M[i].x ,M[i].y, 1, 0, 0, 0, -(m[i].x * M[i].x), -(m[i].x * M[i].y),-m[i].x };
		std::vector<float> L_row_2{ 0,  0,   0, M[i].x,M[i].y, 1, -( m[i].y * M[i].x), -(m[i].y * M[i].y),-m[i].y };

		// add new rows to L-2d vector
		L.push_back(L_row_1);
		L.push_back(L_row_2);
	}
}

template<typename T1>
void CameraCalibration::CalculateNormMatrix(std::vector<T1> data, cv::Mat& N_mat_out)
{
	int N = data.size();
	float s_x, s_y;
	float x_mean = 0.;
	float y_mean = 0.;
	float x_var, y_var;
	
	// calculate mean
	for (int d = 0; d < N; d++)
	{
		x_mean += data[d].x;
		y_mean += data[d].y;
	}

	x_mean = x_mean / N;
	y_mean = y_mean / N;

	// calculate variance
	float cov_x = 0.;
	float cov_y = 0.;

	for (int i = 0; i < N; i++)
	{
		cov_x += (data[i].x - x_mean) * (data[i].x - x_mean);
		cov_y += (data[i].y - y_mean) * (data[i].y - y_mean);
	}

	cov_x = cov_x / N;
	cov_y = cov_y / N;

	s_x = sqrt(2. / cov_x);
	s_y = sqrt(2. / cov_y);

	float N_data[3][3] = {
		{ s_x,  0. ,   -s_x * x_mean  },
		{  0.,  s_y , -s_y * y_mean },
		{  0. ,   0. ,       1.       },
	};

	cv::Mat N_mat_in(3, 3, CV_32F, N_data);
	N_mat_in.copyTo(N_mat_out);

}

void CameraCalibration::ComposeParamVec(cv::Mat& A, cv::Point2d& k, std::vector<cv::Mat>& W_arr, cv::Mat& P)
{
	// CHANGE TO FLOAT64
	double a[7] { A.at<double>(0,0), A.at<double>(1,1), A.at<double>(0,1), A.at<double>(0,2), A.at<double>(1,2), k.x, k.y };

	// init Z with intrinsic values
	cv::Mat Z(1, 7, CV_64F, a);

	// copy Z to P mat accessed by a pointer

	for (unsigned int i = 0; i < W_arr.size(); i++)
	{
		cv::Mat R(3, 3, CV_64F);
		cv::Rect R_roi(0, 0, 3, 3);
		W_arr[i](R_roi).copyTo(R);

		cv::Mat p(1, 3, CV_64F);
		cv::Mat t(1, 3, CV_64F);

		t = W_arr[i].col(3).t();

		// rotation matrix to rodrigues vector
		cv::Rodrigues(R, p);

		// connect cols of rotation params and translation
		cv::Mat matArr[]{ Z, p.t(), t  };
		hconcat(matArr, 3,  Z);
	}

	Z.copyTo(P);
}



std::vector<float> Build_v(std::vector<std::vector<float>>& H, int i, int j)
{

	// variables for debugging reasons
	float val_1 = H[0][i] * H[0][j];
	float val_2 = (H[0][i] * H[1][j]) + (H[1][i] * H[0][j]);
	float val_3 = H[1][i] * H[1][j];
	float val_4 = (H[2][i] * H[0][j]) + (H[0][i] * H[2][j]);
	float val_5 = (H[2][i] * H[1][j] + H[1][i] * H[2][j]);
	float val_6 = H[2][i] * H[2][j];


	std::vector<float> v = { val_1, val_2, val_3, val_4, val_5 , val_6 };

	return v;

}

void CameraCalibration::Build_V(std::vector<std::vector<std::vector<float>>>& H, std::vector<std::vector<float>>& V)
{
	//			[	   v_12^T		]  
	//			[  (v_11 - v_22)^T  ]  * b  = 0
	//					...
	//					V			*    b  = 0

	for (unsigned int i = 0; i < H.size(); i++)
	{
		std::vector<float> v_11, v_12, v_22;
		std::vector<float> v_11_v_22;

		v_11 = Build_v(H[i], 0, 0);
		v_12 = Build_v(H[i], 0, 1);
		v_22 = Build_v(H[i], 1, 1);

		for (unsigned int j = 0; j < v_11.size(); j++)
		{
			v_11_v_22.push_back(v_11[j] - v_22[j]);
		}

		V.push_back(v_12);
		V.push_back(v_11_v_22);
	}
}
bool CameraCalibration::FindHomography(std::vector<Point3f>& wPoints, std::vector<Point2f>& iPoints, std::vector<std::vector<float>>& H_data)
{
	// x = [h1^T, h2^T, h3^T] <-- we are looking for x
	// we are going to solve it by finding eigenvectors of L^T*L
	// eigenvector corresponding to the smallest eigenvalue of L^T*L is the solution to L*x=0

	Mat eigenvectors(9, 9, CV_32F);
	Mat eigenvalues(9, 1, CV_32F);
	Mat s_eigenv(1, 9, CV_32F);

	// initialize normalization matrices
	cv::Mat N_mat_world(3, 3, CV_32F);
	cv::Mat N_mat_image(3, 3, CV_32F);

	camCalib.CalculateNormMatrix(iPoints, N_mat_image);
	camCalib.CalculateNormMatrix(wPoints, N_mat_world);

	std::vector<Point3f> wPoints_norm;
	std::vector<Point2f> iPoints_norm;

	for (int w = 0; w < wPoints.size(); w++)
	{
		// { s_x, 0., -s_x * x_mean   }     [ X ]
		// { 0.,  s_y , -s_y * y_mean } *   [ Y ]
		// { 0. ,   0. ,       1.     }     [ Z ]

		// Note that we are dropping Z=0 for every target point -> so P = [X,Y]^T and then we make it a homogenous coord -> P' = [X,Y,1]^T
		// and only after that we apply norm matrix
		cv::Point3f hom_w_p = { wPoints[w].x,wPoints[w].y, 1 };
		cv::Mat w_val_mat = N_mat_world*cv::Mat(hom_w_p, false);
		cv::Point3f p_w(w_val_mat);

		cv::Point3f i_hom = { iPoints[w].x, iPoints[w].y, 1 };
		cv::Mat i_val_mat = N_mat_image * cv::Mat(i_hom, false);
		cv::Point3f p_i(i_val_mat);

		wPoints_norm.push_back( p_w );
		iPoints_norm.push_back(cv::Point2f(p_i.x, p_i.y) );
	}

	// init data vector and openCV matrix to hold point data
	std::vector<std::vector<float>> L_data;
	Build_L(iPoints_norm, wPoints_norm, L_data);

	cv::Mat L_mat(0, L_data[0].size(), cv::DataType<float>::type);
	vector_to_mat(L_data, L_mat);

	// L^TL 9x9 matrix
	Mat LT_L_mat = L_mat.t() * L_mat;

	eigen(LT_L_mat, eigenvalues, eigenvectors);

	double s_eigenval = INFINITY;

	for (int i = 0; i < eigenvalues.rows; i++)
	{
		const double* eignevalues_row = eigenvalues.ptr<double>(i);
		double curr_eigenval = eignevalues_row[0];

		if (curr_eigenval < s_eigenval)
		{
			s_eigenval = i;
		}
	}

	if (s_eigenval <= eigenvectors.rows)
	{
		eigenvectors.row(s_eigenval).copyTo(s_eigenv.row(0));
	}

	// lets construct and return our newly found H matrix

	int const H_rows = 3;
	int const H_cols = 3;

	

	// Construct H_data
	for (int i = 0; i < H_rows; i++)
	{
		std::vector<float> subArr;
		H_data.push_back(subArr);
		for (int j = 0; j < H_cols; j++)
		{
			H_data[H_data.size() - 1].push_back(s_eigenv.at<float>(0, 3 * i + j));
		}
	}
	Mat check = L_mat * s_eigenv.t();

	// Denormalize

	cv::Mat H_norm(0, 3, CV_32F, DataType<float>::type);
	camCalib.vector_to_mat(H_data, H_norm);
	cv::Mat H_denorm_1 = H_norm * N_mat_world;
	cv::Mat H_denorm = N_mat_image.inv() * H_norm * N_mat_world;


	float X_data[3] = {wPoints[5].x, wPoints[5].y, wPoints[5].z };
	// convert to std::vector

	std::vector<std::vector<float>> H_denorm_out;

	// fill H_denorm_out vector
	for (int i = 0; i < H_denorm.rows; i++)
	{
		std::vector<float> new_row;
		H_denorm_out.push_back(new_row);
		const float* Mi = H_denorm.ptr<float>(i);
		for (int j = 0; j < H_denorm.cols; j++)
		{
			H_denorm_out[H_denorm_out.size() - 1].push_back(Mi[j]);
		}
	}

	H_data = H_denorm_out;

	// Print for calibrate.cpp
	std::cout << H_denorm << std::endl;
	std::cout << "" << std::endl;

	return true;

}

template<typename T1>
bool CameraCalibration::vector_to_mat(std::vector<std::vector<T1>>& v, cv::Mat& mat)
{
	// if (mat.empty() == false) return false;

	int v_row_width = v[0].size();
	for (unsigned int i = 0; i < v.size(); i++)
	{
		cv::Mat row(1, v_row_width, cv::DataType<float>::type, v[i].data());

		mat.push_back(row.clone());
	}
	return true;
}

template<typename T1>
bool CameraCalibration::vector_to_mat_double(std::vector<std::vector<T1>>& v, cv::Mat& mat)
{
	// if (mat.empty() == false) return false;

	int v_row_width = v[0].size();
	for (unsigned int i = 0; i < v.size(); i++)
	{
		cv::Mat row(1, v_row_width, cv::DataType<double>::type, v[i].data());

		mat.push_back(row.clone());
	}
	return true;
}


template<typename T1>
void CameraCalibration::mat_to_vec(cv::Mat& mat, std::vector<std::vector<T1>>& v)
{
	assert(v.empty());

	if (mat.type() != (int)5) mat.convertTo(mat, CV_32F);

	for (unsigned int m = 0; m < mat.rows; m++)
	{
		float* mat_row = mat.ptr<float>(m);

		std::vector<T1> empty_row;
		v.push_back(empty_row);

		for (unsigned int n = 0; n < mat.cols; n++)
		{
			v[v.size() - 1].push_back(mat_row[n]);
		}
	}
}

void CameraCalibration::ExtractIntrinsicParams(std::vector<double>& b, cv::Mat& K_out)
{
	// b = [ B_11, B_12, B_22, B_13, B_23, B_33]^T

	double B_11 = b[0];
	double B_12 = b[1];
	double B_22 = b[2];
	double B_13 = b[3];
	double B_23 = b[4];
	double B_33 = b[5];

	/*double v_0 = (B_12 * B_13 - B_11 * B_23) / (B_11 * B_22 - B_12 * B_12);
	double s = B_33 - ( B_13 * B_13 + v_0*(B_12 * B_13) - (B_11 * B_23) / B_11);
	double a = sqrt( s/ B_11 );
	double beta = sqrt( (s * B_11) / ((B_11 * B_22) - (B_12 * B_12)));
	double y = -B_12 * a * a * beta / s;
	double u_0 = ((y * v_0) / beta) - (B_13 * a * a / s);*/

	double w = (b[0] * b[2] * b[5]) - (b[1] * b[1] * b[5]) - (b[0] * b[4] * b[4]) + (2 * (b[1] * b[3] * b[4])) - (b[2] * b[3] * b[3]);
	double d = (b[0] * b[2]) - (b[1] * b[1]);

	double a = sqrt(w / (d * b[0]));
	double beta = sqrt(w / (d * d) * b[0]);
	double y = sqrt(w / (d * d * b[0])) * b[1];
	double u_0 = ((b[1] * b[4]) - (b[2] * b[3])) / d;
	double v_0 = ((b[1] * b[3]) - (b[0] * b[4])) / d;


	double K_data[3][3] = {
		{  a ,   y,  u_0 },
		{ 0.0, beta, v_0 },
		{ 0.0, 0.0,  1.0 }
	};

	cv::Mat K_in(3, 3, CV_64F, K_data);

	K_in.copyTo(K_out);
}

void CameraCalibration::FindCameraIntrinsics(std::vector<std::vector<std::vector<float>>>& H_arr, cv::Mat& K_out)
{
	std::vector<std::vector<float>> V_data;

	camCalib.Build_V(H_arr, V_data);

	// problem with V_mat
	Mat V_mat(0, V_data[0].size(), DataType<float>::type);
	camCalib.vector_to_mat(V_data, V_mat);

	V_mat.convertTo(V_mat, CV_64F);

	// find eigenvalue of V(T)V 1x6 matrix

	// s_eigenv: eigenvectors corresponding to the smallest found eigenvalue
	Mat VT_V = V_mat.t() * V_mat;
	Mat eigenvectors(6, 6, CV_64F);
	Mat eigenvalues(6, 1, CV_64F);
	Mat s_eigenv(1, 6, CV_64F);

	eigen(VT_V, eigenvalues, eigenvectors);

	double s_eigenval = INFINITY;
	int s_eigenval_indx = INFINITY;

	// Find the smallest eigenvalue
	for (int i = 0; i < eigenvalues.rows; i++)
	{
		const double* eignevalues_row = eigenvalues.ptr<double>(i);
		double curr_eigenval = eignevalues_row[0];

		if (curr_eigenval < s_eigenval)
		{
			s_eigenval = curr_eigenval;
			s_eigenval_indx = i;
		}
	}

	// Eigenvector corresponding to the smallest eigenvalue is our best fit
	if (s_eigenval_indx <= eigenvectors.rows)
	{
		eigenvectors.row(s_eigenval_indx).copyTo(s_eigenv.row(0));
	}

	// Finally, find K
	cv::Mat K(3, 3, CV_64F);
	std::vector<double> s_eigenv_data;
	s_eigenv_data.assign((double*)s_eigenv.datastart, (double*)s_eigenv.dataend);

	camCalib.ExtractIntrinsicParams(s_eigenv_data, K);

	std::cout << "FindCameraIntrinsics:: K: " << std::endl;
	std::cout << K << std::endl;
	std::cout << "" << std::endl;

	// temp -> delete after use

	float h0_temp[3] = { H_arr[0][0][0], H_arr[0][1][0], H_arr[0][2][0] };
	float h1_temp[3] = { H_arr[0][0][1], H_arr[0][1][1], H_arr[0][2][1] };

	cv::Mat h0_mat_temp(3, 1, CV_32F, h0_temp);
	cv::Mat h1_mat_temp(3, 1, CV_32F, h1_temp);

	cv::Mat K_inv = K.inv();

	/*float B_1[3] = { s_eigenv_data[0], s_eigenv_data[1], s_eigenv_data[3] };
	float B_2[3] = { s_eigenv_data[1], s_eigenv_data[2], s_eigenv_data[4] };
	float B_3[3] = { s_eigenv_data[3], s_eigenv_data[4], s_eigenv_data[5] };*/

	/*std::vector<float> B_1_d = { s_eigenv_data[0], s_eigenv_data[1], s_eigenv_data[3] };
	std::vector<float> B_2_d = { s_eigenv_data[1], s_eigenv_data[2], s_eigenv_data[4] };
	std::vector<float> B_3_d = { s_eigenv_data[3], s_eigenv_data[4], s_eigenv_data[5] };

	cv::Mat B;

	cv::Mat B_1(1, 3, CV_32F, B_1_d.data());
	cv::Mat B_2(1, 3, CV_32F, B_2_d.data());
	cv::Mat B_3(1, 3, CV_32F, B_3_d.data());

	B.push_back(B_1);
	B.push_back(B_2);
	B.push_back(B_3);*/

	K_inv.convertTo(K_inv, CV_32F);

	// end temp

	K.copyTo(K_out);
}

void CameraCalibration::ExtractViewParams(cv::Mat& K, std::vector<std::vector<float>>& H, cv::Mat& W_out)
{
	float h_0_data[]{ H[0][0], H[1][0], H[2][0] };
	float h_1_data[]{ H[0][1], H[1][1], H[2][1] };
	float h_2_data[]{ H[0][2], H[1][2], H[2][2] };

	cv::Mat h_0(3, 1, CV_32F, h_0_data);
	cv::Mat h_1(3, 1, CV_32F, h_1_data);
	cv::Mat h_2(3, 1, CV_32F, h_2_data);

	cv::Mat K_inv(3, 3, CV_32F);
	K_inv = K.inv();
	K_inv.convertTo(K_inv, CV_32F);

	cv::Mat K_inv_h_0(1, 3, CV_32F);
	K_inv_h_0 = K_inv* h_0;

	std::vector<std::vector<float>> s_denom;
	camCalib.mat_to_vec(K_inv_h_0, s_denom);

	float mag = sqrt((s_denom[0][0] * s_denom[0][0]) + (s_denom[1][0] * s_denom[1][0]) + (s_denom[1][0] * s_denom[1][0]));

	cv::Mat r_0(3, 1, CV_32F);
	cv::Mat r_1(3, 1, CV_32F);
	cv::Mat r_2(3, 1, CV_32F);
	cv::Mat t(3, 1, CV_32F);

	float s = 1. / mag;

	r_0 = K_inv * h_0;
	r_1 = K_inv * h_1;
	t = K_inv * h_2;

	multiply(r_0, Scalar(s), r_0);
	multiply(r_1, Scalar(s), r_1);
	multiply(t, Scalar(s), t);

	r_2 = r_0.cross(r_1);

	cv::Mat W(0, 4, CV_32F);

	W.push_back(r_0.t());
	W.push_back(r_1.t());
	W.push_back(r_2.t());

	// Converting 3x3 R matrix to 3x1 Rodrigues vector
	cv::Mat R(3, 3, CV_32F);
	R = W.t();

	cv::Mat p_opencv(3, 1, CV_32F);
	cv::Rodrigues(R, p_opencv);

	// Finally we're adding translation parameteres
	W.push_back(t.t());

	W = W.t();

	W.copyTo(W_out);
}

Point2d CameraCalibration::EstRadialDisplacement(cv::Mat& K, std::vector<cv::Mat>& W, std::vector<std::vector<Point3f>>& X, std::vector<std::vector<Point2f>>& U)
{
	std::vector<std::vector<float>> A;
	camCalib.mat_to_vec(K, A);

	K.convertTo(K, CV_64F);

	// A - est intrinsic camera params
	// W - est extrinsic params, camera views
	// X - model points
	// U - observed sensor points

	cv::Mat d_r (0,1,CV_64F);
	std::vector<std::vector<double>> D_vec;

	// go through each image
	for (unsigned int i = 0; i < W.size(); i++)
	{
		for (unsigned int j = 0; j < X[i].size(); j++)
		{
			// normalized projection x = W * hom(X)
			cv::Mat x_hom(3,1,CV_64F);
			cv::Point2d x, d;
			cv::Mat u(2, 1, CV_64F);
			double r;

			// center of image plane coords
			float u_c = A[0][2];
			float v_c = A[1][2];

			double X_j_data[4][1]{ (double)X[i][j].x, (double)X[i][j].y, (double)X[i][j].z, (double)1 };


			cv::Mat X_j(4, 1, CV_64F, X_j_data);

			W[i].convertTo(W[i], CV_64F);

			x_hom = W[i] * X_j;
			x = { x_hom.at<double>(0,0) / x_hom.at<double>(2,0), x_hom.at<double>(1,0) / x_hom.at<double>(2,0) };
			r = sqrt( (x.x * x.x) + (x.y * x.y) );

			double norm_x[3] = { x.x, x.y, 1 };
			cv::Mat mat_norm_x(3, 1, CV_64F, norm_x);

			cv::Rect rect(0, 0, 3, 3);

			u = K(rect) * mat_norm_x;
			// calculate distance of sensor pixel from projection centre
			// which is not u_c and v_c but 0,0
			d.x = u.at<double>(0, 0);
			d.y = u.at<double>(1, 0);

			// difference between sensor observations and predictions
			double d_r_data_u[1]{ (double)U[i][j].x - u.at<double>(0, 0) };
			double d_r_data_v[1]{ (double)U[i][j].y - u.at<double>(1, 0) };


			cv::Mat d_r_u(1, 1, CV_64F, d_r_data_u);
			cv::Mat d_r_v(1, 1, CV_64F, d_r_data_v);

			double r_2 = r * r;
			double r_4 = r_2 * r * r;

			std::vector<double> D_j{ d.x * r_2, d.x * r_4 };
			std::vector<double> D_j_1{ d.y * r_2, d.y * r_4 };

			// update arrays
			D_vec.push_back(D_j);
			D_vec.push_back(D_j_1);

			d_r.push_back(d_r_u);
			d_r.push_back(d_r_v);
		}
	}

	cv::Mat D(0, D_vec[0].size(), CV_64F);
	camCalib.vector_to_mat_double(D_vec, D);

	cv::Mat D_inv(2,2,CV_64F);
	D_inv = D.inv(DECOMP_SVD);

	cv::Mat k(2, 1, CV_64F);

	// D * k = d -> k = D^-1 * d
	k = D_inv * d_r;

	cv::Mat check;

	check = D * k;

	cv::Point2d k_out{ k.at<double>(0,0),k.at<double>(1,0) };
	return k_out;
}

void UnpackParams(cv::Mat P, cv::Mat& K_in, std::vector<cv::Mat>& W_arr, cv::Point2d& k_in)
{
	// loop through P vector

	// intrinsic matrix
	double K_data[3][3]{
		{ P.at<double>(0,0), P.at<double>(0,2), P.at<double>(0,3) },
		{ (double)0 ,        P.at<double>(0,1),   P.at<double>(0,4) },
		{  (double)0,           (double)0,           (double)1}
	};

	// radial displacement coefficients
	double k_data[2]{ P.at<double>(0,5), P.at<double>(0,6) };

	// traverse all views
	for (int i = 0; i < (P.rows - 7) / 6; i++)
	{
		for (int j = 0; j < 6; j++)
		{
			int view_i = j + 6 * i + 7;

			// Rodrigues rotation to Matrix rotation
			double r_data[3]{ P.at<double>(view_i, 0), P.at<double>(view_i + 1, 0), P.at<double>(view_i + 2, 0) };
			cv::Mat r(1, 3, CV_64F, r_data);
			cv::Mat R(3, 3, CV_64F);

			cv::Rodrigues(r, R);

			double t_data[3]{
				P.at<double>(view_i + 3, 0), P.at<double>(view_i + 4, 0), P.at<double>(view_i + 5, 0)
			};
			cv::Mat T(3, 1, CV_64F, t_data);

			// put it all into Rt matrix also known as Intrinsic Parameters Matrix
			cv::Mat Rt(3, 4, CV_64F);
			cv::hconcat(R, T, Rt);

			// update global H_arr object
			W_arr[i] = Rt;
		}
	}
	// insert data inside opencv Matrix format
	cv::Mat K(3, 3, CV_64F, K_data);
	cv::Point2d k;


	K.copyTo(K_in);
	k_in.x = k_data[0];
	k_in.y = k_data[1];
}


void CameraCalibration::RefineParams(cv::Mat& A, cv::Point2d& k, std::vector<cv::Mat>& W_arr, std::vector<std::vector<Point3f>>& worldPoints, std::vector<std::vector<Point2f>>& imagePoints)
{
	/*cv::Mat P( 1, 7+( worldPoints.size() * 6 ), CV_32F );*/
	cv::Mat P(1, 7, CV_64F);

	std::vector<float> X_data;
	std::vector<float> U_data;

	camCalib.ComposeParamVec(A, k, W_arr, P);

	for (unsigned int i = 0; i < worldPoints.size(); i++)
	{
		for (unsigned int j = 0; j < worldPoints[i].size(); j++)
		{
			// its assumed that number of world points and image points is the same for every view
			X_data.push_back(worldPoints[i][j].x);
			X_data.push_back(worldPoints[i][j].y);

			U_data.push_back(imagePoints[i][j].x);
			U_data.push_back(imagePoints[i][j].y);
		}
	}

	// init X and U matrices
	cv::Mat X(1, X_data.size(), CV_32F, X_data.data());
	cv::Mat U(1, U_data.size(), CV_32F, U_data.data());

	// Optimize here

	const int N_VIEWS = worldPoints.size();
	const int N_POINTS = worldPoints[0].size();
	const int N_PARAMS = 7 + 6 * N_VIEWS;

	auto criteria = cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 80, DBL_EPSILON);
	auto solver = CvLevMarq(N_PARAMS, U.cols, criteria, false);
	
	int iter = 0;

	// convert from cv::Mat to cvMat, ughh
	std::vector<double> P_data;
	if (P.isContinuous())
	{
		P_data.assign((double*)P.data, (double*)P.data + P.total() * P.channels());
	}

	// convert to double

	std::vector<double> d_P_data;

	for (int i = 0; i < P_data.size(); i++)
	{
		d_P_data.push_back((double)P_data[i]);
	}


	CvMat param = cvMat(N_PARAMS, 1, CV_64F, d_P_data.data() );

	cvCopy(&param, solver.param);

	// inint J matrix
	cv::Mat J(2 * N_VIEWS * N_POINTS, N_PARAMS, CV_64F);
	cv::Mat err(1, 2 * N_VIEWS * N_POINTS, CV_64F);

	std::cout << "FindParams:: LevMarq non-linear optimization has started" << std::endl;

	while (true)
	{
		const CvMat* _param = 0;
		CvMat* _jac = 0;
		CvMat* _err = 0;

		bool proceed = solver.update(_param, _jac, _err);

		cvCopy(_param, &param);
		std::cout << "--------------------------------------------------------" << std::endl;

		std::cout << "iter=" << iter << " state=" << solver.state
			<< " errNorm= " << solver.errNorm << std::endl;

		if (!proceed || !_err) break;

		if (_jac)
		{
			Mat p = Mat(param.rows, param.cols, CV_64F, param.data.db);
			jacobian_fnc(J, X, p);

			std::vector<double> tmp;
			if (J.isContinuous())

			{
				tmp.assign((double*)J.data, (double*)J.data + J.total() * J.channels());
			}

			// traverse array and convert 1xN matrix to -> 1056x73

			CvMat jac_tmp = cvMat(J.rows, J.cols, CV_64F);
			cvCreateData(&jac_tmp);


			for (int i = 0 ; i < N_POINTS*2*N_VIEWS ; i++)
			{
				for (int j = 0; j < N_VIEWS * 6 + 7; j++)
				{
					double fill = (double)J.at<double>(i, j);

					CV_MAT_ELEM(jac_tmp, double, i, j) = fill;
				}
			}
			
			cvCopy(&jac_tmp, _jac);
		}
		if (_err)
		{
			Mat p = Mat(param.rows, param.cols, CV_64F, param.data.db);
			error_fnc(err, X, U, p);
			iter++;

			std::vector<double> tmp;
			if (err.isContinuous())
			{
				tmp.assign((double*)err.data, (double*)err.data + err.total() * err.channels());
			}

			// convert to double

			double sum = 0.;

			std::vector<double> tmp_d;
			for (int i = 0; i < tmp.size(); i++)
			{
				tmp_d.push_back((double)tmp[i]);
				sum += tmp[i] * tmp[i];
			}

			CvMat err_tmp = cvMat(err.cols, 1 , CV_64F, tmp_d.data());
			cvCopy(&err_tmp, _err);
		}
	}

	UnpackParams(P, A, W_arr, k);
}

//Eigen fcking around

struct LMFunctor
{
	// 'm' pairs of (x, f(x))
	cv::Mat* X = 0;
	cv::Mat* U = 0;

	Eigen::Map< Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >* temp_jvec_type;

	// Compute 'm' errors, one for each data point, for the given parameter values in 'x'
	int operator()(const Eigen::VectorXd& x, Eigen::VectorXd& fvec) const
	{
		// 'x' has dimensions n x 1
		// It contains the current estimates for the parameters.

		// 'fvec' has dimensions m x 1
		// It will contain the error for each data point.

		// you need to provide the error vector to the optimizer here by assigning fvec

		// convert from cv mat to eigen lib format

		std::vector<double> p_vec(x.data(), x.data() + x.rows() * x.cols());
		cv::Mat p_mat(p_vec.size(), 1, CV_64F, p_vec.data());

		// std::vector<double> e_vec(fvec.data(), fvec.data() + fvec.rows() * fvec.cols());

		cv::Mat e_mat( 1 , m, CV_64F );

		error_fnc(e_mat, *X, *U, p_mat);

		std::vector<std::vector<double>> e_vec_temp;
		camCalib.mat_to_vec(e_mat, e_vec_temp);


		Eigen::VectorXd temp_fvec = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(e_vec_temp[0].data(), m);
		fvec = temp_fvec;

		return 0;
	}

	// Compute the jacobian of the errors
	int df(const Eigen::VectorXd& x, Eigen::MatrixXd& fjac) const
	{
		// 'x' has dimensions n x 1
		// It contains the current estimates for the parameters.

		// 'fjac' has dimensions m x n
		// It will contain the jacobian of the errors, calculated numerically in this case.

		float epsilon;
		epsilon = 1e-5f;

		std::vector<double> p_vec(x.data(), x.data() + x.rows() * x.cols());
		cv::Mat p_mat(p_vec.size(), 1, CV_64F, p_vec.data());

		cv::Mat J(m, n, CV_64F);

		jacobian_fnc(J, *X, p_mat);


		std::vector<std::vector<double>> j_vec_temp;
		camCalib.mat_to_vec(J, j_vec_temp);

		Eigen::MatrixXd jac(m, n);

		for (int i = 0; i < m; i++)
		{
			Eigen::VectorXd row_i = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(j_vec_temp[i].data(), 73);
			jac.row(i) = row_i;
		}

		// Eigen::MatrixXd jvec_mat = temp_jvec_type(j_vec_temp.data());

		fjac = jac;

		return 0;
	}

	// Number of data points, i.e. values.
	int m;

	// Returns 'm', the number of values.
	int values() const { return m; }

	// The number of parameters, i.e. inputs.
	int n;

	// Returns 'n', the number of inputs.
	int inputs() const { return n; }

};


void CameraCalibration::EIGEN_RefineParams(cv::Mat& A, cv::Point2d& k, std::vector<cv::Mat>& W_arr, std::vector<std::vector<cv::Point3f>>& worldPoints, std::vector<std::vector<cv::Point2f>>& imagePoints)
{
	/*cv::Mat P( 1, 7+( worldPoints.size() * 6 ), CV_32F );*/
	cv::Mat P(1, 7, CV_64F);

	std::vector<float> X_data;
	std::vector<float> U_data;

	camCalib.ComposeParamVec(A, k, W_arr, P);

	for (unsigned int i = 0; i < worldPoints.size(); i++)
	{
		for (unsigned int j = 0; j < worldPoints[i].size(); j++)
		{
			// its assumed that number of world points and image points is the same for every view
			X_data.push_back(worldPoints[i][j].x);
			X_data.push_back(worldPoints[i][j].y);

			U_data.push_back(imagePoints[i][j].x);
			U_data.push_back(imagePoints[i][j].y);
		}
	}

	// init X and U matrices
	cv::Mat X(1, X_data.size(), CV_32F, X_data.data());
	cv::Mat U(1, U_data.size(), CV_32F, U_data.data());

	// Optimize here

	const int N_VIEWS = worldPoints.size();
	const int N_POINTS = worldPoints[0].size();
	const int N_PARAMS = 7 + 6 * N_VIEWS;

	auto criteria = cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 80, DBL_EPSILON);
	auto solver = CvLevMarq(N_PARAMS, U.cols, criteria, false);

	int iter = 0;

	// convert from cv::Mat to cvMat, ughh
	std::vector<double> P_data;
	if (P.isContinuous())
	{
		P_data.assign((double*)P.data, (double*)P.data + P.total() * P.channels());
	}

	// convert to double

	std::vector<double> d_P_data;

	for (int i = 0; i < P_data.size(); i++)
	{
		d_P_data.push_back((double)P_data[i]);
	}


	CvMat param = cvMat(N_PARAMS, 1, CV_64F, d_P_data.data());

	cvCopy(&param, solver.param);

	// inint J matrix
	cv::Mat J(2 * N_VIEWS * N_POINTS, N_PARAMS, CV_64F);
	cv::Mat err(1, 2 * N_VIEWS * N_POINTS, CV_64F);

	std::cout << "FindParams:: Eigen non-linear optimization has started" << std::endl;

	int m = 2 * N_VIEWS * N_POINTS;
	int n = N_PARAMS;

	Eigen::VectorXd init_x = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(d_P_data.data(), 73);

	LMFunctor functor;
	functor.X = &X;
	functor.U = &U;
	functor.m = m;
	functor.n = n;

	Eigen::LevenbergMarquardt<LMFunctor, double> lm(functor);

	lm.parameters.maxfev = 2000;
	lm.parameters.xtol = 1.0e-2;


	int status = lm.minimize(init_x);
	std::cout << "LM optimization status: " << status << std::endl;
	std::cout << lm.iter << std::endl;
	std::cout << lm.fnorm << std::endl;


	/*while (true)
	{

	}*/

	UnpackParams(P, A, W_arr, k);
}