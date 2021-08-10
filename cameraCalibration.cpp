#include "CameraCalibration.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include <vector>
#include <iostream>
#include <assert.h>
#include <math.h>

using namespace MY;
using namespace cv;

CameraCalibration camCalib;

void CameraCalibration::FindPairs(int C_WIDTH, int C_HEIGHT, std::vector<std::vector<Point3f>>& wPoints, std::vector<std::vector<Point2f>>& iPoints)
{
	int CHECKERBOARD[2]{ 6,8 };

	std::cout << "Hello World!\n";

	Mat checkboard_1 = imread("./data/img_1.jpg");
	namedWindow("checkboard");

	// now finally calibrate camera

	std::vector<Point3f> initWorldPoints;
	std::vector<std::vector<Point3f>> worldPoints;
	std::vector<std::vector<Point2f>> imgPoints;


	std::vector<String> imgs;

	std::string path = "./data/*.jpg";

	// load all string dirs into specified vector list
	glob(path, imgs);

	// define world coords for 3d points

	for (int i = 0; i < CHECKERBOARD[0]; i++)
	{
		for (int j = 0; j < CHECKERBOARD[1]; j++)
		{
			initWorldPoints.push_back(Point3f(j, i, 1));
			std::cout << Point3f(j, i, 1) << std::endl;
		}
	}

	// loop over all imgs
	std::vector<Point2f> corners;
	Size p_size(6, 8);
	Mat frame, gray, resized;
	bool success;


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
		}

		imshow("checkboard", gray);
		waitKey(10);
	}
}
void CameraCalibration::Build_L(std::vector<cv::Point2f> m, std::vector<cv::Point3f> M, std::vector<std::vector<float>>& L)
{
	// # of image pairs = m.size()/M.size()
	for (unsigned int i = 0; i < M.size(); i++)
	{
		std::vector<float> L_row_1{ -M[i].x ,-M[i].y, -M[i].z, 0, 0, 0, (m[i].x * M[i].x), (m[i].x * M[i].y),m[i].x };
		std::vector<float> L_row_2{ 0,  0,   0, -M[i].x, -M[i].y, -M[i].z, ( m[i].y * M[i].x), (m[i].y * M[i].y),m[i].y };

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

// not working properly currently
cv::Point3f CameraCalibration::ToRodriguezVec(cv::Mat& R)
{
	// define p and c
	cv::Point3f p{  (R.at<float>(2,1) - R.at<float>(1,2)) / (float)2,
					(R.at<float>(0,2) - R.at<float>(2,0)) / (float)2,
					(R.at<float>(1,0) - R.at<float>(0,1)) / (float)2, };

	float c = R.at<float>(0, 0) + R.at<float>(1, 1) + R.at<float>(2, 2);
	c = (c - (float)1) / (float)2;

	float p_mag = sqrt( (p.x * p.x) + (p.y * p.y) + (p.z * p.z));
	cv::Mat u(3, 1, CV_32F);

	if (p_mag == (float)0)
	{
		if (c == (float)1)
		{
			p = { 0,0,0 };
		}
		else if (c == -1)
		{
			cv::Mat R_a(3, 3, CV_32F);
			float I_data[3][3] = {
				1.,0,0,
				0,1.,0,
				0,0,1.
			};
			cv::Mat I(3, 3, CV_32F, I_data);

			R_a = R + I;

			cv::Mat R_a_t = R_a.t();

			// Find columnn vector of R_a with max norm
			float max_norm = -INFINITY;
			int max_norm_indx = -1;

			cv::Mat v(1, 3, CV_32F);

			for (unsigned int i = 0; i < R_a_t.rows; i++)
			{
				float* row = R_a_t.ptr<float>(i);

				for (unsigned int j = 0; j < R_a_t.cols; j++)
				{
					if (row[j] > max_norm) max_norm_indx = i;
				}
			}

			R_a_t.row(max_norm_indx).copyTo(v.row(0));
			v.t();
			float v_mag = sqrt((v.at<float>(0, 0) * v.at<float>(0, 0)) + (v.at<float>(1, 0) * v.at<float>(1, 0)) + (v.at<float>(2, 0) * v.at<float>(2, 0)));

			u = v / v_mag;

			bool cond1 = u.at<float>(0, 0) < (float)0;
			bool cond2 = u.at<float>(0, 0) == (float)0 && u.at<float>(1, 0) < (float)0;
			bool cond3 = u.at<float>(0, 0) == u.at<float>(1, 0) == (float)0;
			bool cond4 = u.at<float>(2, 0) < (float)0;


			if (u.at<float>(0, 0) < (float)0 || (u.at<float>(0, 0) == (float)0 && u.at<float>(1, 0) < (float)0) || (u.at<float>(0, 0) == u.at<float>(1, 0) == (float)0 || u.at<float>(2, 0) < 0.))
			{
				u = -u;
			}

			if (cond1 || cond2 || cond3 || cond4)
			{
				std::cout << "TRUE" << std::endl;
			}

			p.x = p_mag * u.at<float>(0, 0);
			p.y = p_mag * u.at<float>(1, 0);
			p.z = p_mag * u.at<float>(2, 0);
		}
		else { p.x = std::numeric_limits<float>::quiet_NaN(); p.y = std::numeric_limits<float>::quiet_NaN(); p.z = std::numeric_limits<float>::quiet_NaN(); }
	}
	else
	{

		float u_data[3][1]{
			p.x / p_mag,
			p.y / p_mag,
			p.z / p_mag
		};

		cv::Mat u_case_3(3, 1, CV_32F, u_data);
		u = u_case_3;

		float theta = atan2(p_mag, c);

		p.x = theta * u.at<float>(0, 0);
		p.y = theta * u.at<float>(1, 0);
		p.z = theta * u.at<float>(2, 0);
	}

	return p;
}

void CameraCalibration::ComposeParamVec(cv::Mat& A, cv::Point2f& k, std::vector<cv::Mat>& W_arr, cv::Mat& P)
{
	float a[7] { A.at<float>(0,0), A.at<float>(1,1), A.at<float>(0,1), A.at<float>(0,2), A.at<float>(1,2), k.x, k.y };

	// init Z with intrinsic values
	cv::Mat Z(1, 7, CV_32F, a);

	// copy Z to P mat accessed by a pointer
	Z.copyTo(P);

	for (unsigned int i = 0; i < W_arr.size(); i++)
	{
		cv::Mat R(3, 3, CV_32F);
		cv::Rect R_roi(0, 0, 3, 3);
		W_arr[i](R_roi).copyTo(R);

		cv::Mat p(3, 1, CV_32F);

		// rotation matrix to rodrigues vector
		cv::Rodrigues(R, p);
		
		// push back rodruges vector and transpose vector from the current view matrix
		P.push_back(p);
		P.push_back(W_arr[i].col(4).t());
	}
}

std::vector<float> Build_v(std::vector<std::vector<float>>& H, int i, int j)
{
	std::vector<float> v = {
		H[i][0] * H[j][0], (H[i][0] * H[j][1]) + (H[i][1] * H[j][0]), H[i][1] * H[j][1],
		(H[i][2] * H[j][0]) + (H[i][0] * H[j][2]), (H[i][2] * H[j][1] + H[i][1] * H[j][2]), H[i][2] * H[j][2]
	};

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
	// x = [h1^T, h2^T, h3^T] <-- we are looking for these values
	// we are going to find it by finding eigenvectors of L^T*L
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

		cv::Mat w_val_mat = N_mat_world*cv::Mat(wPoints[w], false);
		cv::Point3f p_w(w_val_mat);

		std::cout << N_mat_world << std::endl;
		std::cout << wPoints[w] << std::endl;

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

	std::cout << "CameraCalibration::FindHomography:: L_Mat: " << L_mat << std::endl;

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

	std::cout << "Eigenvectors: " << std::endl;
	std::cout << eigenvectors << std::endl;
	std::cout << "Eigenvalues: " << std::endl;
	std::cout << eigenvalues << std::endl;
	std::cout << "H vector: " << std::endl;
	std::cout << s_eigenv << std::endl;

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
	std::cout << "H Normalized matrix: " << std::endl;
	std::cout << H_norm << std::endl;
	cv::Mat H_denorm_1 = H_norm * N_mat_world;
	std::cout << "Inverse of normalize image matrix: " << std::endl;
	std::cout << N_mat_image.inv() << std::endl;
	cv::Mat H_denorm = N_mat_image.inv() * H_norm * N_mat_world;
	std::cout << "Normalize world matrix: " << std::endl;
	std::cout << N_mat_world << std::endl;

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

	std::cout << H_denorm << std::endl;

	H_data = H_denorm_out;

	return true;

}
bool CameraCalibration::vector_to_mat(std::vector<std::vector<float>>& v, cv::Mat& mat)
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

void CameraCalibration::mat_to_vec(cv::Mat& mat, std::vector<std::vector<float>>& v)
{
	assert(v.empty());

	if (mat.type() != (int)5) mat.convertTo(mat, CV_32F);

	for (unsigned int m = 0; m < mat.rows; m++)
	{
		float* mat_row = mat.ptr<float>(m);

		std::vector<float> empty_row;
		v.push_back(empty_row);

		for (unsigned int n = 0; n < mat.cols; n++)
		{
			v[v.size() - 1].push_back(mat_row[n]);
		}
	}
}

void CameraCalibration::ExtractIntrinsicParams(std::vector<float>& b, cv::Mat& K_out)
{
	// b = [ B_11, B_12, B_22, B_13, B_23, B_33]^T

	double B_11 = b[0];
	double B_12 = b[1];
	double B_22 = b[2];
	double B_13 = b[3];
	double B_23 = b[4];
	double B_33 = b[5];

	double v_0 = (B_12 * B_13 - B_11 * B_23) / (B_11 * B_22 - B_12 * B_12);
	double s = B_33 - ( B_13 * B_13 + v_0*(B_12 * B_13) - (B_11 * B_23) / B_11);
	double a = sqrt( s/ B_11 );
	double beta = sqrt( (s * B_11) / (B_11 * B_22 - (B_12 * B_12)));
	double y = -B_12 * a * a * beta / s;
	double u_0 = ((y * v_0) / beta) - (B_13 * a * a / s);

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

	std::cout << V_mat << std::endl;

	// find eigenvalue of V(T)V 1x6 matrix

	// s_eigenv: eigenvectors corresponding to the smallest found eigenvalue
	Mat VT_V = V_mat.t() * V_mat;
	Mat eigenvectors(6, 6, CV_32F);
	Mat eigenvalues(6, 1, CV_32F);
	Mat s_eigenv(1, 6, CV_32F);

	eigen(VT_V, eigenvalues, eigenvectors);

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

	std::cout << "FindCameraIntrinsics::Eigenvectors: " << std::endl;
	std::cout << eigenvectors << std::endl;
	std::cout << "FindCameraIntrinsics::Eigenvalues: " << std::endl;
	std::cout << eigenvalues << std::endl;
	std::cout << "FindCameraIntrinsics::H vector: " << std::endl;
	std::cout << s_eigenv << std::endl;


	std::cout << "FindCameraIntrinsics::V^T*V: " << std::endl;
	std::cout << VT_V << std::endl;

	// Finally, find K
	cv::Mat K(3, 3, CV_32F);
	std::vector<float> s_eigenv_data;
	s_eigenv_data.assign((float*)s_eigenv.datastart, (float*)s_eigenv.dataend);

	camCalib.ExtractIntrinsicParams(s_eigenv_data, K);

	std::cout << "FindCameraIntrinsics:: K: " << std::endl;
	std::cout << K << std::endl;

	K.copyTo(K_out);
}

void CameraCalibration::ExtractViewParams(cv::Mat& K, std::vector<std::vector<float>>& H, cv::Mat& W_out)
{
	/*std::vector<float> h_0_data{ H[0][0], H[1][0], H[2][0] };
	std::vector<float> h_1_data{ H[0][1], H[1][1], H[2][1] };
	std::vector<float> h_2_data{ H[0][2], H[1][2], H[2][2] };*/

	float h_0_data[] { H[0][0], H[1][0], H[2][0] };
	float h_1_data[] { H[0][1], H[1][1], H[2][1] };
	float h_2_data[] { H[0][2], H[1][2], H[2][2] };

	cv::Mat h_0(3, 1, CV_32F, H[0].data());
	cv::Mat h_1(3, 1, CV_32F, H[1].data());
	cv::Mat h_2(3, 1, CV_32F, H[2].data());

	/*h_0.convertTo(h_0, CV_64F);
	h_1.convertTo(h_1, CV_64F);
	h_2.convertTo(h_2, CV_64F);*/

	std::cout << "CameraCalibration::ExtractViewParams:: h_0: " << h_0 << std::endl;

	
	cv::Mat K_inv(3, 3, CV_32F);
	K_inv = K.inv();

	K_inv.convertTo(K_inv, CV_32F);

	std::cout << "CameraCalibration::ExtractViewParams:: K_inv: " << K_inv << std::endl;
	std::cout << "CameraCalibration::ExtractViewParams:: h_0: " << h_0 << std::endl;

	cv::Mat K_inv_h_0(1, 3, CV_32F);
	K_inv_h_0 = K_inv* h_0;

	std::cout << "CameraCalibration::ExtractViewParams:: K_inv_h_0: " << K_inv_h_0 << std::endl;

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

	std::cout << "CameraCalibration::ExtractViewParams:: r_0: " << r_0 << std::endl;
	std::cout << "CameraCalibration::ExtractViewParams:: r_1: " << r_1 << std::endl;
	std::cout << "CameraCalibration::ExtractViewParams:: r_2: " << r_2 << std::endl;
	std::cout << "CameraCalibration::ExtractViewParams:: t: " << t << std::endl;

	W.push_back(r_0.t());
	W.push_back(r_1.t());
	W.push_back(r_2.t());

	cv::Point3f p;
	cv::Mat R(3, 3, CV_32F);
	R = W.t();

	cv::Mat p_opencv(3, 1, CV_32F);
	cv::Rodrigues(R, p_opencv);

	W.push_back(t.t());

	std::cout << "CameraCalibration::ExtractViewParams: p: " << p << std::endl;
	std::cout << "CameraCalibration::ExtractViewParams: p_opencv: " << p_opencv << std::endl;

	W = W.t();

	std::cout << "CameraCalibration::ExtractViewParams:: W: " << W << std::endl;

	W.copyTo(W_out);

	// create vector w/ params like this r_1,r_2, r_3, t


}

Point2f CameraCalibration::EstRadialDisplacement(cv::Mat& K, std::vector<cv::Mat>& W, std::vector<std::vector<Point3f>>& X, std::vector<std::vector<Point2f>>& U)
{
	std::vector<std::vector<float>> A;
	camCalib.mat_to_vec(K, A);

	// A - est intrinsic camera params
	// W - est extrinsic params, camera views
	// X - model points
	// U - observed sensor points

	cv::Mat d_r (0,1,CV_32F);
	std::vector<std::vector<float>> D_vec;

	// go through each image
	for (unsigned int i = 0; i < W.size(); i++)
	{
		for (unsigned int j = 0; j < X[i].size(); j++)
		{
			// normalized projection x = W * hom(X)
			cv::Mat x_hom(3,1,CV_32F);
			cv::Point2f x, d;
			cv::Mat u(2, 1, CV_32F);
			float r;

			// center of image plane coords
			float u_c = A[0][2];
			float v_c = A[1][2];

			float X_j_data[4][1]{ X[i][j].x, X[i][j].y, X[i][j].z, 1 };


			cv::Mat X_j(4, 1, CV_32F, X_j_data);

			x_hom = W[i] * X_j;
			x = { x_hom.at<float>(0,0) / x_hom.at<float>(2,0), x_hom.at<float>(1,0) / x_hom.at<float>(2,0) };
			r = sqrt( (x.x * x.x) + (x.y * x.y) );

			cv::Rect rect(0, 0, 3, 2);

			u = K(rect) * x_hom;
			// calculate distance of sensor pixel from projection centre
			d.x = u.at<float>(0, 0) - u_c;
			d.y = u.at<float>(1, 0) - v_c;

			// difference between sensor observations and predictions
			float d_r_data_u[1]{ U[i][j].x - u.at<float>(0, 0) };
			float d_r_data_v[1]{ U[i][j].x - u.at<float>(1, 0) };

			cv::Mat d_r_u(1, 1, CV_32F, d_r_data_u);
			cv::Mat d_r_v(1, 1, CV_32F, d_r_data_v);

			float r_2 = r * r;
			float r_4 = r_2 * r * r;

			std::vector<float> D_j{ d.x * r_2, d.x * r_4 };
			std::vector<float> D_j_1{ d.y * r_2, d.y * r_4 };

			// update arrays
			D_vec.push_back(D_j);
			D_vec.push_back(D_j_1);

			d_r.push_back(d_r_u);
			d_r.push_back(d_r_v);
		}
	}

	cv::Mat D(0, D_vec[0].size(), CV_32F);
	camCalib.vector_to_mat(D_vec, D);

	cv::Mat D_inv(2,2,CV_32F);
	D_inv = D.inv(DECOMP_SVD);

	cv::Mat k(2, 1, CV_32F);

	// D * k = d -> k = D^-1 * d
	k = D_inv * d_r;

	cv::Point2f k_out{ k.at<float>(0,0),k.at<float>(1,0) };
	return k_out;

	std::cout << "CameraCalibration::ExtractViewParams:: k_1: " << k << std::endl;
}

void CameraCalibration::RefineParams(cv::Mat& A, cv::Point2f& k, std::vector<cv::Mat>& W_arr, std::vector<std::vector<Point3f>>& worldPoints, std::vector<std::vector<Point2f>>& imagePoints)
{
	cv::Mat P;

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

}