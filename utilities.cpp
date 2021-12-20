#include <opencv2/opencv.hpp>
#include "utilities.h"

using namespace cv;

// input X - world points
// P = vector of intrinsic and all view extrinsic params
// the function projects all input world points to sensor coordinates
//void value_fnc(int POINTS_COUNT, std::vector<std::vector<cv::Point3f>> X, cv::Mat& P)
//{
//	float a_data[7] = { P.at<float>(0,0), P.at<float>(0,1), P.at<float>(0,2), P.at<float>(0,3), P.at<float>(0,4), P.at<float>(0,5), P.at<float>(0,6) };
//	cv::Mat a(0, 7, CV_32F, a_data);
//
//	std::vector<cv::Point2f> samples;
//
//	for (unsigned int i = 0; i < X.size(); i++)
//	{
//		// first 7 entries of P are fixed intrinsic params
//		int m = 7 + 6 * i;
//
//		cv::Mat w(0, 6, CV_32F);
//		cv::Rect range(m, 0, m + 6, 0);
//		P(range).copyTo(w);
//
//		for (unsigned int j = 0; j < POINTS_COUNT; j++)
//		{
//			cv::Point2f y = project_point(a, w, X[i][j]);
//
//			samples.push_back(y);
//		}
//	}
//}

void error_fnc(Mat& err, cv::Mat& X , cv::Mat& U, cv::Mat& P)
{
	int N_POINTS = 48;
	int N_VIEWS = 11;
	// intrinsic camera data decomposition
	double a_data[7] = { P.at<double>(0,0), P.at<double>(1,0), P.at<double>(2,0), P.at<double>(3,0), P.at<double>(4,0), P.at<double>(5,0), P.at<double>(6,0) };
	cv::Mat a(1, 7, CV_64F, a_data);

	int p_indx = 0;
	// for every 2*N_POINTS*N_VIEWS
	for (int i = 0; i < N_VIEWS; i++)
	{
			for (int j = 0; j < N_POINTS; j++)
			{
				// first 7 entries of P are fixed intrinsic params
				int m = 7 + 6 * i;
				// ideal projection
				cv::Mat w(6, 1, CV_64F);
				cv::Rect range(0, m, 1 , 6);
				P(range).copyTo(w);

				int indx = (i * N_POINTS * 2) + j * 2;

				float world_x = X.at<float>(0, indx);
				float world_y = X.at<float>(0, indx + 1);

				cv::Point3d p = { (double)world_x, (double)world_y, 0 };

				cv::Mat a_t = a.t();
				cv::Mat w_t = w.t();

				cv::Point2d sample = project_point(a, w_t, p);
				project_point(a, w_t, p);

				float observed_x = U.at<float>(0, indx);
				float observed_y = U.at<float>(0, indx+1);

				// calculate and add err to the vector
				err.at<double>(0,p_indx) = pow((double)(observed_x - sample.x),2);
				err.at<double>(0, p_indx+1) = pow((double)(observed_y - sample.y),2);

				p_indx = p_indx + 2;
			}
	}
}

void jacobian_fnc(cv::Mat J, cv::Mat& X, cv::Mat& P)
{

	const int N_VIEWS = 11;
	const int N_MODEL_PNTS = 48;
	const int N_PARAMS = 7 + 6 * N_VIEWS;

	// first 7 params are intrinsic values
	double a_data[7]{ P.at<double>(0,0), P.at<double>(1,0), P.at<double>(2,0), P.at<double>(3,0), P.at<double>(4,0), P.at<double>(5,0), P.at<double>(6,0) };

	// params all extracted to single vector
	// that's why we're using many loops

	// for 11 views there's 73 params in total
	// first 7-intrinsic, rest-view params for each view
	for (unsigned int k = 0; k < N_PARAMS; k++)
	{
		// current p
		double p = P.at<double>(k, 0);
		const double e = 2.2 * pow(10, -5.0);
		int r = 0;

		double c = sqrt(e) * max(p, (double)1);

		for (unsigned int v = 0; v < N_VIEWS; v++)
		{
			// starting index for another view params
			int w = 7 + 6 * v;

			// fill in arr with 3x rotation params and 3x translation params for the current view
			double w_data[6]{ P.at<double>(w, 0), P.at<double>(w+1, 0),P.at<double>(w+2, 0),P.at<double>(w+3, 0),P.at<double>(w+4, 0), P.at<double>(w+5, 0) };

			cv::Mat w_mat(1, 6, CV_64F, w_data);

				for (unsigned int j = 0; j < N_MODEL_PNTS; j++)
				{
					std::vector<double> J_v_1;
					std::vector<double> J_v_2;

					// j   -> x component
					// j+1 -> y component

					// X_j contains only ints so loseless conversion from float to double
					cv::Point3d X_j = { (double)X.at<float>(0, (v * N_MODEL_PNTS * 2) + j), (double)X.at<float>(0, (v * N_MODEL_PNTS * 2) + j+1), 1 };

					// edit value at P[0][k] -> P[0][k] + c

					// and then P_x(a,w,Xj) - P_x(a_new, w_new, Xj) / c

					cv::Mat a_mat(1, 7, CV_64F, a_data);

					// we need to add some small value to current param
					// figure out which param we are iterating -> k

					int const arr_length = 7;
					double arr[arr_length];

					cv::Point2d P_x_;

					// need to find which param to modify
					// either first part - intrinsic params or second- extrinsic params
					if (k <= 6)
					{
						for (unsigned int i = 0; i < arr_length; i++)
						{
							
							if (k == i) {
								double curr_val = a_data[i];
								double curr_val_c = a_data[i] + c;
								arr[i] = curr_val_c;
							} else { arr[i] = a_data[i]; }
						}
						cv::Mat a_mat_c(1, 7, CV_64F, arr);
						P_x_ = project_point(a_mat_c, w_mat, X_j);
					}
					else
					{
						// find curr view 
						int o = std::floor( (k - 7) / 6);

						// find curr view index
						int w_indx = (k - 7) % 6;


						for (unsigned int i = 0; i < arr_length-1; i++)
						{
							// if curr view and curr view index match curr itereation
							if (o == v && w_indx == i) {
								double curr_val = w_data[i];
								double curr_val_c = w_data[i] + c;

								arr[i] = curr_val_c;
							}
							else { arr[i] = w_data[i]; }
						}
						cv::Mat w_mat_c(1, 6, CV_64F, arr);
						P_x_ = project_point(a_mat, w_mat_c, X_j);
					}
					

					cv::Point2d P_x;
					P_x = project_point(a_mat, w_mat, X_j);

					double der_1 = (P_x_.x - P_x.x) / c;
					double der_2 = (P_x_.y - P_x.y) / c;

					J.at<double>(r, k) = der_1;
					J.at<double>(r+1, k) = der_2;
					r = r + 2;
				}
		}
	}
}

void distort(cv::Mat& k, cv::Mat& hom_point)
{
	double x = hom_point.at<double>(0,0);
	double y = hom_point.at<double>(1, 0);

	double r = sqrt(x * x + y * y);

	double distortion = (k.at<double>(0, 0) * r * r) + (k.at<double>(1, 0) * r * r * r * r);

	double x_distorted_data[3]{ x * ((double)1 + distortion), y * ((double)1 + distortion), 1 };

	cv::Mat x_distorted(3, 1, CV_64F, x_distorted_data);

	x_distorted.copyTo(hom_point);
}

cv::Point2d project_point(cv::Mat& a, cv::Mat& w, cv::Point3d world_point)
{
	// DON'T CONVERT WE NEED HIGH PRECISION

	// convert world_point from inhomogenous/cartesian coords to homogenous 4x1 vector
	double X_data[4]{ world_point.x, world_point.y, world_point.z, (double)1 };

	// init matrices
	cv::Mat K(3, 3, CV_64F);
	cv::Mat k(2, 1, CV_64F);
	cv::Mat Rt(3, 4, CV_64F);
	cv::Mat x(2, 1, CV_64F);
	cv::Mat X(4, 1, CV_64F, X_data);

	cv::Point2d x_out;

	unpack_params(a, w, K, Rt, k);

	cv::Mat x_ext(3, 1, CV_64F);
	cv::Mat x_int(2, 1, CV_64F);

	// 3x4 x 4x1 -> 3x1
	x_ext = Rt * X;

	x_ext = x_ext / x_ext.at<double>(2, 0);
	// apply known distortion params
	distort(k, x_ext);

	// 2x3 x 3x1 -> 2x1
	x_int = K(cv::Rect(0, 0, 3, 2)) * x_ext;

	x_out = { x_int.at<double>(0, 0), x_int.at<double>(1, 0) };
	return x_out;
}

void unpack_params(cv::Mat& a, cv::Mat& w, cv::Mat& K_out, cv::Mat& Rt_out, cv::Mat& k_out)
{
	// intrinsic matrix
	double K_data[3][3]{
		{ a.at<double>(0,0), a.at<double>(0,2), a.at<double>(0,3) },
		{ (double)0 ,        a.at<double>(0,1),   a.at<double>(0,4) },
		{  (double)0,           (double)0,           (double)1}
	};

	// radial displacement coefficients
	double k_data[2]{ a.at<double>(0,5), a.at<double>(0,6) };
	double r_data[3]{ w.at<double>(0,0), w.at<double>(0,1), w.at<double>(0,2) };
	cv::Mat r(1, 3, CV_64F, r_data);
	cv::Mat R(3,3,CV_64F);

	cv::Rodrigues(r, R);

	double t_data[3]{
		w.at<double>(0,3),w.at<double>(0,4),w.at<double>(0,5)
	};

	
	// insert data inside opencv Matrix format
	cv::Mat K(3, 3, CV_64F, K_data);
	cv::Mat k(2, 1, CV_64F, k_data);
	cv::Mat T(3, 1, CV_64F, t_data);
	cv::Mat Rt(3, 4, CV_64F);

	cv::hconcat(R, T, Rt);

	K.copyTo(K_out);
	k.copyTo(k_out);
	Rt.copyTo(Rt_out);

}
