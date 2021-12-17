#pragma once
#pragma once
#include <vector>
#include <opencv2/core/types.hpp>
#include <math.h>
#include <stdexcept>
#include <fstream>

class imagePointsData
{
private:
	std::vector<cv::Point2f> i_points_arr;
public:
	friend std::ifstream& operator>> (std::ifstream& input, imagePointsData& m)
	{
		if (!input)
		{
			throw std::invalid_argument("Nothing to read");
		}
		else
		{
			char char1, char2, char3;

			cv::Point2f p;

			input >> char1 >> p.x >> char2 >> p.y >> char3;


			if (char1 == '[' && char2 == ',' && char3 == ']')
			{
				m.i_points_arr.push_back(p);
				return input;
			}
			else
				throw std::invalid_argument("bad_input_Point");
		}


	}
	void getData(std::vector<cv::Point2f>& outRef)
	{
		if (i_points_arr.empty()) throw std::invalid_argument("No data available to reference");
		outRef = i_points_arr;
	}
};