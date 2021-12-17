#pragma once
#include <vector>
#include <opencv2/core/types.hpp>
#include <math.h>
#include <stdexcept>
#include <fstream>

class worldPointsData
{
private:
	std::vector<cv::Point3f> w_points_arr;
public:
	friend std::ifstream& operator>> (std::ifstream& input, worldPointsData& m)
	{
		if (!input) 
		{
			throw std::invalid_argument("Nothing to read");
		}
		else
		{
			char char1, char2, char3, char4;

			cv::Point3f p;

			input >> char1 >> p.x >> char2 >> p.y >> char3 >> p.z >> char4;
			

			if (char1 == '[' && char2 == ',' && char3 == ',' && char4 == ']')
			{
				m.w_points_arr.push_back(p);
				return input;
			}
			else
				throw std::invalid_argument("bad_input_Point");
		}

		
	}
	void getData(std::vector<cv::Point3f>& outRef)
	{
		if (w_points_arr.empty()) throw std::invalid_argument("No data available to reference");
		outRef = w_points_arr;
	}
};