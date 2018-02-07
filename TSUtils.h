#pragma once

#include <cmath>
#include <vector>

class TSUtils
{
public:

	TSUtils(void)
	{
	}

	~TSUtils(void)
	{
	}

	static std::vector<double> znormalize( const std::vector<double> & ad, int i) 
	{
		int j = ad.size();
		int k = j - i;
		std::vector<double> ad1 ( k );
		double d = mean(ad, i, j);
		double d1 = _std(ad, i, j, d, true);
		for (int l = 0; l < k; l++) {
			ad1[l] = (ad[i + l] - d) / d1;
		}

		return ad1;
	}

	static double mean( const std::vector<double> & ad, int i, int j) 
	{
		double d = 0.0;
		for (int k = i; k < j; k++) {
			d += ad[k];
		}
		return d / (double)(j - i);
	}


	static double _std(const std::vector<double> &ad, int i, int j, double d, bool flag) {
		int k = j - i;
		if (k == 1) {
			return ad[i];
		}
		if (!flag) {
			d = mean(ad, i, j);
		}
		double d1 = 0.0;
		double d2 = 0.0;
		for (int l = i; l < j; l++) {
			double d3 = ad[l] - d;
			d1 += d3 * d3;
		}

		d1 /= k - 1;
		return sqrt(d1);
	}
};

