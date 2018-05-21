#include "stdafx.h"
#include "common.h"
#include <queue>
#include <random>
#include <math.h> 
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

Mat convertMatFloatToUchar3(Mat src)
{
	float goutmin = 0;
	float goutmax = 255;
	float ginminR = src.at<Vec3f>(0, 0)[2], ginmaxR = src.at<Vec3f>(0, 0)[2];
	float ginminG = src.at<Vec3f>(0, 0)[1], ginmaxG = src.at<Vec3f>(0, 0)[1];
	float ginminB = src.at<Vec3f>(0, 0)[0], ginmaxB = src.at<Vec3f>(0, 0)[0];
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if (src.at<Vec3f>(i, j)[0] < ginminB)
				ginminB = src.at<Vec3f>(i, j)[0];
			if (src.at<Vec3f>(i, j)[1] < ginminG)
				ginminG = src.at<Vec3f>(i, j)[1];
			if (src.at<Vec3f>(i, j)[2] < ginminR)
				ginminR = src.at<Vec3f>(i, j)[2];


			if (src.at<Vec3f>(i, j)[0] > ginmaxB)
				ginmaxB = src.at<Vec3f>(i, j)[0];
			if (src.at<Vec3f>(i, j)[1] > ginmaxG)
				ginmaxG = src.at<Vec3f>(i, j)[1];
			if (src.at<Vec3f>(i, j)[2] > ginmaxR)
				ginmaxR = src.at<Vec3f>(i, j)[2];
		}
	}

	Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC3);
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			dst.at<Vec3b>(i, j)[0] = goutmin + (goutmax - goutmin) / (ginmaxB - ginminB)*(src.at<Vec3f>(i, j)[0] - ginminB);
			dst.at<Vec3b>(i, j)[1] = goutmin + (goutmax - goutmin) / (ginmaxG - ginminG)*(src.at<Vec3f>(i, j)[1] - ginminG);
			dst.at<Vec3b>(i, j)[2] = goutmin + (goutmax - goutmin) / (ginmaxR - ginminR)*(src.at<Vec3f>(i, j)[2] - ginminR);
		}
	}
	return dst;
}


Mat convertToSingleScaleRetinex3(Mat src, int sigma)
{
	Mat dst = Mat::zeros(src.rows, src.cols, CV_32FC3);
	Mat gaussian = Mat::zeros(src.rows, src.cols, CV_8UC3);
	GaussianBlur(src, gaussian, Size(0, 0), sigma,(0.0));
	//imshow("Gaus ", gaussian);
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			dst.at<Vec3f>(i, j)[0] = log10(src.at<Vec3b>(i, j)[0]) - log10(gaussian.at<Vec3b>(i, j)[0]);
			dst.at<Vec3f>(i, j)[1] = log10(src.at<Vec3b>(i, j)[1]) - log10(gaussian.at<Vec3b>(i, j)[1]);
			dst.at<Vec3f>(i, j)[2] = log10(src.at<Vec3b>(i, j)[2]) - log10(gaussian.at<Vec3b>(i, j)[2]);
		}
	}

	return dst;
}

Mat convertToMultiScaleRetinex3(Mat src, int sigma1, int sigma2, int sigma3 )
{

	Mat dst = Mat::zeros(src.rows, src.cols, CV_32FC3);
	Mat ret1 = convertToSingleScaleRetinex3(src, sigma1);
	Mat ret2 = convertToSingleScaleRetinex3(src, sigma2);
	Mat ret3 = convertToSingleScaleRetinex3(src, sigma3);
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			dst.at<Vec3f>(i, j)[0] = (ret1.at<Vec3f>(i, j)[0] + ret2.at<Vec3f>(i, j)[0] + ret3.at<Vec3f>(i, j)[0]) / 3;
			dst.at<Vec3f>(i, j)[1] = (ret1.at<Vec3f>(i, j)[1] + ret2.at<Vec3f>(i, j)[1] + ret3.at<Vec3f>(i, j)[1]) / 3;
			dst.at<Vec3f>(i, j)[2] = (ret1.at<Vec3f>(i, j)[2] + ret2.at<Vec3f>(i, j)[2] + ret3.at<Vec3f>(i, j)[2]) / 3;
			/*printf("%.2f %.2f %.2f  ", ret1.at<Vec3f>(i, j)[0], ret1.at<Vec3f>(i, j)[1], ret1.at<Vec3f>(i, j)[2]);
			printf("%.2f %.2f %.2f  ", ret2.at<Vec3f>(i, j)[0], ret2.at<Vec3f>(i, j)[1], ret2.at<Vec3f>(i, j)[2]);
			printf("%.2f %.2f %.2f  \n", ret3.at<Vec3f>(i, j)[0], ret3.at<Vec3f>(i, j)[1], ret3.at<Vec3f>(i, j)[2]);*/
		}
	}

	return dst;
}


Mat getColorRestore3(Mat src, double alpha, double beta)
{
	Mat dst = Mat::zeros(src.rows, src.cols, CV_32FC3);
	Mat sum = Mat::zeros(src.rows, src.cols, CV_32SC1);
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			sum.at<int>(i, j) = src.at<Vec3b>(i, j)[0] + src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i, j)[2];
		}
	}
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			dst.at<Vec3f>(i, j)[0] = beta*(log10(src.at<Vec3b>(i, j)[0] * alpha) - log10(sum.at<int>(i, j)));
			dst.at<Vec3f>(i, j)[1] = beta*(log10(src.at<Vec3b>(i, j)[1] * alpha) - log10(sum.at<int>(i, j)));
			dst.at<Vec3f>(i, j)[2] = beta*(log10(src.at<Vec3b>(i, j)[2] * alpha) - log10(sum.at<int>(i, j)));
		}
	}
	return dst;

}

Mat convertToMSRCR(Mat src, double G, double b, int sigma1, int sigma2, int sigma3)
{
	Mat dstf = Mat::zeros(src.rows, src.cols, CV_32FC3);

	Mat retinex = convertToMultiScaleRetinex3(src,sigma1,sigma2,sigma3);
	Mat color = getColorRestore3(src, 125, 46);
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			dstf.at<Vec3f>(i, j)[0] = G*(retinex.at<Vec3f>(i, j)[0] * color.at<Vec3f>(i, j)[0] + b);
			dstf.at<Vec3f>(i, j)[1] = G*(retinex.at<Vec3f>(i, j)[1] * color.at<Vec3f>(i, j)[1] + b);
			dstf.at<Vec3f>(i, j)[2] = G*(retinex.at<Vec3f>(i, j)[2] * color.at<Vec3f>(i, j)[2] + b);
			/*printf("%f %f %f \t", dstf.at<Vec3f>(i, j)[0], dstf.at<Vec3f>(i, j)[1], dstf.at<Vec3f>(i, j)[2]);
			printf("%f %f %f \t", color.at<Vec3f>(i, j)[0], color.at<Vec3f>(i, j)[1], color.at<Vec3f>(i, j)[2]);
			printf("%f %f %f \n", retinex.at<Vec3f>(i, j)[0], retinex.at<Vec3f>(i, j)[1], retinex.at<Vec3f>(i, j)[2]);*/
		}
	}

	return convertMatFloatToUchar3(dstf);
}

Mat hackyGaussianBlur(Mat src,int sigma)
{
	Mat gaussian = Mat::zeros(src.rows, src.cols, CV_8UC1);
	Mat gaussian3 = Mat::zeros(src.rows, src.cols, CV_8UC3);
	Mat src3 = Mat::zeros(src.rows, src.cols, CV_8UC3);

	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			src3.at<Vec3b>(i, j)[0] = src.at<uchar>(i,j);
			src3.at<Vec3b>(i, j)[1] = src.at<uchar>(i, j);
			src3.at<Vec3b>(i, j)[2] = src.at<uchar>(i, j);
		}
	}

	GaussianBlur(src3, gaussian3, Size(0, 0), sigma, (0.0));

	for (int i = 0; i < gaussian3.rows; i++)
	{
		for (int j = 0; j < gaussian3.cols; j++)
		{
			gaussian.at<uchar>(i, j) = gaussian3.at<Vec3b>(i, j)[0];
		}
	}

	return gaussian;
}

Mat convertToSingleScaleRetinex1(Mat src, int sigma)
{
	Mat dst = Mat::zeros(src.rows, src.cols, CV_32FC1);
	Mat gaussian = Mat::zeros(src.rows, src.cols, CV_8UC1);
	GaussianBlur(src, gaussian, Size(0, 0), sigma, (0.0));
	//gaussian =hackyGaussianBlur(src, sigma);
	//imshow("Gaus ", gaussian);
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			dst.at<float>(i, j) = log10(src.at<uchar>(i, j)) - log10(gaussian.at<uchar>(i, j));
		}
	}

	return dst;
}

Mat convertToMultiScaleRetinex1(Mat src, int sigma1, int sigma2, int sigma3)
{

	Mat dst = Mat::zeros(src.rows, src.cols, CV_32FC1);
	Mat ret1 = convertToSingleScaleRetinex1(src, sigma1);
	Mat ret2 = convertToSingleScaleRetinex1(src, sigma2);
	Mat ret3 = convertToSingleScaleRetinex1(src, sigma3);
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			dst.at<float>(i, j)= (ret1.at<float>(i, j) + ret2.at<float>(i, j) + ret3.at<float>(i, j)) / 3;
		}
	}

	return dst;
}


Mat convertMatFloatToUchar1(Mat src)
{
	float goutmin = 0;
	float goutmax = 255;
	float ginminR = src.at<float>(0, 0), ginmaxR = src.at<float>(0, 0);
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if (src.at<float>(i, j) < ginminR)
				ginminR = src.at<float>(i, j);

			if (src.at<float>(i, j)> ginmaxR)
				ginmaxR = src.at<float>(i, j);
		}
	}

	Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC1);
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			dst.at<uchar>(i, j)= goutmin + (goutmax - goutmin) / (ginmaxR - ginminR)*(src.at<float>(i, j) - ginminR);
		}
	}
	return dst;
}

int maxim(int a, int b, int c)
{
	if (a > b)
	{
		return a > c ? a : c;
	}
	else
	{
		return b > c ? b : c;
	}
}

float minim(float a, float b)
{
	return a < b ? a : b;
}

Mat convertToMSRCP(Mat src, int sigma1, int sigma2, int sigma3)
{
	Mat intensity = Mat::zeros(src.rows, src.cols, CV_8UC1);
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			intensity.at<uchar>(i, j) =( src.at<Vec3b>(i, j)[0] + src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i, j)[2])/3;
		}
	}
	Mat retinex = convertToMultiScaleRetinex1(intensity, sigma1, sigma2, sigma3);
	retinex = convertMatFloatToUchar1(retinex);
	imshow("Intensity", retinex);


	Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC3);
	int b;
	float a;
	float a1, a2;
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			b = maxim(src.at<Vec3b>(i, j)[0], src.at<Vec3b>(i, j)[1], src.at<Vec3b>(i, j)[2]);
			a1 = 255. / b;
			a2 = (float)retinex.at<uchar>(i, j) / intensity.at<uchar>(i, j);
			a = minim(a1, a2) *0.90;
			//printf("%d ", a);
			dst.at<Vec3b>(i, j)[0] = src.at<Vec3b>(i, j)[0] * a;
			dst.at<Vec3b>(i, j)[1] = src.at<Vec3b>(i, j)[1] * a;
			dst.at<Vec3b>(i, j)[2] = src.at<Vec3b>(i, j)[2] * a;

			//printf("%f _ %d %f _ %d %f _ %d %f\n\n", a,
				//src.at<Vec3b>(i, j)[0], src.at<Vec3b>(i, j)[0] * a,
				//src.at<Vec3b>(i, j)[1], src.at<Vec3b>(i, j)[1] * a,
				//src.at<Vec3b>(i, j)[2], src.at<Vec3b>(i, j)[2] * a);
		}
	}
	return dst;//TODO
}

int main(int argc, char ** argv)
{
	char filename[MAX_PATH];
	while (openFileDlg(filename))
	{
		Mat src = imread(filename);
		if (src.empty())
			return -1;

		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				src.at<Vec3b>(i, j)[0] = src.at<Vec3b>(i, j)[0] + 1 >255 ? 255 : src.at<Vec3b>(i, j)[0] + 1;
				src.at<Vec3b>(i, j)[1] = src.at<Vec3b>(i, j)[1] + 1>255 ? 255 : src.at<Vec3b>(i, j)[1] + 1;
				src.at<Vec3b>(i, j)[2] = src.at<Vec3b>(i, j)[2] + 1>255 ? 255 : src.at<Vec3b>(i, j)[2] + 1;

			}
		}


		//convertToSingleScaleRetinex(src, 240);
		//getColorRestore(src, 125, 46);
		imshow("Original", src);
		imshow("MSRCR", convertToMSRCR(src, 5, 25, 15, 80, 170));
		//imshow("MSRCR2", convertToMSRCR(src, 5, 25, 15, 80, 200));
		imshow("MSRCP", convertToMSRCP(src, 5, 50, 170));
		//imshow("MSRCP2", convertToMSRCP(src, 5, 50, 200));
		waitKey(0);
	}
	return 0;
}
