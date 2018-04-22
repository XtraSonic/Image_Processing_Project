// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"

struct Location
{
	int row;
	int col;
};

struct DimletLocationList {
	Location location;
	DimletLocationList* next;
};

DimletLocationList* create9Dimlet()
{
	Location center = { 0,0 };
	Location up = { -1,0 };
	Location down = { 1,0 };
	Location left = { 0,-1 };
	Location right = { 0,1 };
	Location nw = { -1,-1 };
	Location ne = { -1,1 };
	Location sw = { 1,-1 };
	Location se = { 1,1 };

	DimletLocationList* aa = (DimletLocationList*)malloc(sizeof(DimletLocationList));
	aa->location = nw;
	aa->next = NULL;
	DimletLocationList* bb = (DimletLocationList*)malloc(sizeof(DimletLocationList));
	bb->location = ne;
	bb->next = aa;
	DimletLocationList* cc = (DimletLocationList*)malloc(sizeof(DimletLocationList));
	cc->location = sw;
	cc->next = bb;
	DimletLocationList* dd = (DimletLocationList*)malloc(sizeof(DimletLocationList));
	dd->location = se;
	dd->next = cc;


	DimletLocationList* a = (DimletLocationList*)malloc(sizeof(DimletLocationList));
	a->location = right;
	a->next = dd;
	DimletLocationList* b = (DimletLocationList*)malloc(sizeof(DimletLocationList));
	b->location = left;
	b->next = a;
	DimletLocationList* c = (DimletLocationList*)malloc(sizeof(DimletLocationList));
	c->location = down;
	c->next = b;
	DimletLocationList* d = (DimletLocationList*)malloc(sizeof(DimletLocationList));
	d->location = up;
	d->next = c;
	DimletLocationList* res = (DimletLocationList*)malloc(sizeof(DimletLocationList));
	res->location = center;
	res->next = d;

	return res;
}

void runOpenLoop(void(*function)(Mat*))
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		printf("Opening img \n");
		src = imread(fname);
		//Create a window
		namedWindow("Original", 1);

		//run function
		(*function)(&src);


		//show the image
		imshow("Original", src);

		// Wait until key press
		waitKey(0);
		printf("\nimg closed\n\n");
	}
}

Mat correctGamma(Mat* src, double gamma)
{
	float L = 255;
	Mat dst = Mat::zeros(src->rows, src->cols, CV_8UC3);
	for (int i = 0; i < src->rows; i++)
	{
		for (int j = 0; j < src->cols; j++)
		{
			dst.at<Vec3b>(i, j)[0] = L*pow((src->at<Vec3b>(i, j)[0] / L), gamma);
			dst.at<Vec3b>(i, j)[1] = L*pow((src->at<Vec3b>(i, j)[1] / L), gamma);
			dst.at<Vec3b>(i, j)[2] = L*pow((src->at<Vec3b>(i, j)[2] / L), gamma);
		}
	}
	return dst;
}

void gausFilterConcept(Mat* src)
{

	Mat dst = Mat::zeros(src->rows, src->cols, CV_8UC3);

	int sig = 5;
	int size = ((int)((sig + 0.8)/0.3)+1)*2+1;
	GaussianBlur(*src, dst, Size(size, size),0, 0, BORDER_REPLICATE
	);
	imshow("BLur ", dst);
	waitKey(0);
	sig = 20;
	size = ((int)((sig + 0.8) / 0.3) + 1) * 2 + 1;
	GaussianBlur(*src, dst, Size(size, size), 0, 0, BORDER_REPLICATE
	);
	imshow("BLur ", dst);
	waitKey(0);
	sig = 240;
	size = 4*sig + 1;
	GaussianBlur(*src, dst, Size(size, size), 0, 0, BORDER_REPLICATE
	);
	imshow("BLur ", dst);
	waitKey(0);
	sig = 4000;
}

int main()
{
	int op;

	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		printf("Opening img \n");
		src = imread(fname);
		//Create a window
		namedWindow("Original", 1);

		//run function

		Mat dst1 = Mat::zeros(src.rows, src.cols, CV_8UC3);
		Mat dst2 = Mat::zeros(src.rows, src.cols, CV_8UC3);
		
		// convert from sRGB to liniar space
		dst1 = correctGamma(&src, 2.2);

		gausFilterConcept(&dst1);

		// convert from liniar space to  sRGB
		dst2 = correctGamma(&dst1, 1/2.2);

		imshow("lin", dst1);
		imshow("new Original", dst2);

		//show the image
		imshow("Original", src);

		// Wait until key press
		waitKey(0);
		printf("\nimg closed\n\n");
	}

	return 0;
}
/*
TEMPLATES
for (int i = 0; i < src->rows; i++)
	{
		for (int j = 0; j < src->cols; j++)
		{
			if (src->at<Vec3b>(i, j) == color)
			{
			}
		}
	}

	
Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC3);


Mat dst = Mat::zeros(src->rows, src->cols, CV_8UC3);

for (int i = 0; i < dst.rows; i++)
	{
		for (int j = 0; j < dst.cols; j++)
		{
			dst.at<Vec3b>(i, j) = WHITE;

		}
	}
*/
