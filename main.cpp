#include "opencv2/opencv.hpp"

#include "TL_Classifier.h"

#include <cstdio>
#include <vector>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

int main()
{
	TL_Classifier *test = new TL_Classifier;

	string model_path = ".\\mytrain_knn.dat"; // model 경로
	string image_path = ".\\pro_class_sample\\"; // test할 폴더 경로
	//string image_path = ".\\mytest\\"; // test할 폴더 경로

	test->load_model(model_path); // model load

	int result_sum = 0;
	int result = 0;
	Mat test_img;
	vector<Mat> test_img_set;
	vector<int> test_label_set;
	test->load_img(image_path); // test img load
	test_img_set = test->get_test_images(); 
	test_label_set = test->get_test_labels();
	for (int i = 0; i < test_img_set.size(); i++)
	{
		test_img = test_img_set[i];
		result = test->pred_class(test_img); // classification
		if (test_label_set[i] == result)
		{
			//cout << "O맞았다! 실제 ID :" << test_label_set[i] << " , 예측한 ID :" << result << endl;
			result_sum += 1;
		}
		else
		{
			if (test_label_set[i] == -1)
			{
				cout << result << "로 판단하였지만, 정답 Label이 없습니다." << endl;
			}
			//cout << "X틀렸네? 실제 ID :" << test_label_set[i] << " , 예측한 ID :" << result << endl;
		}
	}
	double accuracy = 100* ((double)result_sum / (double)test_label_set.size());
	cout << "전체 test data 개수 " << test_label_set.size() << "개 중 " << result_sum << "개 정답이므로 정확도는 " << accuracy << "%" << endl;
}




void houghCircles_process(Mat input)
{
	Mat cornerMap, Green_Binary_img;
	Mat Hsv_img, Gray_img, Filteredimg;
	//cvtColor(input, Hsv_img, COLOR_BGR2GRAY);
	//cvtColor(input, Gray_img, COLOR_BGR2GRAY);
	Hsv_img = input;
	Gray_img = input;
	
	double sigmaColor = 10.0;
	double sigmaSpace = 10.0;

	bilateralFilter(Gray_img, Filteredimg, -1, sigmaColor, sigmaSpace);
	vector<Vec3f> circles;

	HoughCircles(Filteredimg, circles, HOUGH_GRADIENT, 1, 100, 50, 35, 3, 0); //원 그리는 함수 
	//image,circles,method,dp,minDist,param1,param2,minRadius,maxRadius
	for (int i = 0; i < circles.size(); i++) { cout << "center: " << circles[i][0] << endl; }//각 원 순서대로 중심점 색 정보 출력
	/*-------------------------------------*/
	/*-------------검출 된 원 내부 색상정보----------------*/
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		circle(Gray_img, center, 1, Scalar(0, 0, 255), 1, 2, 0);//circle center 화면에 원 그리는 함수 
		circle(Gray_img, center, radius, Scalar(0, 0, 255), 2, 2, 0);//outline  화면에 원 그리는 함수 
		cout << "center: " << center << "radius: " << radius << endl;
		//imshow("Gray_Img",Gray_Img);

		int k = circles[i][0];//x 
		int j = circles[i][1];//y
		//--------------
		cout << "k: " << k << "j: " << j << endl;
		int B = input.at<Vec3b>(j, k)[0]; //화소접근시 x,y좌표 반대로 되므로 주의!!! 
		int G = input.at<Vec3b>(j, k)[1];
		int R = input.at<Vec3b>(j, k)[2];

		cout << "B" << B << endl;
		cout << "G" << G << endl;
		cout << "R" << R << endl;

		if (R - B > 20 && R - G > 20 && R > 100) { cout << "Stop due to the RedLight" << endl; }

		if (G - R > 20 && G - B > 20 && G > 100)
		{
			cout << "Detected GreenLight" << endl;
			int CenterX = circles[i][0];
			int CenterY = circles[i][1];
			int Cradius = circles[i][2]; //반지름 
			int Width = 2 * circles[i][2]; //width=height (circle) -> rect로 근사 
			int Height = 2 * circles[i][2];
			int Green_cnt = 0;
			int Green_data, Red_data, Blue_data;
			int TotalPixel = cvFloor(3.1415 * Cradius * Cradius);

			for (int k = CenterX - Width; k <= CenterX + Width; k++)//This needs to be made
			{
				for (int j = CenterY - Height; j <= CenterY + Height; j++)
				{
					int Blue_data = input.at<Vec3b>(j, k)[0];
					int Green_data = input.at<Vec3b>(j, k)[1];
					int Red_data = input.at<Vec3b>(j, k)[2];

					if (Green_data - Blue_data > 20 && Green_data - Red_data > 20 && Green_data > 100) {
						Green_cnt++;
					}
				}
			}cout << "Green_cnt: " << Green_cnt << " TotalPixel: " << TotalPixel << endl;

			if (TotalPixel * 0.6 < Green_cnt) { cout << "--------------------------GoRight" << endl; }
			else { cout << "---------------------------GoLeft" << endl; }
		}
		else { cout << "stay stop!" << endl; }
	}
}
