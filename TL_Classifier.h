#pragma once

// general includes
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
using namespace std;

// OpenCV includes
#include "opencv2/opencv.hpp"
using namespace cv;

class TL_Classifier
{
private:

    int pred_class_ID;

    vector<int> labels;
    vector<int> labels_train;
    vector<int> labels_test;

    vector<Mat> images;
    vector<Mat> images_train;
    vector<Mat> images_test;



public:
    Ptr<cv::ml::KNearest> knn;

    TL_Classifier();
    ~TL_Classifier();

    int size_average_cols;
    int size_average_rows;

    // call private function
    vector<Mat> get_images() {
        return images;
    }
    vector<Mat> get_train_images() {
        return images_train;
    }
    vector<Mat> get_test_images() {
        return images_test;
    }
    vector<int> get_labels() {
        return labels;
    }
    vector<int> get_train_labels() {
        return labels_train;
    }
    vector<int> get_test_labels() {
        return labels_test;
    }
    Ptr<cv::ml::KNearest> get_knn() {
        return knn;
    }

    // essential function
    void load_model(string model_path);
    int pred_class(Mat traffic_light);

    // made function
    void load_img(string img_path);
    Mat preprocess(Mat src);
    int calc_mode(vector<int> input);

    // for svm (not used now)
    vector< vector<float> > ExtractFeatures(Mat img, vector<int>* left = NULL, vector<int>* top = NULL);
    bool data_arrange(vector<Mat> images, int label, int num_for_test,
        vector<float>& trainingData, vector<int>& responsesData,
        vector<float>& testData, vector<float>& testResponsesData);

};

