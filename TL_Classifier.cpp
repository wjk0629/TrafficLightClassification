#include "TL_Classifier.h"

// ������~
TL_Classifier::TL_Classifier()
{
    images.clear();
    images_train.clear();
    images_test.clear();

    labels.clear();
    labels_train.clear();
    labels_test.clear();
    
    this->pred_class_ID = -1;
    this->size_average_cols = 85;
    this->size_average_rows = 29;


}
// ���~
TL_Classifier::~TL_Classifier()
{

}



// for KNN (used now)
// Knearest�� train�� �̿��Ͽ� �н��մϴ�.
Ptr<cv::ml::KNearest> train_knn(TL_Classifier* mytrain, int col_size, int row_size)
{
    //TL_Classifier* mytrain = new TL_Classifier;

    Mat train_images, train_labels;
    vector<Mat> digits = mytrain->get_train_images();
    vector<int> labels = mytrain->get_train_labels();

    vector<Mat> test_img;

    for (int i = 0; i < labels.size(); i++) {
        Mat roi, roi_float, roi_flatten;
        roi = mytrain->preprocess(digits[i]);
        roi.convertTo(roi_float, CV_32F);
        roi_flatten = roi_float.reshape(1, 1);
        train_images.push_back(roi_flatten);
        train_labels.push_back(labels[i]);
    }
    Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();
    knn->train(train_images, cv::ml::ROW_SAMPLE, train_labels);
    return knn;
}

// model �� load �մϴ�. --> (model �� ���� ���) ���⼭ �� �ۼ�Ʈ�� validation �Ҳ��� ���� ����
void TL_Classifier::load_model(string model_path)
{
	knn = get_knn();
	// load model
	ifstream file;
	file.open(model_path);
	knn = knn->load(model_path);
	
	if (knn.empty() || !file)
	{
		file.close();
		// if there is no model (first time)
	    cout << "there is no model ! ! !" << endl;
		cout << "for making model with knn train . . ." << endl;
		// method1 : KNearest
		string image_path = ".\\pro_class_sample\\";

		vector<Mat> images;
		vector<Mat> images_resize;
		vector<Mat> images_train_resize;
		vector<Mat> images_test_resize;

		Mat img;
		Mat img_resize;

		vector<int> labels;
		int label;

		int num_class = 9;
		string path_now;
		
		int size_cols;
		int size_rows;
		int size_cols_sum = this->size_average_cols;
		int size_rows_sum = this->size_average_rows;
		int size_str_sum = 0;

		// ������ train. test �Ҳ��� true, �ƴϸ� fasle
		bool data_split = true;


		// load image part
		cout << "data load is started . . ." << endl;
		for (int i = 0; i < num_class; i++)
		{

			path_now = image_path + to_string(i);

			vector<string> str;

			glob(path_now, str, false);

			int train_num, test_num;
			train_num = (int)str.size() * 0.8; // 80% �� train���� ���ڴ�.
			test_num = str.size() - train_num;
			cout << i << "th class data number loaded : " << str.size() << ", and train data num : " << train_num << ", else : test " << endl;
			size_str_sum += str.size();
			if (str.size() == 0) { cout << "there is no img data.\n" << endl; }

			for (int cnt = 0; cnt < str.size(); cnt++)
			{
				img = imread(str[cnt]);
				label = i;
				size_cols_sum += img.cols;
				size_rows_sum += img.rows;

				images.push_back(img);
				labels.push_back(label);
				if (data_split == true)
				{
					if (cnt < train_num && data_split)
					{
						images_train.push_back(img);
						labels_train.push_back(label);
					}
					else
					{
						images_test.push_back(img);
						labels_test.push_back(label);
					}
				}
				else
				{
					images_train.push_back(img);
					labels_train.push_back(label);
				}
			}
		}
		string path_test;
		if (data_split == false)
		{
			vector<string> tet;
			glob(path_test, tet, false);
			if (tet.size() == 0) { cout << "there is no test data. (All of data used train)\n" << endl; }

			for (int cnt = 0; cnt < tet.size(); cnt++)
			{
				img = imread(tet[cnt]);
				images_test.push_back(img);
				// label�� �˷���� �������� �ƴµ�.. ���?
			}
		} // ���߿� ������ ��

		// resize part ( according to the average value ) 
		size_cols = size_cols_sum / size_str_sum;
		size_rows = size_rows_sum / size_str_sum;
		this->size_average_cols = size_cols;
		this->size_average_rows = size_rows;
		for (int cnt2 = 0; cnt2 < images.size(); cnt2++)
		{
			resize(images[cnt2], img_resize, Size(size_cols, size_rows));
			Mat testimg = img_resize.clone();
			images_resize.push_back(testimg);
		}
		for (int cnt2 = 0; cnt2 < images_train.size(); cnt2++)
		{
			resize(images_train[cnt2], img_resize, Size(size_cols, size_rows));
			Mat testimg = img_resize.clone();
			images_train_resize.push_back(testimg);
		}
		for (int cnt2 = 0; cnt2 < images_test.size(); cnt2++)
		{
			resize(images_test[cnt2], img_resize, Size(size_cols, size_rows));
			Mat testimg = img_resize.clone();
			images_test_resize.push_back(testimg);
		}

		cout << "total data number : " << images_resize.size() << endl;
		this->images = images_resize;
		this->images_train = images_train_resize;
		this->images_test = images_test_resize;

		this->labels = labels;
		this->labels_train = labels_train;
		this->labels_test = labels_test;

		cout << "data load is completed . . . " << endl;

		knn = train_knn(this, this->size_average_cols, this->size_average_rows);
		if (knn.empty()) {
			cerr << "Training failed!" << endl;
			return;
		}
		this->knn = knn;
		knn->save("mytrain_knn.dat");
		cout << "save the model ! ! ! " << endl;
	}
	else
	{
	    file.close();
	    this->knn = knn; 
		cout << "model load is completed ! ! ! " << endl;
	}
}

// image �� load �մϴ�. (model test��) --> ���⼭ total class number ���� ����, mytest ���� ����
void TL_Classifier::load_img(string img_path)
{
	// img_path ��ο� �ִ� img ��  load �Ѵ�.
	// load �� �̹����� images_test�� data ��,
	//                  labels_test�� ID �� ����Ǿ� test ������ ���δ�.
	// ���� ��Ȯ���� �ʿ��ϴٸ� img_path ���ο� �������� ������ �־���� ������ �־�� �Ѵ�.
	// �׷��� �׳� �������� �̹����� classification�� �ϰ� �ʹٸ�(������ �´��� �ƴ����� ������)
	// need_acc�� false�� �θ� �ȴ�.
	bool need_acc = true;
	Mat img;
	int label;
	// ���⿡ class�� �����մϴ�.
	int num_class = 9;
	string path_now;
	if (need_acc == true)
	{
		cout << "test data load is started . . ." << endl;
		for (int i = 0; i < num_class; i++)
		{

			path_now = img_path + to_string(i);

			vector<string> str;

			glob(path_now, str, false);

			cout << "(For test)  " << i << "th class data number loaded : " << str.size() << endl;
			if (str.size() == 0) { cout << "there is no img data.\n" << endl; }

			for (int cnt = 0; cnt < str.size(); cnt++)
			{
				img = imread(str[cnt]);
				label = i;
				images_test.push_back(img);
				labels_test.push_back(label);
			}
		}
	}
	else // ���� �ȿ� �ƹ� ��ȣ�� �����̳� �������� ����ְ� ����� ���������, �ٸ� �󺧸��� ���Ͽ� �������� �ƴ��� �Ǵ� �Ұ�
	{
		cout << "test data load is started . . . (only classfication ver)" << endl;
		path_now = img_path;
		
		vector<string> str;

		glob(path_now, str, false);

		cout << "(For test)  total data number loaded : " << str.size() << endl;
		if (str.size() == 0) { cout << "there is no img data.\n" << endl; }

		for (int cnt = 0; cnt < str.size(); cnt++)
		{
			img = imread(str[cnt]);
			label = -1; // ���� �����Ƿ� -1�� labeling
			images_test.push_back(img);
			labels_test.push_back(label);
		}
	}
}

// ID �� classification �մϴ�. --> ���⼭ k�� ���� ����
int TL_Classifier::pred_class(Mat traffic_light)
{
    int pred_class_ID = -1;
	int k_max = 9;
	int k = 5;
	// ���⼭ k_use�� true�� �ϸ� k�� 1~k_max���� �� �غ��� �� �� ���� ���� Ŭ������ �����Ѵ�.
	// �׷��� �ð��� ���� �ɸ�����. �׷��� k_use�� false�� �ϸ� k������ �ѹ��� �����Ѵ�.
	bool k_use = true;
	vector<int> res_tmp;
	int res_final;
	Mat resize_img;
	Mat feature_img;
	Mat res;

	resize(traffic_light, resize_img, Size(this->size_average_cols, this->size_average_rows));
	feature_img = TL_Classifier::preprocess(resize_img);
	feature_img.convertTo(feature_img, CV_32F);
	feature_img = feature_img.reshape(1, 1);
	knn = get_knn();
	if (k_use == true)
	{
		for (int j = 1; j <= k_max; j++)
		{
			knn->findNearest(feature_img, j, res);
			res_tmp.push_back(cvRound(res.at<float>(0, 0)));
		}
		res_final = TL_Classifier::calc_mode(res_tmp);
	}
	else
	{
		knn->findNearest(feature_img, k, res);
		res_final = cvRound(res.at<float>(0, 0));
	}
	
	pred_class_ID = res_final;

    return pred_class_ID;
}

// preprocess �� �����մϴ�. --> ���⼭ ���� ���� ��� ���� ���� ����
Mat TL_Classifier::preprocess(Mat src)
{
	// preprocess image
	// hsv�� ��ȯ�Ѵ��� ���򺰷� �з�
	Mat cornerMap, Green_Binary_img;
	Mat Hsv_img, Gray_img, Filteredimg;
	//bilateralFilter(src, src, 0, 30, 30);
	cvtColor(src, Hsv_img, COLOR_BGR2HSV);
	cvtColor(src, Gray_img, COLOR_BGR2GRAY);

	Scalar lower_green = Scalar(45, 60, 60);
	Scalar upper_green = Scalar(100, 240, 255);

	Scalar lower_yellow = Scalar(15, 60, 60);
	Scalar upper_yellow = Scalar(40, 240, 255);

	Scalar lower_red1 = Scalar(0, 0, 60);
	Scalar upper_red1 = Scalar(15, 255, 255);
	Scalar lower_red2 = Scalar(160, 0, 60);
	Scalar upper_red2 = Scalar(180, 255, 255);

	Mat green_mask, yellow_mask, red_mask, red_mask1, red_mask2;

	inRange(Hsv_img, lower_green, upper_green, green_mask);
	inRange(Hsv_img, lower_yellow, upper_yellow, yellow_mask);
	inRange(Hsv_img, lower_red1, upper_red1, red_mask1);
	inRange(Hsv_img, lower_red2, upper_red2, red_mask2);
	red_mask = red_mask1 | red_mask2;

	// ��ü ã��.. ũ�� ������ �Ĺ�����.. �߽��� �����ڸ��ΰ� �Ĺ�����.. �ʹ� ū�͵� �Ĺ�����..
	// ������� �������� ȥ���ϴ� ��찡 ���Ƽ� ����ġ �����ִ� temp_mask�� �������.
	vector<vector<Point> >contours;
	vector<Vec4i>hierarchy;
	double maxArea = src.cols * src.rows / 2.5;
	double minArea = 20.0;
	double x_margin = 8.0;
	double y_margin = 8.0;
	double switch_margin = 15;
	Moments m_g, m_y, m_r;
	Mat temp_mask;
	bool switch_bool = false;
	// find green
	findContours(green_mask.clone(), contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_NONE, Point());
	for (int i = 0; i < contours.size(); i++)
	{
		m_g = moments(contours[i], true);
		Point c_g(m_g.m10 / m_g.m00, m_g.m01 / m_g.m00);
		double area = contourArea(contours[i]);
		if (c_g.x > src.cols - x_margin || c_g.x < x_margin || c_g.y > src.rows - y_margin || c_g.y < y_margin || area < minArea || area > maxArea)
		{
			drawContours(green_mask, contours, i, Scalar(0), FILLED, 8);
		}
	}
	// find yellow
	findContours(yellow_mask.clone(), contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_NONE, Point());
	for (int i = 0; i < contours.size(); i++)
	{
		m_y = moments(contours[i], true);
		Point c_y(m_y.m10 / m_y.m00, m_y.m01 / m_y.m00);
		double area = contourArea(contours[i]);
		if ((c_y.x > src.cols - x_margin || c_y.x < x_margin || c_y.y > src.rows - y_margin || c_y.y < y_margin || area < minArea || area > maxArea))
		{
			drawContours(yellow_mask, contours, i, Scalar(0), FILLED, 8);
		}
		else if ((c_y.x < src.cols / 3 - switch_margin || c_y.x > 2 * src.cols / 3 + switch_margin) && switch_bool == false)
		{
			// ���� yellow�� ���Դµ� �Ӹ��� ��� ���� ������ yellow�� �ƴ϶� red��� ġ��
			temp_mask = red_mask;
			red_mask = yellow_mask;
			yellow_mask = temp_mask;
			switch_bool = true;
		}
	}
	// find red
	findContours(red_mask.clone(), contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_NONE, Point());
	for (int i = 0; i < contours.size(); i++)
	{
		m_r = moments(contours[i], true);
		Point c_r(m_r.m10 / m_r.m00, m_r.m01 / m_r.m00);
		double area = contourArea(contours[i]);
		if ((c_r.x > src.cols - x_margin || c_r.x < x_margin || c_r.y > src.rows - y_margin || c_r.y < y_margin || area < minArea || area > maxArea))
		{
			drawContours(red_mask, contours, i, Scalar(0), FILLED, 8);
		}
		else if ((c_r.x > src.cols / 3 + switch_margin) && switch_bool == false)
		{
			// ���� red�� ���Դµ� �Ӹ��� ���ʿ� ���� ������ red�� �ƴ϶� yellow��� ġ��
			temp_mask = yellow_mask;
			yellow_mask = red_mask;
			red_mask = temp_mask;
			switch_bool = true;
		}
	}

	switch_bool = false;
	Mat green_img, yellow_img, red_img;

	// for visualize. ������ ����ũ�Ѱ͸� ���� ��� ���̴��� ���� ���ؼ�.
	/*bitwise_and(Hsv_img, Hsv_img, green_img, green_mask);
	bitwise_and(Hsv_img, Hsv_img, yellow_img, yellow_mask);
	bitwise_and(Hsv_img, Hsv_img, red_img, red_mask);*/

	// morphology ����
	Mat element5(2, 2, CV_8U, Scalar(1));
	Mat red_mask_test, red_mask_test2;
	// CLOSED : ä�򿬻�(��â->ħ������ �޲�), OPEN : ���ſ���(ħ��->��â���� ����������) 
	morphologyEx(green_mask, green_mask, MORPH_CLOSE, element5);
	morphologyEx(yellow_mask, yellow_mask, MORPH_CLOSE, element5);
	morphologyEx(red_mask, red_mask, MORPH_CLOSE, element5);
	//morphologyEx(green_mask, green_mask, MORPH_OPEN, element5);
	//morphologyEx(yellow_mask, yellow_mask, MORPH_OPEN, element5);
	//morphologyEx(red_mask, red_mask, MORPH_OPEN, element5); 


	// ã���Ϳ��� �� houghCircle ã�Ƽ� ��ü�� ���̶�� ���
	/*houghCircles_process(green_mask);
	houghCircles_process(yellow_mask);
	houghCircles_process(red_mask);*/

	// ���� 3ä�� �̹����� �ϳ��� ��ģ �̹����� ����� Ư¡�̹������ ����
	Mat feature_image;
	vconcat(green_mask, yellow_mask, feature_image);
	vconcat(feature_image, red_mask, feature_image);

	return feature_image;
}

// �ֺ� �� ����մϴ�.
int TL_Classifier::calc_mode(vector<int> input)
{
	vector<int> count;
	int mode = 0;
	int max = 0;

	for (int i = 0; i < input.size(); i++)
	{
		count.push_back(0);
	}

	for (int j = 0; j < input.size(); j++)
	{
		count[input[j]]++;
	}

	for (int k = 0; k < input.size(); k++)
	{
		if (max < count[k]) {
			max = count[k];
			mode = k;
		}
	}

	return mode;
}




// for SVM (not used now)
// SVM�� train�� �̿��Ͽ� �н��մϴ�.
Ptr<cv::ml::SVM> train_svm(TL_Classifier* mytrain, int col_size, int row_size)
{
	Ptr<cv::ml::SVM> svm;
	//TL_Classifier* mytrain = new TL_Classifier;

	vector< float > trainingData;
	vector< int > responsesData;
	vector< float > testData;
	vector< float > testResponsesData;

	int num_for_test = 20;

	Mat train_images, train_labels;
	vector<Mat> myimages = mytrain->get_images();
	vector<int> mylabels = mytrain->get_labels();


	// Get feature
	//TL_Classifier::data_arrange(myimages, 0, num_for_test, trainingData, responsesData, testData, testResponsesData);

	cout << "Num of train samples: " << responsesData.size() << endl;
	cout << "Num of test samples: " << testResponsesData.size() << endl;

	// Merge all data 
	Mat trainingDataMat(trainingData.size() / 2, 2, CV_32FC1, &trainingData[0]);
	Mat responses(responsesData.size(), 1, CV_32SC1, &responsesData[0]);

	Mat testDataMat(testData.size() / 2, 2, CV_32FC1, &testData[0]);
	Mat testResponses(testResponsesData.size(), 1, CV_32FC1, &testResponsesData[0]);

	Ptr<cv::ml::TrainData> tdata = cv::ml::TrainData::create(trainingDataMat, cv::ml::ROW_SAMPLE, responses);

	svm = cv::ml::SVM::create();
	svm->setType(cv::ml::SVM::NU_SVC);
	//svm->setNu(0.05);
	//svm->setKernel(cv::ml::SVM::INTER);
	//svm->setDegree(1.0);
	//svm->setGamma(0.2);
	//svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	//svm->train(tdata);
	svm->trainAuto(tdata);

	if (testResponsesData.size() > 0) {
		cout << "Evaluation" << endl;
		cout << "==========" << endl;
		// Test the ML Model
		Mat testPredict;
		svm->predict(testDataMat, testPredict);
		cout << "Prediction Done" << endl;
		// Error calculation
		Mat errorMat = testPredict != testResponses;
		float error = 100.0f * countNonZero(errorMat) / testResponsesData.size();
		cout << "Error: " << error << "\%" << endl;
		// Plot training data with error label
		//plotTrainData(trainingDataMat, responses, &error);
		cout << " train !!" << endl;

	}
	else {
		//plotTrainData(trainingDataMat, responses);
		cout << " train !!_else" << endl;
	}
	return svm;
}

// data�� feature�� arrange �մϴ�.
bool TL_Classifier::data_arrange(vector<Mat> images, int label, int num_for_test,
	vector<float>& trainingData, vector<int>& responsesData,
	vector<float>& testData, vector<float>& testResponsesData)
{
	// SVM ������ ��������� �������� �ʴ� �Լ�
	int img_index = 0;
	for (int i = 0; i < images.size(); i++) {
		//// Preprocess image
		Mat pre = images[i];
		// Extract features
		vector< vector<float> > features = ExtractFeatures(pre);
		for (int i = 0; i < features.size(); i++) {
			if (img_index >= num_for_test) {
				trainingData.push_back(features[i][0]);
				trainingData.push_back(features[i][1]);
				responsesData.push_back(label);
			}
			else {
				testData.push_back(features[i][0]);
				testData.push_back(features[i][1]);
				testResponsesData.push_back((float)label);
			}
		}
		img_index++;
	}
	return true;
}
vector< vector<float> > TL_Classifier::ExtractFeatures(Mat img, vector<int>* left, vector<int>* top)
{
	// SVM ������ ��������� �������� �ʴ� �Լ�
	vector< vector<float> > output;
	vector<vector<Point> > contours;
	Mat input = img.clone();

	vector<Vec4i> hierarchy;
	findContours(input, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	// Check the number of objects detected
	if (contours.size() == 0) {
		return output;
	}
	RNG rng(0xFFFFFFFF);
	for (int i = 0; i < contours.size(); i++) {

		Mat mask = Mat::zeros(img.rows, img.cols, CV_8UC1);
		drawContours(mask, contours, i, Scalar(1), FILLED, LINE_8, hierarchy, 1);
		Scalar area_s = sum(mask);
		float area = area_s[0];

		if (area > 0) { //if the area is greather than min.

			RotatedRect r = minAreaRect(contours[i]);
			float width = r.size.width;
			float height = r.size.height;
			float ar = (width < height) ? height / width : width / height;

			vector<float> row;
			row.push_back(area);
			row.push_back(ar);
			output.push_back(row);
			if (left != NULL) {
				left->push_back((int)r.center.x);
			}
			if (top != NULL) {
				top->push_back((int)r.center.y);
			}
		}
	}
	return output;
}