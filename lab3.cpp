#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

const int max_value_H = 360/2;
const int max_value = 255;

int th = 0;
Mat img, thimg;
int low_H = 0, low_S = 0, low_V = 0;
int high_H = max_value_H, high_S = max_value, high_V = max_value, minArea=0;

void proc_img(int,void* user_data){
	int* th_type = (int*)user_data;
	threshold(img, thimg, th, 255, *th_type);
	imshow("thimg", thimg);
	vector<vector<Point>> cnts;
	findContours(thimg, cnts, RETR_LIST, CHAIN_APPROX_NONE);
	Mat draw = Mat::zeros(thimg.size(), CV_8UC3);
	int maxAreaInd = 0;
	double maxArea = 0;
	for (size_t i = 0; i < cnts.size(); i++)
	{
		double area = contourArea(cnts[i]);
		if (area > maxArea){
			maxArea = area;
			maxAreaInd = i;
		}
	}
	Moments mnts = moments(cnts[maxAreaInd]);
	int centerX = mnts.m10 / mnts.m00;
	int centerY = mnts.m01 / mnts.m00;
	circle(draw, Point(centerX, centerY), 5, Scalar(255, 0, 0));
	drawContours(draw, cnts, maxAreaInd, Scalar(0, 0, 255));
	imshow("cntimg", draw);
}

int alibabah(int argc, char** argv){ //задание 1
    string fn;
	if (argc>1) fn= argv[1];
	else fn = "../img_zadan/allababah/ig_1.jpg";
	img = imread(fn,0);
	imshow(fn, img);

	int th_type = THRESH_BINARY;

	proc_img(0,&th_type);
	createTrackbar("th", "thimg", &th, 255, proc_img, &th_type);
	waitKey();
	return 0;
}

static void on_low_H_thresh_trackbar(int, void *)
{
    low_H = min(high_H-1, low_H);
    setTrackbarPos("Low H", "thimg", low_H);
}
static void on_high_H_thresh_trackbar(int, void *)
{
    high_H = max(high_H, low_H+1);
    setTrackbarPos("High H", "thimg", high_H);
}
static void on_low_S_thresh_trackbar(int, void *)
{
    low_S = min(high_S-1, low_S);
    setTrackbarPos("Low S", "thimg", low_S);
}
static void on_high_S_thresh_trackbar(int, void *)
{
    high_S = max(high_S, low_S+1);
    setTrackbarPos("High S", "thimg", high_S);
}
static void on_low_V_thresh_trackbar(int, void *)
{
    low_V = min(high_V-1, low_V);
    setTrackbarPos("Low V", "thimg", low_V);
}
static void on_high_V_thresh_trackbar(int, void *)
{
    high_V = max(high_V, low_V+1);
    setTrackbarPos("High V", "thimg", high_V);
}

int teplenko(int argc, char** argv){ //задание 2
    namedWindow("thimg");
	string fn;
	Mat img1, thimg;
	if (argc>1) fn= argv[1];
	else fn = "../img_zadan/teplovizor/size0-army.mil-2008-08-28-082221.jpg";
	img1 = imread(fn);
	imshow(fn, img1);

	int th_type = THRESH_BINARY;

	createTrackbar("Low H", "thimg", &low_H, max_value_H, on_low_H_thresh_trackbar); //мин знач оттенка
    createTrackbar("High H", "thimg", &high_H, max_value_H, on_high_H_thresh_trackbar); //макс знач оттенка
    createTrackbar("Low S", "thimg", &low_S, max_value, on_low_S_thresh_trackbar); //мин знач насыщенности
    createTrackbar("High S", "thimg", &high_S, max_value, on_high_S_thresh_trackbar); //макс знач насыщенности
    createTrackbar("Low V", "thimg", &low_V, max_value, on_low_V_thresh_trackbar); // мин яркость
    createTrackbar("High V", "thimg", &high_V, max_value, on_high_V_thresh_trackbar); //макс яркость
	while(true){
		Mat bgrImg, hsvImg;
		cvtColor(img1, bgrImg, COLOR_YUV2BGR);
		cvtColor(bgrImg, hsvImg, COLOR_BGR2HSV);
		inRange(hsvImg, Scalar(low_H, low_S, low_V), Scalar(high_H, high_S, high_V), thimg);
		imshow("thimg", thimg);
		vector<vector<Point>> cnts;
		findContours(thimg, cnts, RETR_LIST, CHAIN_APPROX_NONE);
		Mat draw = Mat::zeros(thimg.size(), CV_8UC3);
		for (size_t i = 0; i < cnts.size(); i++)
		{
			Moments mnts = moments(cnts[i]);
			int centerX = mnts.m10 / mnts.m00;
			int centerY = mnts.m01 / mnts.m00;
			circle(draw, Point(centerX, centerY), 2, Scalar(255, 0, 0), 2);
			drawContours(draw, cnts, i, Scalar(0, 0, 255));
		}
		imshow("cntimg", draw);
		waitKey(5);
	}
}

void find_robots(Mat &src, vector<vector<Point>> &contours, Scalar &range_min, Scalar &range_max, int minArea=0){
    Mat thimg_robots;
    inRange(src, range_min, range_max, thimg_robots);
    Mat opened, closed, dilated;
    morphologyEx(thimg_robots, opened, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(5, 5)));
    morphologyEx(opened, closed, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(7, 7)), cv::Point(-1, -1), 3);
    morphologyEx(closed, opened, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(3, 3)));
    morphologyEx(opened, closed, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(5, 5)), cv::Point(-1, -1), 4);
    morphologyEx(closed, dilated, MORPH_DILATE, getStructuringElement(MORPH_RECT, Size(3, 3)), cv::Point(-1, -1), 1);
    vector<vector<Point>> cnts;
    findContours(dilated, cnts, RETR_LIST, CHAIN_APPROX_NONE);
    for (size_t i = 0; i < cnts.size(); i++)
    {
        if (contourArea(cnts[i]) < minArea)
            continue;
        contours.push_back(cnts[i]);
    }
}

int robot_valli(int argc, char** argv){ //задание 3
	namedWindow("thimg");
	string fn;
	Mat img_robot, thimg_robot;
	if (argc>1) fn= argv[1];
	else fn = "../img_zadan/roboti/roi_robotov_1.jpg";
	img_robot = imread(fn);
    Mat bgrImg, hsvImg, gray;

    Scalar red_low_0(1, 230, 70); 
    Scalar red_high_0(50, 255, 170);
    Scalar red_low_1(90, 180, 190);
    Scalar red_high_1(180, 230, 255);

    Scalar green_low(90, 140, 0);
    Scalar green_high(128, 255, 255);
    Scalar green_low_1(142, 0, 254);
    Scalar green_high_1(150, 122, 255);

    Scalar blue_low(60, 139, 180);
    Scalar blue_high(89, 254, 255);
    Scalar blue_low_1(98, 0, 254);
    Scalar blue_high_1(157, 111, 255);
	
    Scalar blue_clr(255, 0, 0);
    Scalar green_clr(0, 255, 0);
    Scalar red_clr(0, 0, 255);
    cvtColor(img_robot, bgrImg, COLOR_YUV2BGR);
    cvtColor(bgrImg, hsvImg, COLOR_BGR2HSV);
    
    vector<vector<Point>> contours_green;
    find_robots(hsvImg, contours_green, green_low, green_high);
    find_robots(hsvImg, contours_green, green_low_1, green_high_1);
    vector<vector<Point>> contours_blue;
    find_robots(hsvImg, contours_blue, blue_low, blue_high);
    find_robots(hsvImg, contours_blue, blue_low_1, blue_high_1);
    vector<vector<Point>> contours_red;
    find_robots(hsvImg, contours_red, red_low_0, red_high_0);
    find_robots(hsvImg, contours_red, red_low_1, red_high_1);

    Mat draw = img_robot.clone();
    vector<Point> greenRobotsCenters;
    for (size_t i = 0; i < contours_green.size(); i++)
    {
        Moments mnts = moments(contours_green[i]);
        int centerX = mnts.m10 / mnts.m00;
        int centerY = mnts.m01 / mnts.m00;
        greenRobotsCenters.push_back(Point(centerX, centerY));
        drawContours(draw, contours_green, i, green_clr, 3);
    }
    vector<Point> blueRobotsCenters;
    for (size_t i = 0; i < contours_blue.size(); i++)
    {
        Moments mnts = moments(contours_blue[i]);
        int centerX = mnts.m10 / mnts.m00;
        int centerY = mnts.m01 / mnts.m00;
        blueRobotsCenters.push_back(Point(centerX, centerY));
        drawContours(draw, contours_blue, i, blue_clr, 3);
    }
    vector<Point> redRobotsCenters;
    for (size_t i = 0; i < contours_red.size(); i++)
    {
        Moments mnts = moments(contours_red[i]);
        int centerX = mnts.m10 / mnts.m00;
        int centerY = mnts.m01 / mnts.m00;
        redRobotsCenters.push_back(Point(centerX, centerY));
        drawContours(draw, contours_red, i, red_clr, 3);
    }

    gray = imread(fn, 0);
    
    threshold(gray, thimg_robot, 247, 255, THRESH_BINARY);

    vector<vector<Point>> cnts;
    findContours(thimg_robot, cnts, RETR_LIST, CHAIN_APPROX_NONE);
    double maxArea = 0;
    int lampCenterX = 0;
    int lampCenterY = 0;
    for (size_t i = 0; i < cnts.size(); i++)
    {
        if (contourArea(cnts[i]) > maxArea){
            Moments mnts = moments(cnts[i]);
            lampCenterX = mnts.m10 / mnts.m00;
            lampCenterY = mnts.m01 / mnts.m00;
        }
    }
    circle(draw, Point(lampCenterX, lampCenterY), 3, Scalar(0, 0, 0), 3);
    float minDistance = 0;

    Point closestGreen;
    for (size_t i = 0; i < greenRobotsCenters.size(); i++)
    {
        float distance = sqrt((greenRobotsCenters[i].x - lampCenterX) * (greenRobotsCenters[i].x - lampCenterX) + 
                            (greenRobotsCenters[i].y - lampCenterY) * (greenRobotsCenters[i].y - lampCenterY));
        if ((i == 0) || (distance < minDistance)){
            minDistance = distance;
            closestGreen.x = greenRobotsCenters[i].x;
            closestGreen.y = greenRobotsCenters[i].y;
        }
    }
    circle(draw, closestGreen, 3, Scalar(255, 255, 255), 3);

    minDistance = 0;
    
    Point closestBlue;
    for (size_t i = 0; i < blueRobotsCenters.size(); i++)
    {
        float distance = sqrt((blueRobotsCenters[i].x - lampCenterX) * (blueRobotsCenters[i].x - lampCenterX) + 
                            (blueRobotsCenters[i].y - lampCenterY) * (blueRobotsCenters[i].y - lampCenterY));
        if ((i == 0) || (distance < minDistance)){
            minDistance = distance;
            closestBlue.x = blueRobotsCenters[i].x;
            closestBlue.y = blueRobotsCenters[i].y;
        }
    }
    circle(draw, closestBlue, 3, Scalar(255, 255, 255), 3);

    minDistance = 0;
    Point closestRed;
    for (size_t i = 0; i < redRobotsCenters.size(); i++)
    {
        float distance = sqrt((redRobotsCenters[i].x - lampCenterX) * (redRobotsCenters[i].x - lampCenterX) + 
                            (redRobotsCenters[i].y - lampCenterY) * (redRobotsCenters[i].y - lampCenterY));
        if ((i == 0) || (distance < minDistance)){
            minDistance = distance;
            closestRed.x = redRobotsCenters[i].x;
            closestRed.y = redRobotsCenters[i].y;
        }
    }
    circle(draw, closestRed, 3, Scalar(255, 255, 255), 3);
    
    imshow("closest", draw);

    waitKey();
	return 0;
}

int bolt_gayka(int argc, char** argv){ //задание 4
	string fn;
	if (argc>1) fn= argv[1];
	else fn = "../img_zadan/gk/gk_tmplt.jpg";
    Mat templ = imread(fn, IMREAD_GRAYSCALE);
    Mat templ_th;
    threshold(templ, templ_th, 200, 255, THRESH_BINARY);
    vector<vector<Point>> cnts_template;
    findContours(templ_th, cnts_template, RETR_LIST, CHAIN_APPROX_NONE);
    Mat draw = Mat::zeros(templ.size(), CV_8UC3);
    drawContours(draw, cnts_template, 0, Scalar(255, 0, 0));

    Mat img = imread("../img_zadan/gk/gk.jpg", IMREAD_GRAYSCALE);
    Mat gray, opened, closed, dilated;
    threshold(img, gray, 230, 255, THRESH_BINARY);
    morphologyEx(gray, opened, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(3, 3)));
    morphologyEx(opened, closed, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(3, 3)));
    vector<vector<Point>> cnts;
    Mat gk = imread("../img_zadan/gk/gk.jpg");
    findContours(closed, cnts, RETR_LIST, CHAIN_APPROX_NONE);
    for (size_t i = 0; i < cnts.size(); i++)
    {
        if (contourArea(cnts[i])>100000)
            continue;
        double diff = matchShapes(cnts[i], cnts_template[0], CONTOURS_MATCH_I2, 0);
        if (diff < 1){
            drawContours(gk, cnts, i, Scalar(0, 255, 0), 5);  
        }
        else
            drawContours(gk, cnts, i, Scalar(0, 0, 255), 5);  
    }
    imshow("gk", gk);
    waitKey();
	return 0;
}

int main(int argc, char** argv){
	//alibabah(argc, argv);
    //teplenko(argc,argv);
	//robot_valli(argc,argv);
	bolt_gayka(argc,argv);
}