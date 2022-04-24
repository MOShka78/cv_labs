#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp> 
#include <opencv2/highgui.hpp> 
#include <opencv2/core.hpp> 
#include <opencv2/imgcodecs.hpp> 
#include <opencv2/core/mat.hpp>
#include <opencv2/core/utility.hpp>

#include <iostream>
#include <cmath>

int disparity = 100;


void unsharpMask(cv::Mat& original, cv::Mat& blurred, cv::Mat& output, double k = 1)
{
	output = original - blurred;
	output = k * output + original;
}

int subsctract(cv::Mat& img1, cv::Mat& img2, cv::Mat& result)
{
	double sumDiff = 0;
	for (int u = 0; u < img1.cols; u++)
	{
		for (int v = 0; v < img1.rows; v++)
		{
			uint8_t diff = abs(img1.at<uint8_t>(v, u) - img2.at<uint8_t>(v, u));
			sumDiff += diff;
			result.at<uint8_t>(v, u) = diff;

		}
	}
	sumDiff = sumDiff * 100 / 255;
	sumDiff = sumDiff / img1.size().area();
	return (int)sumDiff;
}


void box_filter(cv::Mat& image, int boxSize)
{
    boxSize = (boxSize - 1) / 2;
    for (int u = boxSize; u < image.cols - boxSize; u++)
    {
        for (int v = boxSize; v < image.rows - boxSize; v++)
        {
            cv::Rect roiRect(u - boxSize, v - boxSize, 2 * boxSize + 1, 2 * boxSize + 1);
            cv::Mat roi = image(roiRect);
            int sum = cv::sum(roi)[0];
            int mean = (int)(sum / (roi.rows * roi.cols));
            image.at<uint8_t>(v, u) = mean;

        }
    }
}

void task1_3(cv::Mat plane)
{
    cv::TickMeter tm; 
    cv::Mat diff(plane.rows, plane.cols, plane.type());
    cv::Mat box_output = plane.clone();
    cv::Mat blur_output = plane.clone();
    tm.start();
    box_filter(box_output, 7);  
    tm.stop();
    std::cout << "box filter: " << tm.getTimeMilli() << std::endl;
    tm.start();
    cv::blur(blur_output, blur_output, cv::Size(7,7));
    tm.stop();
    std::cout << "Blur: " << tm.getTimeMilli() << std::endl;
    disparity = subsctract(box_output, blur_output, diff);
    std::cout << "Image  similarity: " << 100 - disparity << "%" << std::endl;


    cv::imshow("plane", plane);
    cv::imshow("box_output", box_output);
    cv::imshow("blur_output", blur_output);
    cv::imshow("diff", diff);
}


void gauss_filter(cv::Mat& image, double sigma, int kernelSize)
{
	cv::Mat kernel(kernelSize, kernelSize, CV_32FC1);
	int kernelOffset = (kernelSize - 1) / 2;

    for (int u = -kernelOffset; u <= kernelOffset; u++)
	{
		for (int v = -kernelOffset; v <= kernelOffset; v++)
		{
			kernel.at<float>(v + (kernelSize - 1) / 2, u + (kernelSize - 1) / 2) = pow(M_E, -(v * v + u * u) / (2*sigma));
		}
	}
    cv::Scalar kernel_sum = cv::sum(kernel);
	kernel = kernel / kernel_sum[0];			// normalization

 	cv::filter2D(image, image, -1, kernel, cv::Point(kernelOffset, kernelOffset));

}

void logTransform(cv::Mat& img, double c)
{
	
	cv::MatIterator_<uint8_t> it, end;
	for (it = img.begin<uint8_t>(), end = img.end<uint8_t>(); it != end; ++it)
	{
		*it = (uint8_t)(c * log(*it + 1));
	}
}

void task4(cv::Mat plane)
{
    cv::Mat diff(plane.rows, plane.cols, plane.type());
	
    cv::Mat gauss_copy = plane.clone();
    cv::Mat box_copy = plane.clone();
    cv::Mat gaussblur_copy = plane.clone();
    box_filter(box_copy, 3);
    gauss_filter(gauss_copy, 1, 3);
    cv::GaussianBlur(gaussblur_copy, gaussblur_copy, cv::Size(3, 3), 0.8);
    disparity = subsctract(plane, gaussblur_copy, diff);
    std::cout << "Image  similarity: " << 100 - disparity << "%" << std::endl;
    if (disparity < 5) logTransform(diff, 30);
    cv::imshow("gauss_copy", gauss_copy);
    cv::imshow("gaussblur_copy", gaussblur_copy);
    cv::imshow("box_copy", box_copy);
    cv::imshow("diff", diff);
    cv::imshow("orig", plane);
}


void task5(cv::Mat plane)
{
    cv::Mat diff(plane.rows, plane.cols, plane.type());
    cv::Mat gauss_copy = plane.clone();
    cv::Mat box_copy = plane.clone();
    cv::Mat unsharped(plane.rows, plane.cols, plane.type());
	cv::Mat unsharped2(plane.rows, plane.cols, plane.type());
	
    gauss_filter(gauss_copy, 1.4, 7);
    unsharpMask(plane, gauss_copy, unsharped, 2);
    box_filter(box_copy, 7);
    unsharpMask(plane, box_copy, unsharped2, 2);
    disparity = subsctract(unsharped, unsharped2, diff);
    std::cout << "Image  similarity: " << 100 - disparity << "%" << std::endl;
    cv::imshow("unsharped", unsharped);
    cv::imshow("unsharped2", unsharped2);
    cv::imshow("diff", diff);
    cv::imshow("plane", plane);

}

void laplase_filter(cv::Mat plane)
{
    float kernelArray[] = {0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0};
	
    cv::Mat kernel(3, 3, CV_32FC1, kernelArray);

    cv::filter2D(plane, plane, -1, kernel, cv::Point(-1, -1));
}


void task6_7(cv::Mat plane)
{
    cv::Mat laplase_copy = plane.clone();
    cv::Mat unsharped3 = plane.clone();
    laplase_filter(laplase_copy);
    unsharped3 = unsharped3 - 1 * laplase_copy;
    cv::imshow("laplase_copy", laplase_copy);
    cv::imshow("plane", plane);
    cv::imshow("unsharped3", unsharped3);
}


int main( int argc, char** argv ) 
{
    cv::Mat plane;
    plane = imread("../images/photo_2022-02-02_16-05-54.jpg" ,cv::IMREAD_GRAYSCALE);
    cv::resize(plane, plane, cv::Size(500, 500));
    // task1_3(plane);
    // task4(plane);
    // task5(plane);
    //task6_7(plane);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}