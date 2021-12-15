
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp> 
#include <opencv2/highgui.hpp> 
#include <opencv2/core.hpp> 
#include <opencv2/imgcodecs.hpp> 

#include <iostream>
using namespace cv;


Mat3b robot(Mat3b bkg, Mat4b fgd, Point pos)
{
  double alpha = 1; // alpha in [0,1]

  Mat3b roi = bkg(Rect(pos.x, pos.y, fgd.cols, fgd.rows));

  for (int r = 0; r < roi.rows; ++r)
  {
    for (int c = 0; c < roi.cols; ++c)
    {
      const Vec4b& vf = fgd(r,c);
      if (vf[3] > 0) // alpha channel > 0
      {
        // Blending
        Vec3b& vb = roi(r,c);
        vb[0] = alpha * vf[0] + (1 - alpha) * vb[0];
        vb[1] = alpha * vf[1] + (1 - alpha) * vb[1];
        vb[2] = alpha * vf[2] + (1 - alpha) * vb[2];
      }
    }
  }

  return bkg;
}

int main( int argc, char** argv ) 
{
  
  Mat3b plane;
  plane = imread("../images/field.jpg");
  Mat4b robot_img = imread("../images/robot.png" ,cv::IMREAD_UNCHANGED);
  resize(robot_img, robot_img, Size(100, 100), INTER_LINEAR);
  
  
  if(! plane.data ) {
    std::cout <<  "Image plane not found or unable to open" << std::endl ;
    return -1;
  }
  
  Mat3b image = plane.clone();
  Point current_point = Point(0, image.rows/2 - 30);
  
  namedWindow( "image", WINDOW_AUTOSIZE );
  while (1)
  {

    current_point = Point(current_point.x + 1, (image.rows/2 - 30) + sin(((double)current_point.x + 1.0)/20)*100);
    std::cout << current_point.x << "   " << current_point.y << std::endl;
    circle(image, Point(current_point.x+40,  current_point.y+40), 3, Scalar(0,0,255), FILLED);
    Mat3b new_image = image.clone();
    robot(new_image,robot_img, current_point);
    imshow("image", new_image);
    if (cv::waitKey(8) == 'q') break;
    if (current_point.x == (image.cols / 2 - 40))
    {
      imwrite("sin_robot.png", new_image);
    }
    if (current_point.x == (image.cols - 100))
    {
      break;
    }
  }

  destroyAllWindows();
  return 0;
}