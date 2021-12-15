
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp> 
#include <opencv2/highgui.hpp> 
#include <opencv2/core.hpp> 
#include <opencv2/imgcodecs.hpp> 
#include <math.h>

#include <iostream>

using namespace cv;

struct turn_s{ enum {
        top   =   0
      , left   =   1
      , down     =   2
      , rirht  =   3
    };};



class bullet
{
public:
  Point pos;
  int turn;
};
std::vector<bullet> bullets;

Mat3b draw_tank(Mat3b bkg, Mat4b fgd, Point pos)
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


Mat3b draw_bullet(Mat3b bkg, Mat4b bullet_img, Point pos, int turn, int is_new)
{

  if (is_new)
  {
    bullet bul;
    bul.pos.x = pos.x+30;
    bul.pos.y = pos.y+30;
    bul.turn = turn;
    bullets.push_back(bul);
  }
  for (int j = 0; j < bullets.size(); j++)
  {
  
    Mat4b bullet_copy = bullet_img.clone();
    for (int i = 0; i < bullets[j].turn; i++)
    {
      cv::rotate(bullet_copy, bullet_copy, cv::ROTATE_90_COUNTERCLOCKWISE);
    }

    switch (bullets[j].turn)
    {
    case turn_s::top:
      bullets[j].pos.y -= 10;
      break;
    case turn_s::left:
      bullets[j].pos.x -= 10;
      break;
    case turn_s::down:
      bullets[j].pos.y += 10;
      break;
    case turn_s::rirht:
      bullets[j].pos.x += 10;
      break;
    }

    if ((bullets[j].pos.x < 0) || (bullets[j].pos.x > 820-100) || (bullets[j].pos.y < 0)|| (bullets[j].pos.y > 860-100))
    {
      bullets.erase(bullets.begin() + j);
      continue;
    }
    bkg = draw_tank(bkg, bullet_copy, bullets[j].pos);
  }
  return bkg;
}




int main( int argc, char** argv ) 
{
  Mat3b bkg = imread("../images/malinovka.webp");
  Mat4b fgd = imread("../images/tank.png", IMREAD_UNCHANGED);
  Mat4b bullet_img = imread("../images/bullet.png", IMREAD_UNCHANGED);
  cv::rotate(fgd, fgd, cv::ROTATE_90_COUNTERCLOCKWISE);
  cv::rotate(bullet_img, bullet_img, cv::ROTATE_90_COUNTERCLOCKWISE);
  resize(bkg, bkg, Size(820, 860), INTER_LINEAR);
  resize(bullet_img, bullet_img, Size(40, 40), INTER_LINEAR);
  resize(fgd, fgd, Size(100, 100), INTER_LINEAR);

  // Mat3b bkg_copy = bkg.clone();
  Point pos;
  pos.x = 0;
  pos.y = 0;
  int turn = turn_s::top; 
  while (1) {
    Mat3b bkg_copy = bkg.clone();
    int Key = waitKey(10);
    std::cout << Key << std::endl;
    if (Key == 27) break;
    if (Key == 81) {
      cv::rotate(fgd, fgd, cv::ROTATE_90_COUNTERCLOCKWISE);
      turn = (turn + 1) % 4;
    }
    if (Key == 83){
      cv::rotate(fgd, fgd, cv::ROTATE_90_CLOCKWISE);
      turn = (turn - 1) % 4;
      if (turn < 0) turn = 3;
    }
    if (Key == 82) {
      switch (turn)
      {
      case turn_s::top:
        pos.y -= 20;
        break;
      case turn_s::down:
        pos.y += 20;
        break;
      case turn_s::left:
        pos.x -= 20;
        break;
      case turn_s::rirht:
        pos.x += 20;
        break;

      } 
      
    }
    if (Key == 84){

      switch (turn)
      {
      case turn_s::top:
        pos.y += 20;
        break;
      case turn_s::down:
        pos.y -= 20;
        break;
      case turn_s::left:
        pos.x += 20;
        break;
      case turn_s::rirht:
        pos.x -= 20;
        break;

      } 
    }
    if (Key == 32){
      draw_bullet(bkg_copy, bullet_img, pos, turn, 1);
    }
    if (pos.x < 0) pos.x = 0;
    if (pos.x > 820-100) pos.x = 820-100;
    if (pos.y < 0) pos.y = 0;
    if (pos.y > 860-100) pos.y = 860-100;
    
    draw_bullet(bkg_copy, bullet_img, pos, turn, 0);
    
    Mat3b output = draw_tank(bkg_copy, fgd, pos);


    imshow("output", output);
 
  }
  
  destroyAllWindows();

  return 0;

}
