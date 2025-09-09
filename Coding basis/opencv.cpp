#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/imgproc/imgproc.hpp" 
#include<iostream>
//#include "opencv2/opencv.hpp" 
using namespace cv;
using namespace std
//origin

void RGB2Y(unsigned char *Src, unsigned char *Dest, int Width, int Height, int Stride) 
{
    for (int Y = 0; Y < Height; Y++)
    {
        unsigned char *LinePS = Src + Y * Stride;
        unsigned char *LinePD = Dest + Y * Width;
        for (int X = 0; X < Width; X++, LinePS += 3)
        {
            LinePD[X] = int(0.114 * LinePS[0] + 0.587 * LinePS[1] + 0.299 * LinePS[2]);
        }
    }
}

int main()
{
    Mat src = imread("/home/phytium/opencv_text/1.jpg");
    int Height = src.rows;
    int Width = src.cols;
    int Stride = Width*3;
    unsigned char *Src = src.data;
    unsigned char *Dest1 = new unsigned char[Height * Width];
    RGB2Y(Src, Dest1, Width, Height, Stride);
    Mat dst(Height, Width, CV_8UC1, Dest1);
    imshow("origin", src);
    imshow("result1", dst);
    waitKey(0);
    return 0
}
