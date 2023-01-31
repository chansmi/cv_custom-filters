//
//  filter.hpp
//  Display Video
//
//  Created by Chandler Smith on 1/22/23.
// Header file
//

#ifndef filter_hpp
#define filter_hpp

#include <stdio.h>

int grayscale( cv::Mat &src, cv::Mat &dst );

int blur5x5( cv::Mat &src, cv::Mat &dst );

int sobelX3x3( cv::Mat &src, cv::Mat &dst );

int sobelY3x3( cv::Mat &src, cv::Mat &dst );

int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst );

int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels );

int cartoon( cv::Mat &src, cv::Mat&dst, int levels, int magThreshold );

int rainbowSparkle(cv::Mat &src, cv::Mat&dst, int magThreshold);

int cinema( cv::Mat &src, cv::Mat &dst );

int oilPainting( cv::Mat &src, cv::Mat &dst );

int bwFilm( cv::Mat &src, cv::Mat &dst );

int laplacian( cv::Mat &src, cv::Mat &dst );

int faceDetect( cv::Mat &src, cv::Mat &dst );


#endif /* filter_hpp */
