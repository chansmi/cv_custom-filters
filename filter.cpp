//
//  filter.cpp
//  Display Video
//
//  Created by Chandler Smith on 1/22/23.
//
#include <cstdio> // changed
#include <opencv2/core.hpp>
#include <opencv2/video.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xphoto.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/face.hpp>

#include "filter.hpp"

// src: input image
// dst: output image, allcated by this function

// Custom Grayscale
int grayscale( cv::Mat &src, cv::Mat &dst ) {
    // rows, columns, channels
    for(int i=0;i<src.rows;i++) {
        for(int j=0;j<src.cols;j++) { // Combined with rows to identify pixel
            int sum = 0;
            for(int c=0; c< 3; c++){
                sum += src.at<cv::Vec3b>(i,j)[c]/3;
                }
            for(int c=0; c< 3; c++){
                dst.at<cv::Vec3b>(i,j)[c] = sum;
                }
                    
            }
        }
    return 0;
}


// 5x5 Gaussian as separable 1x5 - a blur filter that incirments bell curve.
// G(sigma, mean) = 1 / sqrt (pi * sigma) e^-(x-mean)^2 / 2 sigma ^2
// Sigma controls high requency deduction (blur) - is the SD
// 5x5 14641 - create two filters and iterate over them to create the matrix

// Steps: 5 x 1, iterate over columns and rows, Sigma,
// Shout out to Gopal for assisting with troubleshooting

int blur5x5( cv::Mat &src, cv::Mat &dst ){
    // define a sigma for 2 5x1 kernals
    // for each pixel, blur by editing the pizel by kernal * sigma
    dst = src.clone();

    int sigma = 2;
    //declare temp image
    uchar kernal_h[5] = {1, 4, 6, 4, 1}; // dimension 1
    uchar kernal_v[5] = {1, 4, 6, 4, 1}; // dimension 2
    cv:: Mat3s temp(src.rows, src.cols, CV_16SC3);

    // Iterate over the rows and columns
    for(int i=2;i<src.rows - 2;i++) { // Running over vertically
        for(int j=0;j<src.cols;j++) {
            int sum_b = 0, sum_g = 0, sum_r = 0; // declare bgr values
            // Iterate over the filter kernel
            for(int c = -2; c <= 2; c++)
            {
                // Get the pixel value
                cv::Vec3b pixel = src.at<cv::Vec3b>(i + c, j); // vertically
                // Multiply the pixel value with the filter coefficient and add to the sum
                sum_b += pixel[0] * kernal_v[c + sigma];
                sum_g += pixel[1] * kernal_v[c + sigma];
                sum_r += pixel[2] * kernal_v[c + sigma];
            }
            // Set the new pixel value in the destination image
            // normalize
            sum_b /= 16;
            sum_g /= 16;
            sum_r /= 16;
            
            temp.at<cv::Vec3b>(i, j) = cv::Vec3b(sum_b, sum_g, sum_r);
        }
    }
    // Repeat for kernal h
    for(int i = 0; i < temp.rows; i++){
        for(int j = 2; j < temp.cols - 2; j++){
            short sum_b = 0, sum_g = 0, sum_r = 0;
            for(int c = -2; c <= 2; c++)
            {
                cv::Vec3b pixel = temp.at<cv::Vec3b>(i, j + c);
                sum_b += pixel[0] * kernal_h[c + sigma];
                sum_g += pixel[1] * kernal_h[c + sigma];
                sum_r += pixel[2] * kernal_h[c + sigma];
            }
            // Set the new pixel value in the destination image
            // normalize
            sum_b /= 16;
            sum_g /= 16;
            sum_r /= 16;
            dst.at<cv::Vec3b>(i, j) = cv::Vec3b(sum_b, sum_g, sum_r);
        }
    }
    return 0;
}


// Sobol filter
// 3x3 x values, first dirivitive in x direction
// pass it over every pizel to determine edge gradients
int sobelX3x3( cv::Mat &src, cv::Mat &dst ){
    // src is unsigned (0, 255)
    // need destination to be sign
    cv::Mat temp(src.rows, src.cols, CV_16SC3);
    dst = src.clone();
    dst.convertTo(dst, CV_16SC3);

    // iterating over rows
    for(int i=0;i<src.rows;i++) {
        for(int j=1;j<src.cols-1;j++) {
            cv::Vec3s &tptr = temp.at<cv::Vec3s>(i,j);
            for(int c = 0; c <= 2; c++)
            {
                tptr[c] = (1*src.at<cv::Vec3b>(i, j-1)[c] + 0*src.at<cv::Vec3b>(i, j)[c] + -1*src.at<cv::Vec3b>(i, j+1)[c]);
            }
        }
    }
    // iterating over columns
    for(int i=1;i<src.rows-1;i++) {
        for(int j=0;j<src.cols;j++){
            cv::Vec3s &dptr = dst.at<cv::Vec3s>(i,j);
            for(int c = 0; c <= 2; c++)
            {
                dptr[c] = (1*temp.at<cv::Vec3s>(i-1, j)[c] + 2*temp.at<cv::Vec3s>(i, j)[c] + 1*temp.at<cv::Vec3s>(i+1, j)[c]);
                dptr[c] = dptr[c]/4;// Normalize by 4
            }
        }
    }
    return 0;
}
// 3x3 y values, the inverse of the above function
int sobelY3x3( cv::Mat &src, cv::Mat &dst ){
    cv:: Mat3s temp(src.rows, src.cols, CV_16SC3);
    dst = src.clone();
    dst.convertTo(dst, CV_16SC3);
    // Columns
    for(int i=0;i<src.rows;i++) {
        for(int j=1;j<src.cols-1;j++){
            cv::Vec3s &tptr = temp.at<cv::Vec3s>(i,j);
            for(int c = 0; c <= 2; c++)
            {
                tptr[c] = (1*src.at<cv::Vec3b>(i, j-1)[c] + 2*src.at<cv::Vec3b>(i, j)[c] + 1*src.at<cv::Vec3b>(i, j+1)[c]);
                //tptr[c] = tptr(i,j)[c]/4;
            }
        }
    }
        // iterate over columns
    for(int i=1;i<src.rows-1;i++) {
        for(int j=0;j<src.cols;j++) {
            cv::Vec3s &dptr = dst.at<cv::Vec3s>(i,j);
            for(int c = 0; c <= 2; c++)
            {
                dptr[c] = (-1*temp.at<cv::Vec3s>(i-1, j)[c] + 0*temp.at<cv::Vec3s>(i, j)[c] + 1*temp.at<cv::Vec3s>(i+1, j)[c]);
                dptr[c] = dptr[c]/4;
            }
        }
    }
    return 0;
}

// magnitude
int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst ){
    dst = sx.clone();
    // magnitute is sqrt of sx2 + sy2
    for(int i=0;i<sx.rows;i++) { //sx is interchangable with sy
        for(int j=0;j<sx.cols;j++) {
            // x value from sx and y value from sy
            cv::Vec3s &x = sx.at<cv::Vec3s>(i, j);
            cv::Vec3s &y = sy.at<cv::Vec3s>(i, j);
            // run the Euclidean calculation for BGR so it is color
            short b = sqrt(pow(x[0], 2) + pow(y[0], 2));
            short g = sqrt(pow(x[1], 2) + pow(y[1], 2));
            short r = sqrt(pow(x[2], 2) + pow(y[2], 2));
            dst.at<cv::Vec3s>(i, j)[0] = b;
            dst.at<cv::Vec3s>(i, j)[1] = g;
            dst.at<cv::Vec3s>(i, j)[2] = r;
        }
    }
    return 0;
}

//blur quantize
int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels ){
    cv::Mat temp;
    dst = src.clone();
    blur5x5(dst, temp);
    
    int bucketSize = 255/levels;
    for(int i=0;i< temp.rows;i++) {
        for(int j=0;j< temp.cols;j++) {
            cv::Vec3b &pixel = temp.at<cv::Vec3b>(i, j);
            for(int c=0; c<=2; c++){
                int quantize = pixel[c] / bucketSize;
                pixel[c] = quantize * bucketSize;
                dst.at<cv::Vec3b>(i, j)[c] = pixel[c];
            }
        }
    }
    return 0;
}
 
// cartoon
int cartoon( cv::Mat &src, cv::Mat&dst, int levels, int magThreshold ){
    cv::Mat x, y, mag;
    dst = src.clone();
    // gradient Magnitude calculation
    sobelX3x3(src, x);
    sobelY3x3(src, y);
    magnitude(x, y, mag);
    
    // blurQuantize
    blurQuantize(src, dst, levels);
    // loop through and apply dark lines
    for(int i=0; i< dst.rows;i++) {
        for(int j=0;j< dst.cols;j++) {
            cv::Vec3b &pixel = dst.at<cv::Vec3b>(i, j);
            cv::Vec3s gradient = mag.at<cv::Vec3s>(i, j);
            for( int c=0; c<=2; c++){
                if(gradient[c] > magThreshold){
                    pixel[c] = 0;
                    }
                }
            }
        }
        return 0;
    }

// Rainbow sparkle with border
int rainbowSparkle( cv::Mat &src, cv::Mat&dst, int magThreshold ){
    cv::Mat x, y, mag;
    dst = src.clone();
    // gradient Magnitude calculation
    sobelX3x3(src, x);
    sobelY3x3(src, y);
    magnitude(x, y, mag);
    int border = 10;
    
    cv::copyMakeBorder(dst, dst, border, border, border, border, CV_HAL_BORDER_CONSTANT, std::rand() % 255);
    // loop through and apply dark lines
    for(int i=0; i< dst.rows;i++) {
        for(int j=0;j< dst.cols;j++) {
            cv::Vec3b &pixel = dst.at<cv::Vec3b>(i, j);
            cv::Vec3s gradient = mag.at<cv::Vec3s>(i, j);
            for( int c=0; c<=2; c++){
                if(gradient[c] > magThreshold){
                    pixel[c] = std::rand() % 255;
                    }
                }
            }
        }
        return 0;
    }

// Extension 1: Cinema view - widescreen
int cinema(cv::Mat &src, cv::Mat&dst){
    dst = src.clone();
    cartoon(src, dst, 15, 15);
    int border = 50;
    // Make widescreen
    cv::copyMakeBorder(dst, dst, border, border, 0, 0, CV_HAL_BORDER_CONSTANT, 0);
    
    return 0;
}

// Extension 2: Film
int bwFilm(cv::Mat &src, cv::Mat&dst){
    cv::Mat sx, sy;
    dst = src.clone();
    int border = 50;
    // Make widescreen
    cv::copyMakeBorder(dst, dst, border, border, 0, 0, CV_HAL_BORDER_CONSTANT, 0);
    grayscale(dst, dst);
    cv::resize(dst, dst, dst.size(), 0, 0, cv::INTER_NEAREST);

    return 0;
}


// Extension 3: Oil Painting
int oilPainting(cv::Mat &src, cv::Mat&dst){
    dst = src.clone();
    int border = 5;
    // Make border
    cv::copyMakeBorder(dst, dst, border, border, border, border, CV_HAL_BORDER_CONSTANT, 0);
    cv::xphoto::oilPainting(dst, dst, 10, 1, cv::COLOR_BGR2Lab);
    
    return 0;
}

// Extension 4: laplacian
int laplacian(cv::Mat &src, cv::Mat&dst){
    cv::Mat temp;
    temp = src.clone();
    int border = 5;
    // Make border
    grayscale(temp, temp);
    cv::Laplacian(temp, dst, CV_32F, 3);
    cv::convertScaleAbs(dst, dst);
    
    cv::copyMakeBorder(dst, dst, border, border, border, border, CV_HAL_BORDER_CONSTANT, 0);
    return 0;
}

// Extension 5: face detection
int faceDetect(cv::Mat &src, cv::Mat&dst){
    dst = src.clone();
    cv::cvtColor(dst, dst, cv::COLOR_BGR2GRAY);
    cv::CascadeClassifier face_cascade;
    face_cascade.load("haarcascade_frontalface_default.xml");
    //cv::CascadeClassifier eyes_cascade;
    //eyes_cascade.load("haarcascade_eye_tree_eyeglasses.xml");
    
    std::vector<cv::Rect> faces;
    face_cascade.detectMultiScale(dst, faces, 1.1, 3);
    
    //std::vector<cv::Rect> eyes;
    //eyes_cascade.detectMultiScale(dst, eyes, 1.1, 3);
    
    for(size_t i=0; i<faces.size();i++){
        cv::rectangle(dst, faces[i], cv::Scalar(0, 255, 0));
    }
    //for(int i=0; i<eyes.size();i++){
    //    cv::rectangle(dst, eyes[i], cv::Scalar(0, 0, 255));
    //}
    return 0;
}
    
