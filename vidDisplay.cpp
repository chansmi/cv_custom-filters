//
//  main.cpp
//  Display Video
//
//  Created by Chandler Smith on 1/13/23.
// cv:: is the same as turtle. - identifies package
//

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/video.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "filter.hpp"

//using namespace cv;
//using namespace std;
 
int main(int argc, char *argv[]) {
        cv::VideoCapture *capdev;

        // open the video device
        capdev = new cv::VideoCapture(0);
        if( !capdev->isOpened() ) {
                printf("Unable to open video device\n");
                return(-1);
        }

        // get some properties of the image
        cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                       (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
        printf("Expected size: %d %d\n", refS.width, refS.height);

        cv::namedWindow("Video", 1); // identifies a window
        cv::Mat frame, testFrame;
        
        int color_mode = 0;
    
        for(;;) {
            *capdev >> frame; // get a new frame from the camera, treat as a stream
            //std::cout << color_mode << std::endl;
            if( frame.empty() ) {
                printf("frame is empty\n");
                break;
            }
            
            // see if there is a waiting keystroke
            char key = cv::waitKey(10);
         
            // Define Keystrokes
            // Quit
            if( key == 'q') {
                break;
            }
            
            //Normal
            if( key == 'a') {
                color_mode = 0;
            }
            
            // Grayscale
            if( key == 'g') {
                color_mode = 1;
            }
            
            //Grayscale Custom
            if( key == 'h') {
                color_mode = 2;
            }
            
            //Grayscale Custom
            if( key == 'b') {
                color_mode = 3;
            }
            
            //sobelx
            if( key == 'x') {
                color_mode = 4;
            }
            
            //sobely
            if( key == 'y') {
                color_mode = 5;
            }
            
            //magnitude
            if( key == 'm') {
                color_mode = 6;
            }
            
            //blurQuantize
            if( key == 'l') {
                color_mode = 7;
            }
            
            //Cartoon
            if( key == 'c') {
                color_mode = 8;
            }
            
            //Rainbow Sparkle
            if( key == 'r') {
                color_mode = 9;
            }
            
            // widescreen cinema
            if( key == 'w') {
                color_mode = 10;
            }
            
            // Old film
            if( key == 'f') {
                color_mode = 11;
            }
            
            // Oil Painting cinema
            if( key == 'o') {
                color_mode = 12;
            }
            
            // laplacian
            if( key == 'j') {
                color_mode = 13;
            }
            
            // Face Detect
            if( key == 'z') {
                color_mode = 14;
            }
            
            // Adding video save for an image
            else if( key == 's') {
                bool success = cv::imwrite("/Users/chandlersmith/Desktop/CS5330/Project\ 1/ChandlerImage.jpg", frame);
                if (success) {
                    printf("save successful\n");
                }
            }
            
            /* DEFINE COLOR MODES */
            
            if(color_mode == 0) {
                cv::imshow("Video", frame);
            }
            
            // Convert Video to Grayscale
            if( color_mode == 1) {
                cv::cvtColor(frame, frame, cv::COLOR_RGB2GRAY);
                cv::imshow("Video", frame);
                //printf("grayscale successful\n");
            }
            // custom grayscale
            if( color_mode == 2) {
               // printf("color mode 2");
                grayscale( frame, frame);
                cv::imshow("Video", frame);
             }
            
            // blur
            if( color_mode == 3) {
                cv::Mat temp, output;
                blur5x5( frame, temp);
                blur5x5( temp, output); // doubled blur for increased impact
                cv::imshow("Video", output);
             }
            
            // sobelx
            if( color_mode == 4) {
                cv::Mat output;
                sobelX3x3( frame, output);
                cv::convertScaleAbs(output, frame);
                cv::imshow("Video", frame);
             }
            // sobely
            if( color_mode == 5) {
                cv::Mat output;
                sobelY3x3( frame, output);
                cv::convertScaleAbs(output, frame);
                cv::imshow("Video", frame);
             }
            
            // magintude
            if( color_mode == 6) {
                cv::Mat sx, sy, output;
                sobelX3x3(frame, sx);
                sobelY3x3(frame, sy);
                magnitude( sx, sy, output);
                cv::convertScaleAbs(output, frame);
                cv::imshow("Video", frame);
             }
            
            // blurQuantize
            if( color_mode == 7) {
                cv::Mat output;
                blurQuantize(frame, output, 15);
                cv::imshow("Video", output);
             }
            
            // cartoon
            if( color_mode == 8) {
                cv::Mat output;
                cartoon(frame, output, 15, 15);
                cv::imshow("Video", output);
             }
            // sparkle
            if( color_mode == 9) {
                cv::Mat output;
                rainbowSparkle(frame, output, 30);
                cv::imshow("Video", output);
             }
            
            // widescreen cinema
            if( color_mode == 10) {
                cv::Mat output;
                cinema(frame, output);
                cv::imshow("Video", output);
             }
            
           
            // bwFilm
            if( color_mode == 11) {
                cv::Mat output;
                
                bwFilm(frame, output);
                cv::imshow("Video", output);
             }
            
            // oil painting
            if( color_mode == 12) {
                cv::Mat output;
                oilPainting(frame, output);
                cv::imshow("Video", output);
             }
            
            // laplacian
            if( color_mode == 13) {
                cv::Mat output;
                laplacian(frame, output);
                cv::imshow("Video", output);
             }
            
            // face detect
            if( color_mode == 14) {
                cv::Mat output;
                faceDetect(frame, output);
                cv::imshow("Video", output);
             }
            
        }

        delete capdev;
        return(0);
}

