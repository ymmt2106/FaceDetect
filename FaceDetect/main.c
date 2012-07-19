//
//  main.c
//  FaceDetect
//
//  Created by 努武 山本 on 12/05/30.
//  Copyright (c) 2012年 名古屋学芸大学. All rights reserved.
//

#include <stdio.h>
#include <cv.h>
#include <highgui.h>

// #define DETECTION_FROM_FILE
#define FILENAME "group_photo_from_flickr.jpg"
// #define CASCADE_FILE "/Users/ymmt/Dropbox/dev/xcode/haarcascades/haarcascade_frontalface_alt2.xml"
#define CASCADE_FILE "/Users/ymmt/Dropbox/dev/xcode/haarcascades/haarcascade_animeface2.xml"
#define DOWNSCALE (1)
#define MIN_OBJECT_WIDTH (20)
#define MIN_OBJECT_HEIGHT (20)
#define MAX_OBJECT_WIDTH (200)
#define MAX_OBJECT_HEIGHT (200)
#define DRAW_MODE (0)

static CvHaarClassifierCascade *cascade;
static CvMemStorage *storage;

void detect_and_draw(IplImage *currentImage,
					 IplImage *resultImage,
					 int scale,
					 int min_width,
					 int min_height,
					 int max_width,
					 int max_height)
{
	static CvScalar color = { 255, 255, 255 };
	int i;
	IplImage *gray = cvCreateImage(cvSize(currentImage->width,
										  currentImage->height),
								   8, 1);
	IplImage *small_img = cvCreateImage(cvSize(cvRound(currentImage->width / scale),
											   cvRound(currentImage->height / scale)),
										8, 1);
	cvClearMemStorage(storage);
	cvCvtColor(currentImage, gray, CV_BGR2GRAY);
	cvResize(gray, small_img, CV_INTER_LINEAR);
	cvEqualizeHist(small_img, small_img);
	
	CvSeq *faces = cvHaarDetectObjects(small_img, 
									   cascade, 
									   storage, 
									   1.1, 
									   2, 
									   CV_HAAR_DO_CANNY_PRUNING, 
									   cvSize(cvRound(min_width / scale),
											  cvRound(min_height / scale)
											  ),
									   cvSize(cvRound(max_width / scale),
											  cvRound(max_height / scale)
											  )
									   );
	cvCopy(currentImage, resultImage, NULL);
	for(i=0; i<(faces ? faces->total:0); i++)
	{
		CvRect *r = (CvRect*)cvGetSeqElem(faces, i);
		if(DRAW_MODE == 0)
		{
			CvPoint center;
			int radius;
			center.x = cvRound((r->x + r->width * 0.5) * scale);
			center.y = cvRound((r->y + r->height * 0.5) *scale);
			radius = cvRound((r->width + r->height) * 0.25 * scale);
			cvCircle(resultImage, center, radius, color, 3, 8, 0);
		}else if(DRAW_MODE == 1){
			CvPoint start;
			CvPoint end;
			start.x = r->x * scale;
			start.y = r->y * scale;
			end.x = (r->x + r->width) * scale;
			end.y = (r->y + r->height) * scale;
			cvRectangle(resultImage, start, end, color, 3, 8, 0);
		}
	}
	cvReleaseImage(&gray);
	cvReleaseImage(&small_img);
}

int main(int argc, char *argv[])
{
	CvCapture *capture = NULL;
	IplImage *srcImage;
	IplImage *dstImage = NULL;
	int key;
	int use_file_flag = 0;
	
#ifdef DETECTION_FROM_FILE
	use_file_flag = 1;
#endif
	
	if(use_file_flag ==0)
	{
		if((capture = cvCreateCameraCapture(0)) == NULL)
		{
			printf("camera not found \n");
			return -1;
		}
	}else{
		srcImage = cvLoadImage(FILENAME, CV_LOAD_IMAGE_COLOR);
		dstImage = cvCloneImage(srcImage);
	}
	
	cvNamedWindow("Face Detection", CV_WINDOW_AUTOSIZE);
	cvMoveWindow("Face Detection", 50, 50);
	
	cascade = (CvHaarClassifierCascade *)cvLoad(CASCADE_FILE, 0, 0, 0);
	if(cascade == NULL)
	{
		printf("%s not found \n", CASCADE_FILE);
		return -1;
	}
	storage = cvCreateMemStorage(0);
	
	while (1) {
		if(use_file_flag ==0)
		{
			srcImage = cvQueryFrame(capture);
			if(srcImage == NULL)
				continue;
			if(dstImage == NULL)
				dstImage = cvCloneImage(srcImage);
		}
		detect_and_draw(srcImage, 
						dstImage, 
						DOWNSCALE, 
						MIN_OBJECT_WIDTH, 
						MIN_OBJECT_HEIGHT, 
						MAX_OBJECT_WIDTH, 
						MAX_OBJECT_HEIGHT
						);
		cvShowImage("Face Detection", dstImage);
		
		key = cvWaitKey(10);
		if(key == 'q')
			break;
	}
	
	if(use_file_flag == 0)
	{
		cvReleaseCapture(&capture);
	}else{
		cvReleaseImage(&srcImage);
	}
	cvReleaseImage(&dstImage);
	cvDestroyWindow("Face Detection");
	return 0;
}

