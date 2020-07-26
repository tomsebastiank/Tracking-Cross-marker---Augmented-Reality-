//============================================================================
// Name        : EndoSight2.cpp

// Author      : Tom K Sebastian
// Version     : 2.0
// Description : OpenCv code for tracking a CROSS MARKER for augmented reality
//============================================================================

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d.hpp>
using namespace std;
using namespace cv;



// This function will keep track of the corners of the detected square
// Assumption is that from one frame to another distance between the tracked points is very less.
void trackingCorners(vector<Point2f> &a,Point2f  inp[4])
{



  	vector<Point2f> temp;
  	temp.push_back(Point2f(a[0].x,a[0].y));
  	temp.push_back(Point2f(a[1].x,a[1].y));
  	temp.push_back(Point2f(a[2].x,a[2].y));
  	temp.push_back(Point2f(a[3].x,a[3].y));

  	for(int i=0;i<4;i++)
  	{
  		double a0 = (inp[0].x-temp[i].x)*(inp[0].x-temp[i].x)+(inp[0].y-temp[i].y)*(inp[0].y-temp[i].y);
  		double a1 = (inp[1].x-temp[i].x)*(inp[1].x-temp[i].x)+(inp[1].y-temp[i].y)*(inp[1].y-temp[i].y);
  		double a2 = (inp[2].x-temp[i].x)*(inp[2].x-temp[i].x)+(inp[2].y-temp[i].y)*(inp[2].y-temp[i].y);
  		double a3 = (inp[3].x-temp[i].x)*(inp[3].x-temp[i].x)+(inp[3].y-temp[i].y)*(inp[3].y-temp[i].y);

  		double minVal = min(min(a0,a1),min(a2,a3));


  		if(minVal==a0)
  		{
  			a.at(0)=Point2f(inp[0].x,inp[0].y);

  		}
  		else if(minVal==a1)
  		{
  			a.at(1)=Point2f(inp[1].x,inp[1].y);

  		}
  		else if(minVal==a2)
  		{
  			a.at(2)=Point2f(inp[2].x,inp[2].y);
  		}
  		else
  		{
  			a.at(3)=Point2f(inp[3].x,inp[3].y);
  		}

  	}




}


int main(int argc,char* argv[])
{


	ifstream myfile (argv[1]);


	long double val1,val2,val3,val4,val5,val6;

	vector<Point3f> bar3d;vector<Point2f> bar2d;

	while ( myfile >> val1 >> val2 >> val3 >> val4>> val5>> val6)
	{
		bar3d.push_back(Point3f(val1, val2, val3));
	}


		//camera calibration parameter obtained from checker board images.
		Mat intrinsics = (Mat1d(3, 3) << 3316, 0, 2327, 0, 3310, 1687, 0, 0, 1);
	    Mat distortion = (Mat1d(1, 5) << .1674, -1.0781, -0.0018, -0.0026, 2.7180);


	    // markerCorners2D iteratively stores the CORNER LOCATION of the CROSS MARKER

	    vector<Point2f> markerCorners2D;
	    markerCorners2D.push_back(Point2f(0,0));
	    markerCorners2D.push_back(Point2f(0,0));
	    markerCorners2D.push_back(Point2f(0,0));
	    markerCorners2D.push_back(Point2f(0,0));





	    vector<Point3f> markerCornersreal3D;
	    markerCornersreal3D.push_back(Point3f(-2.2,  0, 0));
	    markerCornersreal3D.push_back(Point3f( 0,  2.2, 0));
	    markerCornersreal3D.push_back(Point3f( 2.2,  0, 0));
	    markerCornersreal3D.push_back(Point3f( 0, -2.2, 0));




	    // Create a VideoCapture object and open the input file
	    // If the input is the web camera, pass 0 instead of the video file name
	    VideoCapture cap(argv[2]);

	    // Check if camera opened successfully
	    if(!cap.isOpened()){
	    cout << "Error opening video stream or file" << endl;
	    return -1;
	    }

	    int frame_width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	    int frame_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);

	    // Define the codec and create VideoWriter object.The output is stored in 'outcpp.avi' file.
	    VideoWriter videoOut("outVideo.avi",CV_FOURCC('M','J','P','G'),10, Size(frame_width,frame_height));

	    int frameCount=0;
	    vector<Point2f> boxGlobal;


	    while(1){

		frameCount++;
		cout<<"Reading frame number"<<frameCount<<endl;
	    Mat frame;
	    // Capture frame-by-frame
	    cap >> frame;



	    Mat imageGray ;
	    Mat circleCopy(frame);
	    cvtColor(frame,imageGray, CV_RGB2GRAY);


	    Mat ch1, ch2, ch3;
	    // "channels" is a vector of 3 Mat arrays:
	    vector<Mat> channels(3);
	    // split img:
	    split(frame, channels);
	    // get the channels following BGR order in OpenCV
	    ch1 = channels[0];
	    ch2 = channels[1];
	    ch3 = channels[2];

	    //GaussianBlur( imageGray, imageGray, Size(3, 3), 2, 2 );
	    medianBlur( ch1, ch1, 5 );
	    //Thresholding operation
	    threshold( ch1, ch1, 180, 255,1 );




        Mat labels;
	    Mat stats;
	    Mat centroids;



	    // connectedComponentsWithStats gives the labelled output of each component

	    int n= cv::connectedComponentsWithStats(ch1, labels, stats, centroids);


	    for(int i=0; i<stats.rows; i++)
	    {
	    	int x = stats.at<int>(Point(0, i));
	        int y = stats.at<int>(Point(1, i));
	        int w = stats.at<int>(Point(2, i));
	        int h = stats.at<int>(Point(3, i));

	        Scalar color(255,0,0);
	        Rect rect(x,y,w,h);

	        vector<Point> collectedPoints;



	        int count =0;

	        for(int jj=y;jj<y+h ;jj++)
	        {
	        	for (int ii=x;ii<x+w;ii++)
	        	{
	        		int vall=(int)labels.at<int>(jj,ii);

	        	  	if(vall==i)
	        	  	{
	        	  	    count++;

	        	  	    collectedPoints.push_back(Point(ii,jj));


	        	  	}


	        	}

	        }



	          int araRect = w*h; 			//Total area of the bounded Rectangle
	          double percc = count/araRect; // Crucial parameter
	          double ratio = (double)(w/h); //Aspect ratio of box


	          // specific constrain for mining the CROSS MARKER from all other connected components  in the image.
	          //For a CROSS liked structure they will have very less ratio (total count /total area of bounded rectangle)


	          //Constraint checking
	          float areaLowerBound  = atof(argv[3]);
	          float areaUpperBound  = atof(argv[4]);
	          float ratioLowerBound = atof(argv[5]);


	          if(count>areaLowerBound && count< areaUpperBound && percc<ratioLowerBound )
	          {
	        	  RotatedRect box = minAreaRect(Mat(collectedPoints));
	        	  Point2f  vtx[4];
	        	  box.points(vtx);

	              for( int iter = 0; iter < 4; iter++ )
	                  line(frame, vtx[iter], vtx[(iter+1)%4], Scalar(0, 0, 255), 1, LINE_AA);

	              //In the starting of the video boxGlobal is empty
	              if(frameCount==1)
	              	 {
	            	  boxGlobal.push_back(Point(vtx[0].x,vtx[0].y));
	            	  boxGlobal.push_back(Point(vtx[1].x,vtx[1].y));
	            	  boxGlobal.push_back(Point(vtx[2].x,vtx[2].y));
	            	  boxGlobal.push_back(Point(vtx[3].x,vtx[3].y));

	            	  cout<<vtx[0].x<<endl;
	              	 }
	              // Else track the CORNERS of the CROSS
	              else
	              trackingCorners(boxGlobal,vtx);

	              circle(frame,Point(boxGlobal[3].x,boxGlobal[3].y),5, Scalar(0, 0, 255),2);


	          }

	    	}


	    	//Reading 2D marker point

	        markerCorners2D.at(0)= Point2f(Point2f(boxGlobal[0].x,boxGlobal[0].y));
	        markerCorners2D.at(1)= Point2f(Point2f(boxGlobal[1].x,boxGlobal[1].y));
	        markerCorners2D.at(2)= Point2f(Point2f(boxGlobal[2].x,boxGlobal[2].y));
	        markerCorners2D.at(3)= Point2f(Point2f(boxGlobal[3].x,boxGlobal[3].y));



	        Mat markerCornersreal(markerCornersreal3D);
	       	Mat markerCornersImage (markerCorners2D);


	       	Mat rvec;
	       	Mat tvec;

	       	//Finding out projection parameters rvec & tvec

	       	solvePnP(markerCornersreal, markerCornersImage, intrinsics, distortion, rvec, tvec);



	       	// projecting the point cloud of bars to the PERSEPECTIVE found from rvec & tvec

	        projectPoints(bar3d, rvec, tvec, intrinsics, distortion, bar2d);


	        for(int i=0;i<bar2d.size();i++)
	        {
	            double valx = bar2d[i].x;
	            double valy = bar2d[i].y;

	            int ii = (int) valx;
	            int jj = (int) valy;


	            if(jj<frame.rows && ii<frame.cols && jj>0 && ii>0)
	             {
	                   	frame.at<Vec3b>(jj,ii)[0] = 0;
	                   	frame.at<Vec3b>(jj,ii)[1] = 255;
	                   	frame.at<Vec3b>(jj,ii)[2] = 255;

	             }


	        }



	    // If the frame is empty, break immediately
	    if (frame.empty())
	      break;
	    videoOut.write(frame);



	    // Display the resulting frame
	    imshow( "Frame", frame );

	    // Press  ESC on keyboard to exit
	    char c=(char)waitKey(25);
	    if(c==27)
	      break;
	  }

	  // When everything done, release the video capture object
	  cap.release();
	  videoOut.release();

	  // Closes all the frames
	  destroyAllWindows();


    return 0;
}
