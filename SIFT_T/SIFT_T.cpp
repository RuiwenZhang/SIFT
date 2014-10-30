// SIFT_T.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

///*
//This program detects image features using SIFT keypoints. For more info,
//refer to:
//
//Lowe, D. Distinctive image features from scale-invariant keypoints.
//International Journal of Computer Vision, 60, 2 (2004), pp.91--110.
//
//Copyright (C) 2006-2010  Rob Hess <hess@eecs.oregonstate.edu>
//
//Note: The SIFT algorithm is patented in the United States and cannot be
//used in commercial products without a license from the University of
//British Columbia.  For more information, refer to the file LICENSE.ubc
//that accompanied this distribution.
//
//Version: 1.1.2-20100521
//*/
//
//#include "..\sift\sift.h"
//#include "..\sift\imgfeatures.h"
//#include "..\sift\utils.h"
//
//#include <highgui.h>
//
//#include <stdio.h>
///******************************** Globals ************************************/
//
//char* img_file_name = "..\\SIFT_T\\beaver.png";
//char* out_file_name  = "..\\SIFT_T\\beaver.sift";;
//char* out_img_name = "..\\SIFT_T\\beaver1.png";
//int display = 1;
//int intvls = SIFT_INTVLS;
//double sigma = SIFT_SIGMA;
//double contr_thr = SIFT_CONTR_THR;
//int curv_thr = SIFT_CURV_THR;
//int img_dbl = SIFT_IMG_DBL;
//int descr_width = SIFT_DESCR_WIDTH;
//int descr_hist_bins = SIFT_DESCR_HIST_BINS;
//
//
///********************************** Main *************************************/
//
//int main( int argc, char** argv )
//{
//	
//	IplImage* img;
//	struct feature* features;
//	int n = 0;
//	//CvScalar a;
//	fprintf( stderr, "Finding SIFT features...\n" );
//	img = cvLoadImage( img_file_name, 1 );
//	if( ! img )
//	{
//		fprintf( stderr, "unable to load image from %s", img_file_name );
//		exit( 1 );
//	}
//	n = _sift_features( img, &features, intvls, sigma, contr_thr, curv_thr,
//						img_dbl, descr_width, descr_hist_bins );
//	fprintf( stderr, "Found %d features.\n", n );
//
//	if( display )
//	{
//		draw_features( img, features, n );
//		cvNamedWindow( "lan", 1 );
//		cvShowImage( "lan", img );
//		cvWaitKey( 0 );
//	}
//
//	if( out_file_name != NULL )
//		export_features( out_file_name, features, n );
//	
//	if( out_img_name != NULL )
//		cvSaveImage( out_img_name, img, NULL );
//	return 0;
//}


/*
Detects SIFT features in two images and finds matches between them.

Copyright (C) 2006-2010  Rob Hess <hess@eecs.oregonstate.edu>

@version 1.1.2-20100521
*/

#include "..\\sift\\sift.h"
#include "..\\sift\\imgfeatures.h"
#include "..\\sift\\kdtree.h"
#include "..\\sift\\utils.h"
#include "..\\sift\\xform.h"

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

#include <stdio.h>


/* the maximum number of keypoint NN candidates to check during BBF search */
#define KDTREE_BBF_MAX_NN_CHKS 200

/* threshold on squared ratio of distances between NN and 2nd NN */
#define NN_SQ_DIST_RATIO_THR 0.49

/******************************** Globals ************************************/

char img1_file[] = "..\\SIFT_T\\beaver.png";
char img2_file[] = "..\\SIFT_T\\beaver_xform.png";

char img1_sfile[] = "..\\SIFT_T\\beaver_output.png";
char img2_sfile[] = "..\\SIFT_T\\beaver_xform_output.png";

/********************************** Main *************************************/

void draw_features_o(IplImage* img, struct feature* feat, int n);

int main( int argc, char** argv )
{
	IplImage* img1, * img2, * stacked;
	struct feature* feat1, * feat2, * feat;
	struct feature** nbrs;
	struct kd_node* kd_root;
	CvPoint pt1, pt2;
	double d0, d1;
	int n1, n2, k, i, m = 0;

	img1 = cvLoadImage( img1_file, 1 );
	if( ! img1 )
		fatal_error( "unable to load image from %s", img1_file );
	img2 = cvLoadImage( img2_file, 1 );
	if( ! img2 )
		fatal_error( "unable to load image from %s", img2_file );
	stacked = stack_imgs( img1, img2 );

	fprintf( stderr, "Finding features in %s...\n", img1_file );
	n1 = sift_features( img1, &feat1 );
	fprintf( stderr, "Finding features in %s...\n", img2_file );
	n2 = sift_features( img2, &feat2 );

	kd_root = kdtree_build( feat2, n2 );//½¨Á¢feat2µÄkd tree
	for( i = 0; i < n1; i++ )
	{
		feat = feat1 + i;
		k = kdtree_bbf_knn( kd_root, feat, 2, &nbrs, KDTREE_BBF_MAX_NN_CHKS );
		if( k == 2 )
		{
			d0 = descr_dist_sq( feat, nbrs[0] );
			d1 = descr_dist_sq( feat, nbrs[1] );
			if( d0 < d1 * NN_SQ_DIST_RATIO_THR )
			{
				pt1 = cvPoint( cvRound( feat->x ), cvRound( feat->y ) );//feat1
				pt2 = cvPoint( cvRound( nbrs[0]->x ), cvRound( nbrs[0]->y ) );//feat2
				pt2.y += img1->height;
				cvLine( stacked, pt1, pt2, CV_RGB(255,0,255), 1, 8, 0 );
				m++;
				feat1[i].fwd_match = nbrs[0];
			}
		}
		free( nbrs );
	}

	fprintf( stderr, "Found %d total matches\n", m );
	cvNamedWindow( "Matches", 1 );
	cvShowImage( "Matches", stacked );
	cvWaitKey( 0 );

	draw_features_o( img1,feat1,n1);
	draw_features_o( img2,feat2,n2);
	cvSaveImage(img1_sfile,img1);
	cvSaveImage(img2_sfile,img2);
	cvShowImage("img1",img1);
	cvShowImage("img2",img2);
	cvWaitKey(0);

	/* 
	UNCOMMENT BELOW TO SEE HOW RANSAC FUNCTION WORKS

	Note that this line above:

	feat1[i].fwd_match = nbrs[0];

	is important for the RANSAC function to work.
	*/
	
	/*
	{
		CvMat* H;
		H = ransac_xform( feat1, n1, FEATURE_FWD_MATCH, lsq_homog, 4, 0.01,
			homog_xfer_err, 3.0, NULL, NULL );
		if( H )
		{
			IplImage* xformed;
			xformed = cvCreateImage( cvGetSize( img2 ), IPL_DEPTH_8U, 3 );
			cvWarpPerspective( img1, xformed, H, 
				CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS,
				cvScalarAll( 0 ) );
			cvNamedWindow( "Xformed", 1 );
			cvShowImage( "Xformed", xformed );
			cvWaitKey( 0 );
			cvReleaseImage( &xformed );
			cvReleaseMat( &H );
		}
	}
	*/


	cvReleaseImage( &stacked );
	cvReleaseImage( &img1 );
	cvReleaseImage( &img2 );
	kdtree_release( kd_root );
	free( feat1 );
	free( feat2 );
	return 0;
}



void draw_features_o(IplImage* img, struct feature* feat, int n)
{
	for(int i=0;i<n;i++)
	{
		cvCircle(img,cvPoint(feat[i].x,feat[i].y),2,cvScalar(255,0,0),1);
	}
}