#ifndef INCLUDE_IMAGE_H
#define INCLUDE_IMAGE_H


#include "utility.h"
#include <cstdint>

//pcl
#include <pcl/point_cloud.h>  
#include <pcl/point_types.h>  
#include <pcl/io/pcd_io.h> 

//opencv
#include <opencv2/opencv.hpp>   
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

namespace roadmarking
{
	class Imageprocess
	{
	  public:

		  void pcgrid(Bounds &boundingbox, float resolution);                   
		  
		  void savepcgrid(Bounds &boundingbox, float resolution, const pcXYZIPtr &c);   
		  void savepcgrid(Bounds &boundingbox, float resolution,  pcXYZIPtr &c,  pcXYZIPtr &gc, pcXYZIPtr &ngc);   
		  void pc2imgI(const pcXYZIPtr &cloud, int whatcloud,  Mat &img, float times_std);         //whatcloud[ 0: Original Cloud , 1: Ground Cloud , 2: Non-ground Cloud]; //times_std: maxI = meanI+ times_std*stdI
		  void pc2imgZ(const pcXYZIPtr &cloud, int whatcloud,  Mat &img);                          //whatcloud[ 0: Original Cloud , 1: Ground Cloud , 2: Non-ground Cloud]
		  void pc2imgD(const pcXYZIPtr &cloud, int whatcloud,  Mat &img ,float k);                 //whatcloud[ 0: Original Cloud , 1: Ground Cloud , 2: Non-ground Cloud] k:expected max point number in a pixel
		 
		  void img2pc_g(const Mat &img, const pcXYZIPtr &incloud, pcXYZIPtr & outcloud);  
		  void img2pc_c(const Mat &img, const pcXYZIPtr &incloud, pcXYZIPtr & outcloud);  

		  void img2pclabel_g(const Mat &img, const pcXYZIPtr &incloud, vector<pcXYZI> &outclouds, double dZ);   
		  void img2pclabel_c(const Mat &img, const pcXYZIPtr &incloud, vector<pcXYZI> &outclouds, double dZ); 

		  Mat Sobelboundary(Mat img0);                                                  //Sobel
		  
		  float caculateCurrentEntropy(Mat hist, int threshold);                      
		  Mat maxEntropySegMentation(Mat inputImage);                                 
		  Mat ExtractRoadPixelIZD(const Mat & _binI,const Mat & _binZ,const Mat & _binD); //(for MLS)
		  Mat ExtractRoadPixelIZ (const Mat & _binI,const Mat & _binZ);                   //(for ALS)

		  void RemoveSmallRegion(const Mat &_binImg, Mat &_binfilter, int AreaLimit);   
		  void CcaByTwoPass(const Mat & _binfilterImg, Mat & _labelImg);                
		  void CcaBySeedFill(const Mat& _binfilterImg, Mat & _lableImg);                
		  
		  void ImgReverse(const Mat &img, Mat &img_reverse);                            
		  void ImgFilling(const Mat &img, Mat &img_fill);                              

		  void LabelColor(const Mat & _labelImg, Mat & _colorImg);                     
		  Scalar GetRandomColor();                                                      

		  void DetectCornerHarris(const Mat & src, const Mat & colorlabel, Mat & cornershow, Mat & cornerwithimg, int threshold);                                //Harris
		  void DetectCornerShiTomasi(const Mat & src, const Mat & colorlabel, Mat & cornerwithimg, int minDistance, double mincorenerscore);                     //Shi-Tomasi

		  void Truncate( Mat & Img, Mat & TImg);                                   

		  //Save images
		  void saveimg(const Mat &ProjI, const Mat &ProjZ, const Mat &ProjD, const Mat &ProjImf, const Mat &GI, const Mat &GZ, const Mat &BZ, const Mat &BD, const Mat &GIR, const Mat &BI, const Mat &BIF, const Mat &Label, const Mat &Corner);
		  void saveimg(std::string base_folder, int file_index, const Mat &ProjI, const Mat &ProjZ, const Mat &ProjImf, const Mat &GI, const Mat &GZ, const Mat &BZ, const Mat &GIR, const Mat &BI, const Mat &BIF, const Mat &Label);
		  void saveimg(std::string base_folder, int file_index, const Mat &ProjI, const Mat &ProjZ, const Mat &ProjD, const Mat &ProjImf, const Mat &GI, const Mat &GZ, const Mat &BZ, const Mat &BD, const Mat &GIR, const Mat &BI, const Mat &BIF, const Mat &Label); 
		  void saveimg(const Mat &ProjI, const Mat &ProjImf, const Mat &GI, const Mat &BI, const Mat &BIF, const Mat &Label);

		  void releaseGridIndices(int whatcloud);

		  int nx = 0, ny = 0; //pixel number
		  int timin = 0, tjmin = 0; //truncated pixel no.
		  float minX, minY, minZ;  //bounding box minimum value
		  float res;  //resolution
		  int totallabel = 1;
		  vector<vector <int>>  labelx;  
		  vector<vector <int>>  labely;  


	protected:
	
	  private:
          struct PixelPointIndex
          {
              std::vector<std::uint32_t> offsets;
              std::vector<std::uint32_t> points;
          };
          PixelPointIndex point_indices_[3];
          void buildPointIndex(const pcXYZIPtr &cloud, PixelPointIndex &index);
          const PixelPointIndex &pointIndex(int whatcloud) const;
          std::size_t pixelIndex(const pcl::PointXYZI &point) const;
          void imageToCloud(const Mat &img, const pcXYZIPtr &cloud, pcXYZIPtr &out, int whatcloud);
          void labelsToCloud(const Mat &img, const pcXYZIPtr &cloud, vector<pcXYZI> &out, double dZ, int whatcloud);
		 
	};

}

#endif
