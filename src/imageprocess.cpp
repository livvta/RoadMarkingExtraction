#include "imageprocess.h"
#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <boost/filesystem.hpp>

using namespace std;
using namespace cv;

namespace roadmarking
{

    void Imageprocess::pcgrid(Bounds &boundingbox, float resolution)
    {
        const float lx = boundingbox.max_x - boundingbox.min_x;
        const float ly = boundingbox.max_y - boundingbox.min_y;
        const float lz = boundingbox.max_z - boundingbox.min_z;
        if (!std::isfinite(resolution) || resolution <= 0 ||
            !std::isfinite(lx) || !std::isfinite(ly) || lx < 0 || ly < 0 ||
            static_cast<double>(lx) / resolution >= INT_MAX ||
            static_cast<double>(ly) / resolution >= INT_MAX)
            throw std::invalid_argument("Invalid projection bounds or grid resolution");
        minX = -0.5f * lx;
        minY = -0.5f * ly;
        minZ = -0.5f * lz;
        nx = lx / resolution + 1;
        ny = ly / resolution + 1;
        res = resolution;
        const std::size_t pixels = static_cast<std::size_t>(nx) * ny;
        if (pixels >= std::numeric_limits<std::uint32_t>::max())
            throw std::length_error("Projection grid exceeds the compact index limit; increase resolution");
        for (int which = 0; which < 3; ++which) releaseGridIndices(which);
        cout << "Image Size: " << nx << " x " << ny << endl;
    }

    std::size_t Imageprocess::pixelIndex(const pcl::PointXYZI &point) const
    {
        const float x = (point.x - minX) / res;
        const float y = (point.y - minY) / res;
        // Small coordinate rounding at the grid boundary is handled without
        // permitting non-finite or genuinely out-of-bounds coordinates.
        if (!std::isfinite(x) || !std::isfinite(y) || x < -0.001f || y < -0.001f ||
            x >= nx || y >= ny)
            throw std::out_of_range("Point lies outside the projection grid");
        return static_cast<std::size_t>(std::max(0, static_cast<int>(x))) * ny +
               std::max(0, static_cast<int>(y));
    }

    void Imageprocess::buildPointIndex(const pcXYZIPtr &cloud, PixelPointIndex &index)
    {
        const std::size_t pixels = static_cast<std::size_t>(nx) * ny;
        if (cloud->size() >= std::numeric_limits<std::uint32_t>::max())
            throw std::length_error("Point cloud exceeds the compact index limit");
        index.offsets.assign(pixels + 1, 0);
        index.points.resize(cloud->size());
        for (const auto &point : *cloud) ++index.offsets[pixelIndex(point) + 1];
        for (std::size_t pixel = 1; pixel <= pixels; ++pixel)
            index.offsets[pixel] += index.offsets[pixel - 1];
        // Reuse the offsets as insertion cursors and restore them afterwards.
        // Stable insertion retains the previous per-pixel point order.
        for (std::size_t point = 0; point < cloud->size(); ++point)
            index.points[index.offsets[pixelIndex((*cloud)[point])]++] = point;
        for (std::size_t pixel = pixels; pixel > 0; --pixel)
            index.offsets[pixel] = index.offsets[pixel - 1];
        index.offsets[0] = 0;
    }

    const Imageprocess::PixelPointIndex &Imageprocess::pointIndex(int whatcloud) const
    {
        if (whatcloud < 0 || whatcloud > 2 || point_indices_[whatcloud].offsets.empty())
            throw std::invalid_argument("Requested point index has not been built or was released");
        return point_indices_[whatcloud];
    }

    void Imageprocess::releaseGridIndices(int whatcloud)
    {
        if (whatcloud < 0 || whatcloud > 2) throw std::invalid_argument("Invalid cloud type");
        PixelPointIndex empty;
        point_indices_[whatcloud].offsets.swap(empty.offsets);
        point_indices_[whatcloud].points.swap(empty.points);
    }

    void Imageprocess::savepcgrid(Bounds &boundingbox, float resolution,
                                  pcXYZIPtr &cloud, pcXYZIPtr &ground, pcXYZIPtr &non_ground)
    {
        pcgrid(boundingbox, resolution);
        // The main pipeline never projects or retrieves original-cloud points.
        cloud.reset(new pcXYZI());
        buildPointIndex(ground, point_indices_[1]);
        buildPointIndex(non_ground, point_indices_[2]);
    }

    void Imageprocess::savepcgrid(Bounds &boundingbox, float resolution, const pcXYZIPtr &cloud)
    {
        pcgrid(boundingbox, resolution);
        buildPointIndex(cloud, point_indices_[0]);
    }

    void Imageprocess::pc2imgI(const pcXYZIPtr &cloud, int whatcloud, Mat &img, float times_std)
    {
        const auto &index = pointIndex(whatcloud);
        const std::size_t pixels = static_cast<std::size_t>(nx) * ny;
        std::vector<float> averages(pixels);
        double mean = 0, variance = 0;
        std::size_t non_zero_pixels = 0;
        for (std::size_t pixel = 0; pixel < pixels; ++pixel)
        {
            double sum = 0;
            const auto begin = index.offsets[pixel], end = index.offsets[pixel + 1];
            for (auto k = begin; k < end; ++k) sum += (*cloud)[index.points[k]].intensity;
            // Preserve the original zero seed and floating-point averaging.
            averages[pixel] = (0.001 + sum) / (static_cast<double>(end - begin) + 1);
            if (averages[pixel] > 1)
            {
                mean += averages[pixel];
                ++non_zero_pixels;
            }
        }
        img = Mat::zeros(nx, ny, CV_8UC1);
        if (non_zero_pixels == 0) return;
        mean /= non_zero_pixels;
        for (float value : averages)
            if (value > 1) variance += (value - mean) * (value - mean);
        const double maximum = mean + times_std * std::sqrt(variance / non_zero_pixels);
        if (!std::isfinite(maximum) || maximum <= 0) return;
        for (int row = 0; row < nx; ++row)
        {
            auto *output = img.ptr<uchar>(row);
            for (int col = 0; col < ny; ++col)
            {
                const std::size_t pixel = static_cast<std::size_t>(row) * ny + col;
                if (averages[pixel] > maximum) averages[pixel] = maximum;
                output[col] = 255 * averages[pixel] / maximum;
            }
        }
    }

    void Imageprocess::pc2imgZ(const pcXYZIPtr &cloud, int whatcloud, Mat &img)
    {
        const auto &index = pointIndex(whatcloud);
        const std::size_t pixels = static_cast<std::size_t>(nx) * ny;
        std::vector<float> averages(pixels);
        float minimum = FLT_MAX, maximum = -FLT_MAX;
        for (std::size_t pixel = 0; pixel < pixels; ++pixel)
        {
            double sum = minZ;
            const auto begin = index.offsets[pixel], end = index.offsets[pixel + 1];
            for (auto k = begin; k < end; ++k) sum += (*cloud)[index.points[k]].z;
            averages[pixel] = (0.1 + sum) / (static_cast<double>(end - begin) + 1);
            minimum = std::min(minimum, averages[pixel]);
            maximum = std::max(maximum, averages[pixel]);
        }
        img = Mat::zeros(nx, ny, CV_8UC1);
        if (maximum <= minimum) return;
        for (int row = 0; row < nx; ++row)
        {
            auto *output = img.ptr<uchar>(row);
            for (int col = 0; col < ny; ++col)
                output[col] = 255 * (averages[static_cast<std::size_t>(row) * ny + col] - minimum) /
                              (maximum - minimum);
        }
    }

    void Imageprocess::pc2imgD(const pcXYZIPtr &cloud, int whatcloud, Mat &img, float expected_max)
    {
        if (!std::isfinite(expected_max) || expected_max <= 0)
            throw std::invalid_argument("Expected point count per pixel must be positive");
        const auto &index = pointIndex(whatcloud);
        img.create(nx, ny, CV_8UC1);
        // Preserve the historical integer cap, but keep empty pixels black
        // when fine grids imply fewer than one expected point per pixel.
        const double cap = std::max(1.0, std::floor(static_cast<double>(expected_max)));
        for (int row = 0; row < nx; ++row)
        {
            auto *output = img.ptr<uchar>(row);
            for (int col = 0; col < ny; ++col)
            {
                const std::size_t pixel = static_cast<std::size_t>(row) * ny + col;
                const auto count = index.offsets[pixel + 1] - index.offsets[pixel];
                output[col] = count < cap ? 255 * count / cap : 255;
            }
        }
    }

    void Imageprocess::imageToCloud(const Mat &img, const pcXYZIPtr &cloud,
                                    pcXYZIPtr &out, int whatcloud)
    {
        const auto &index = pointIndex(whatcloud);
        Mat gray, binary;
        cvtColor(img, gray, COLOR_BGR2GRAY);
        threshold(gray, binary, 0, 1, cv::THRESH_BINARY);
        for (int row = 0; row < img.rows; ++row)
            for (int col = 0; col < img.cols; ++col)
                if (binary.at<uchar>(row, col))
                {
                    const std::size_t pixel = static_cast<std::size_t>(row + timin) * ny + col + tjmin;
                    for (auto k = index.offsets.at(pixel); k < index.offsets.at(pixel + 1); ++k)
                        out->points.push_back((*cloud)[index.points[k]]);
                }
    }

    void Imageprocess::img2pc_g(const Mat &img, const pcXYZIPtr &cloud, pcXYZIPtr &out)
    {
        imageToCloud(img, cloud, out, 1);
    }

    void Imageprocess::img2pc_c(const Mat &img, const pcXYZIPtr &cloud, pcXYZIPtr &out)
    {
        imageToCloud(img, cloud, out, 0);
    }

    void Imageprocess::labelsToCloud(const Mat &img, const pcXYZIPtr &cloud,
                                     vector<pcXYZI> &out, double dZ, int whatcloud)
    {
        const auto &index = pointIndex(whatcloud);
        out.clear();
        out.resize(std::max(0, totallabel - 1));
        for (int row = 0; row < img.rows; ++row)
        {
            const int *labels = img.ptr<int>(row);
            for (int col = 0; col < img.cols; ++col)
            {
                const int label = labels[col];
                if (label <= 1) continue;
                const std::size_t pixel = static_cast<std::size_t>(row + timin) * ny + col + tjmin;
                const auto begin = index.offsets.at(pixel), end = index.offsets.at(pixel + 1);
                if (begin == end) continue; // Filled pixels need not contain points.
                double minimum = DBL_MAX, maximum = -DBL_MAX;
                for (auto k = begin; k < end; ++k)
                {
                    const auto &point = (*cloud)[index.points[k]];
                    minimum = std::min(minimum, static_cast<double>(point.z));
                    maximum = std::max(maximum, static_cast<double>(point.z));
                }
                auto &candidate = out.at(label - 2);
                if (maximum - minimum < dZ)
                    for (auto k = begin; k < end; ++k)
                        candidate.points.push_back((*cloud)[index.points[k]]);
                else if (candidate.empty())
                    candidate.points.push_back((*cloud)[index.points[begin]]);
            }
        }
    }

    void Imageprocess::img2pclabel_g(const Mat &img, const pcXYZIPtr &cloud,
                                     vector<pcXYZI> &out, double dZ)
    {
        timin = tjmin = 0;
        labelsToCloud(img, cloud, out, dZ, 1);
    }

    void Imageprocess::img2pclabel_c(const Mat &img, const pcXYZIPtr &cloud,
                                     vector<pcXYZI> &out, double dZ)
    {
        labelsToCloud(img, cloud, out, dZ, 0);
    }

	Mat Imageprocess::Sobelboundary(Mat img0)
	{
		//Using Sobel Operation
		Mat grad_xg, grad_yg, abs_grad_xg, abs_grad_yg, dstg;

		Sobel(img0, grad_xg, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);
		convertScaleAbs(grad_xg, abs_grad_xg);

		Sobel(img0, grad_yg, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT);
		convertScaleAbs(grad_yg, abs_grad_yg);

		addWeighted(abs_grad_xg, 0.5, abs_grad_yg, 0.5, 0, dstg);
		return dstg;
	}

	float Imageprocess::caculateCurrentEntropy(Mat hist, int threshold)
	{
		float BackgroundSum = 0, targetSum = 0;
		const float *pDataHist = (float *)hist.ptr<float>(0);
		for (int i = 0; i < 256; i++)
		{
			if (i < threshold)
				BackgroundSum += pDataHist[i];
			else
				targetSum += pDataHist[i];
		}
		float BackgroundEntropy = 0, targetEntropy = 0;
		for (int i = 0; i < 256; i++)
		{
			if (i < threshold)
			{
				if (pDataHist[i] == 0)
					continue;
				float ratio1 = pDataHist[i] / BackgroundSum;
				BackgroundEntropy += -ratio1 * logf(ratio1);
			}
			else
			{
				if (pDataHist[i] == 0)
					continue;
				float ratio2 = pDataHist[i] / targetSum;
				targetEntropy += -ratio2 * logf(ratio2);
			}
		}
		return (targetEntropy + BackgroundEntropy); 
	}

	Mat Imageprocess::maxEntropySegMentation(Mat inputImage)
	{
		// Max Entropy Binarization
		// Using the distribution of histogram to calculate the threshold leading to the max entropy.
		const int channels[1] = {0};
		const int histSize[1] = {256};
		float pranges[2] = {0, 256};
		const float *ranges[1] = {pranges};
		MatND hist;
		calcHist(&inputImage, 1, channels, Mat(), hist, 1, histSize, ranges);
		float maxentropy = 0;
		int max_index = 0;
		Mat result;
		for (int i = 0; i < 256; i++) 
		{
			float cur_entropy = caculateCurrentEntropy(hist, i);
			if (cur_entropy > maxentropy)
			{
				maxentropy = cur_entropy;
				max_index = i;
			}
		}
		threshold(inputImage, result, max_index, 1, cv::THRESH_BINARY); // > max_index assign as 1   < max_index assign as 0
		return result;
	}

	Mat Imageprocess::ExtractRoadPixelIZD(const Mat &_imgI, const Mat &_binZ, const Mat &_binD)
	{
		Mat result;
		_imgI.convertTo(result, CV_8UC1);
		//int mini, minj, maxi, maxj;
		//vector <int> arrayi, arrayj;
		for (int i = 0; i < _imgI.rows; i++)
		{
			for (int j = 0; j < _imgI.cols; j++)
			{
				if (_binZ.at<uchar>(i, j) == 1 || _binD.at<uchar>(i, j) == 0)
				{
					result.at<uchar>(i, j) = 0;
				}
				else
				{
					result.at<uchar>(i, j) = _imgI.at<uchar>(i, j);
				}
			}
		}
		return result;
	}
	Mat Imageprocess::ExtractRoadPixelIZ(const Mat &_imgI, const Mat &_binZ)
	{
		Mat result;
		_imgI.convertTo(result, CV_8UC1);
		for (int i = 0; i < _imgI.rows; i++)
		{
			for (int j = 0; j < _imgI.cols; j++)
			{
				if (_binZ.at<uchar>(i, j) == 1)
				{
					result.at<uchar>(i, j) = 0;
				}
				else
				{
					result.at<uchar>(i, j) = _imgI.at<uchar>(i, j);
				}
			}
		}
		return result;
	}
	void Imageprocess::CcaByTwoPass(const Mat &_binImg, Mat &_labelImg)
	{
		// connected component analysis (8-component)
		// use two-pass algorithm 
		// 1. first pass: label each foreground pixel with a label
		// 2. second pass: visit each labeled pixel and merge neighbor labels
		//
		// foreground pixel: _binImg(x,y) = 1
		// background pixel: _binImg(x,y) = 0

		if (_binImg.empty() ||
			_binImg.type() != CV_8UC1)
		{
			return;
		}

		// 1. first pass

		_labelImg.release();
		_binImg.convertTo(_labelImg, CV_32SC1); // _labelImg -> _binImg  32 Signed 

		int label = 1;		   // start by 2
		vector<int> labelSet;  
		labelSet.push_back(0); // background: 0
		labelSet.push_back(1); // foreground: 1

		int rows = _binImg.rows - 1;
		int cols = _binImg.cols - 1;
		for (int i = 1; i < rows; i++) 
		{
			int *data_preRow = _labelImg.ptr<int>(i - 1); 
			int *data_curRow = _labelImg.ptr<int>(i);	  
			for (int j = 1; j < cols; j++)				  
			{
				if (data_curRow[j] == 1) 
				{
					vector<int> neighborLabels;			
					neighborLabels.reserve(2);			
					int leftPixel = data_curRow[j - 1]; 
					int upPixel = data_preRow[j];		
					//int leftupPixel = data_preRow[j - 1];                 

					if (leftPixel > 1) //
					{
						neighborLabels.push_back(leftPixel); 
					}
					if (upPixel > 1) 
					{
						neighborLabels.push_back(upPixel); 
					}

					if (neighborLabels.empty()) //
					{
						labelSet.push_back(++label); // assign to a new label
						data_curRow[j] = label;	
						labelSet[label] = label;
					}
					else
					{
						sort(neighborLabels.begin(), neighborLabels.end());
						int smallestLabel = neighborLabels[0];
						data_curRow[j] = smallestLabel; 

						// save equivalence
						for (size_t k = 1; k < neighborLabels.size(); k++) 
						{
							int tempLabel = neighborLabels[k]; //
							int &oldSmallestLabel = labelSet[tempLabel];
							if (oldSmallestLabel > smallestLabel)
							{
								labelSet[oldSmallestLabel] = smallestLabel;
								oldSmallestLabel = smallestLabel;
							}
							else if (oldSmallestLabel < smallestLabel)
							{
								labelSet[smallestLabel] = oldSmallestLabel;
							}
						}
					}
				}
			}
		}

		// update equivalent labels
		// assigned with the smallest label in each equivalent label set
		for (size_t i = 2; i < labelSet.size(); i++) 
		{
			int curLabel = labelSet[i];
			int preLabel = labelSet[curLabel];
			while (preLabel != curLabel)
			{
				curLabel = preLabel;
				preLabel = labelSet[preLabel];
			}
			labelSet[i] = curLabel;
		}

		// 2. second pass
		for (int i = 0; i < rows; i++)
		{
			int *data = _labelImg.ptr<int>(i);
			for (int j = 0; j < cols; j++)
			{
				int &pixelLabel = data[j];
				pixelLabel = labelSet[pixelLabel];
			}
		}
		totallabel = label;
		//cout << "Number label: " << totallabel << endl;
	}

    void Imageprocess::CcaBySeedFill(const Mat &binary, Mat &labels)
    {
        if (binary.empty() || binary.type() != CV_8UC1)
            throw std::invalid_argument("Connected components require a binary 8-bit image");
        binary.convertTo(labels, CV_32SC1);
        int label = 1;
        std::vector<std::pair<int, int>> pending;
        for (int row = 1; row < binary.rows - 1; ++row)
            for (int col = 1; col < binary.cols - 1; ++col)
            {
                if (labels.at<int>(row, col) != 1) continue;
                ++label;
                labels.at<int>(row, col) = label;
                pending.emplace_back(row, col);
                while (!pending.empty())
                {
                    const auto pixel = pending.back();
                    pending.pop_back();
                    for (int dr = -1; dr <= 1; ++dr)
                        for (int dc = -1; dc <= 1; ++dc)
                        {
                            const int r = pixel.first + dr, c = pixel.second + dc;
                            if (r < 0 || r >= labels.rows || c < 0 || c >= labels.cols) continue;
                            int &value = labels.at<int>(r, c);
                            if (value != 1) continue;
                            // Mark on insertion so a pixel is never queued repeatedly.
                            value = label;
                            pending.emplace_back(r, c);
                        }
                }
            }
        totallabel = label;
    }

	void Imageprocess::RemoveSmallRegion(const Mat &Src, Mat &Dst, int AreaLimit)
	{
		int RemoveCount = 0;
		Src.convertTo(Dst, CV_8UC1);
		int CheckMode = 1;	 
		int NeihborMode = 1; 
		Mat PointLabel = Mat::zeros(Src.size(), CV_8UC1);
		if (CheckMode == 1) 
		{
			for (int i = 0; i < Src.rows; i++)
			{
				for (int j = 0; j < Src.cols; j++)
				{
					if (Src.at<uchar>(i, j) < 1)
						PointLabel.at<uchar>(i, j) = 3; 
				}
			}
		}
		else 
		{
			for (int i = 0; i < Src.rows; i++)
			{
				for (int j = 0; j < Src.cols; j++)
				{
					if (Src.at<uchar>(i, j) > 10)
						PointLabel.at<uchar>(i, j) = 3; 
				}
			}
		}

		vector<Point2i> NeihborPos;
		NeihborPos.push_back(Point2i(-1, 0));
		NeihborPos.push_back(Point2i(1, 0));
		NeihborPos.push_back(Point2i(0, -1));
		NeihborPos.push_back(Point2i(0, 1));
		if (NeihborMode == 1)
		{
			NeihborPos.push_back(Point2i(-1, -1));
			NeihborPos.push_back(Point2i(-1, 1));
			NeihborPos.push_back(Point2i(1, -1));
			NeihborPos.push_back(Point2i(1, 1));
		}
		else
			int a = 0; 
		int NeihborCount = 4 + 4 * NeihborMode;
		int CurrX = 0, CurrY = 0;
		for (int i = 0; i < Src.rows; i++)
		{
			for (int j = 0; j < Src.cols; j++)
			{
				if (PointLabel.at<uchar>(i, j) == 0) 
				{									
					vector<Point2i> GrowBuffer;		 
					GrowBuffer.push_back(Point2i(j, i));
					PointLabel.at<uchar>(i, j) = 1;
					int CheckResult = 0;

					for (int z = 0; z < GrowBuffer.size(); z++)
					{
						for (int q = 0; q < NeihborCount; q++)
						{
							CurrX = GrowBuffer.at(z).x + NeihborPos.at(q).x;
							CurrY = GrowBuffer.at(z).y + NeihborPos.at(q).y;
							if (CurrX >= 0 && CurrX < Src.cols && CurrY >= 0 && CurrY < Src.rows) 
							{
								if (PointLabel.at<uchar>(CurrY, CurrX) == 0)
								{
									GrowBuffer.push_back(Point2i(CurrX, CurrY)); 
									PointLabel.at<uchar>(CurrY, CurrX) = 1;		
								}
							}
						}
					}
					if (GrowBuffer.size() > AreaLimit)
						CheckResult = 2;
					else
					{
						CheckResult = 1;
						RemoveCount++; 
					}
					for (int z = 0; z < GrowBuffer.size(); z++)
					{
						CurrX = GrowBuffer.at(z).x;
						CurrY = GrowBuffer.at(z).y;
						PointLabel.at<uchar>(CurrY, CurrX) += CheckResult; 
					}
				}
			}
		}
		CheckMode = 255 * (1 - CheckMode);
		for (int i = 0; i < Src.rows; ++i)
		{
			for (int j = 0; j < Src.cols; ++j)
			{
				if (PointLabel.at<uchar>(i, j) == 2)
				{
					Dst.at<uchar>(i, j) = 0;
				}
				else
				{
					Dst.at<uchar>(i, j) = Src.at<uchar>(i, j);
				}
			}
		}
	}

	Scalar Imageprocess::GetRandomColor()
	{
		uchar r = 255 * (rand() / (1.0 + RAND_MAX)); // rand() / (1.0+ RAND_MAX) : a random float number between 0 and 1 (can't be equal to 1)
		uchar g = 255 * (rand() / (1.0 + RAND_MAX)); 
		uchar b = 255 * (rand() / (1.0 + RAND_MAX));
		return Scalar(b, g, r);
	}

	void Imageprocess::LabelColor(const Mat &_labelImg, Mat &_colorLabelImg)
	{
		if (_labelImg.empty() ||
			_labelImg.type() != CV_32SC1)
		{
			return;
		}

		std::map<int, Scalar> colors;

		int rows = _labelImg.rows;
		int cols = _labelImg.cols;

		_colorLabelImg.release();
		_colorLabelImg.create(rows, cols, CV_8UC3);
		_colorLabelImg = Scalar::all(0);

		for (int i = 0; i < rows; i++)
		{
			const int *data_src = (int *)_labelImg.ptr<int>(i);
			uchar *data_dst = _colorLabelImg.ptr<uchar>(i);
			for (int j = 0; j < cols; j++)
			{
				int pixelValue = data_src[j];
				if (pixelValue > 1)
				{
					//if(j%100==0) cout << pixelValue << endl;
					//labelx[pixelValue - 2].push_back(i);
					//labely[pixelValue - 2].push_back(j);

					if (colors.count(pixelValue) <= 0)
					{
						colors[pixelValue] = GetRandomColor();
					}
					Scalar color = colors[pixelValue];
					*data_dst++ = color[0];
					*data_dst++ = color[1];
					*data_dst++ = color[2];
				}
				else
				{
					data_dst++;
					data_dst++;
					data_dst++;
				}
			}
		}
	}
	void Imageprocess::Truncate(Mat &Img, Mat &TruncatedImg)
	{
		int mini, minj, maxi, maxj, di, dj;
		mini = INT_MAX;
		minj = INT_MAX;
		maxi = 0;
		maxj = 0;

		for (int i = 0; i < Img.rows; i++)
		{
			Img.at<uchar>(i, 0) = 0;
			Img.at<uchar>(i, Img.cols - 1) = 0;
		}
		for (int j = 0; j < Img.cols; j++)
		{
			Img.at<uchar>(0, j) = 0;
			Img.at<uchar>(Img.rows - 1, j) = 0;
		}

		for (int i = 0; i < Img.rows; i++)
		{
			for (int j = 0; j < Img.cols; j++)
			{
				if (Img.at<uchar>(i, j) != 0)
				{
					if (i < mini)
						mini = i;
					if (i > maxi)
						maxi = i;
					if (j < minj)
						minj = j;
					if (j > maxj)
						maxj = j;
				}
			}
		}

		timin = mini - 1;
		tjmin = minj - 1;
		di = maxi - mini + 3;
		dj = maxj - minj + 3;
		cv::Rect rect(tjmin, timin, dj, di);
		TruncatedImg = Img(rect);

		/*TruncatedImg.create(di, dj, CV_8UC1);
	for (int i = 0; i < di; i++)
	{
		for (int j = 0; j < dj; j++)
		{
			TruncatedImg.at<uchar>(i, j) = Img.at<uchar>(timin + i, tjmin + j);
		}
	}*/

		//cout << "X: " << tjmin << "  Y: " << timin << "  dX: " << dj << "  dY: " << di<< endl;

		//imshow("Truncated", 255*TruncatedImg);
	}

	void Imageprocess::DetectCornerHarris(const Mat &src, const Mat &colorlabel, Mat &cornershow, Mat &cornerwithimg, int threshold)
	{
		Mat corner, corner8u, imageGray;
		src.convertTo(imageGray, CV_8UC1);
		corner = Mat::zeros(src.size(), CV_32FC1);
		cornerHarris(imageGray, corner, 3, 3, 0.04, BORDER_DEFAULT);
		normalize(corner, corner8u, 0, 255, cv::NORM_MINMAX);
		convertScaleAbs(corner8u, cornershow);
		cornerwithimg = colorlabel.clone();
		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				if (cornershow.at<uchar>(i, j) > threshold)
				{
					circle(cornerwithimg, Point(j, i), 2, Scalar(255, 255, 255), 2); 
				}
			}
		}
	}
	void Imageprocess::ImgReverse(const Mat &img, Mat &img_reverse)
	{
		img.convertTo(img_reverse, CV_8UC1);

		for (int i = 1; i < img.rows - 1; i++)
		{
			for (int j = 1; j < img.cols - 1; j++)
			{
				if (img.at<uchar>(i, j) == 1)
					img_reverse.at<uchar>(i, j) = 0;
				if (img.at<uchar>(i, j) == 0)
					img_reverse.at<uchar>(i, j) = 1;
			}
		}
		//imshow("imgReverse", 255*img_reverse);
	}
	void Imageprocess::ImgFilling(const Mat &img, Mat &img_fill)
	{
		img.convertTo(img_fill, CV_8UC1);

		Mat img_reverse, img_reverse_label;
		//threshold(img, img_reverse, 1, 1, CV_THRESH_BINARY);
		ImgReverse(img, img_reverse);
		CcaBySeedFill(img_reverse, img_reverse_label);

		for (int i = 1; i < img.rows - 1; i++)
		{
			for (int j = 1; j < img.cols - 1; j++)
			{
				if (img_reverse_label.at<int>(i, j) == 2)
					img_reverse.at<uchar>(i, j) = 0;
				img_fill.at<uchar>(i, j) = img.at<uchar>(i, j) + img_reverse.at<uchar>(i, j);
			}
		}

		//imshow("imgfill", 255 * img_fill);
	}

	void Imageprocess::DetectCornerShiTomasi(const Mat &src, const Mat &colorlabel, Mat &cornerwithimg, int minDistance, double qualityLevel)
	{
		Mat imageGray;
		src.convertTo(imageGray, CV_8UC1);
		vector<Point2f> corners;
		int maxCorners = INT_MAX;
		int blockSize = 3;
		bool useHarrisDetector = false;
		double k = 0.04;

		cornerwithimg = colorlabel.clone();
		/// Apply corner detection :Determines strong corners on an image.
		goodFeaturesToTrack(imageGray, corners, maxCorners, qualityLevel, minDistance, Mat(), blockSize, useHarrisDetector, k);

		// Draw corners detected
		for (int i = 0; i < corners.size(); i++)
		{
			circle(cornerwithimg, corners[i], 3, Scalar(255, 255, 255), 1, 8, 0);
		}
	}

	void Imageprocess::saveimg(const Mat &ProjI, const Mat &ProjZ, const Mat &ProjD, const Mat &ProjImf, const Mat &GI, const Mat &GZ, const Mat &BZ, const Mat &BD, const Mat &GIR, const Mat &BI, const Mat &BIF, const Mat &Label, const Mat &Corner)
	{

		imwrite("1_Intensity Projection Image.jpg", ProjI);
		imwrite("2_Elevation Projection Image.jpg", ProjZ);
		imwrite("3_Density Projection Image.jpg", ProjD);
		imwrite("4_Intensity Projection Image after Median Filter.jpg", ProjImf);
		imwrite("5_Intensity Gradient Image.jpg", GI);
		imwrite("6_Slope Image.jpg", GZ);
		imwrite("7_Slope Binary Image.jpg", 255 * BZ);
		imwrite("8_Density Binary Image.jpg", 255 * BD);
		imwrite("9_Road Intensity Gradient Image.jpg", GIR);
		imwrite("10_Road Intensity Binary Image.jpg", 255 * BI);
		imwrite("11_CCA Filter Road Intensity Binary Image.jpg", 255 * BIF);
		imwrite("12_RoadMarkings.jpg", Label);
		imwrite("13_Marking Corners.jpg", Corner);

		//cout << "Image Output Done." << endl;
	}

	void Imageprocess::saveimg(std::string outputFolder,
							   int file_index, const Mat &ProjI, const Mat &ProjZ, const Mat &ProjD, const Mat &ProjImf, const Mat &GI, const Mat &GZ, const Mat &BZ, const Mat &BD, const Mat &GIR, const Mat &BI, const Mat &BIF, const Mat &Label)
	{
		
		if (!boost::filesystem::exists(outputFolder))
		{
			boost::filesystem::create_directory(outputFolder);
		}

		string img1, img2, img3, img4, img5, img6, img7, img8, img9, img10, img11, img12;

		img1 = outputFolder + "/" + "1_Intensity Projection Image.jpg";
		img2 = outputFolder + "/" + "2_Elevation Projection Image.jpg";
		img3 = outputFolder + "/" + "3_Density Projection Image.jpg";
		img4 = outputFolder + "/" + "4_Intensity Projection Image after Median Filter.jpg";
		img5 = outputFolder + "/" + "5_Intensity Gradient Image.jpg";
		img6 = outputFolder + "/" + "6_Slope Image.jpg";
		img7 = outputFolder + "/" + "7_Slope Binary Image.jpg";
		img8 = outputFolder + "/" + "8_Density Binary Image.jpg";
		img9 = outputFolder + "/" + "9_Road Intensity Gradient Image.jpg";
		img10 = outputFolder + "/" + "10_Road Intensity Binary Image.jpg";
		img11 = outputFolder + "/" + "11_CCA Filter Road Intensity Binary Image.jpg";
		img12 = outputFolder + "/" + "12_RoadMarkings.jpg";

		imwrite(img1, ProjI);
		imwrite(img2, ProjZ);
		imwrite(img3, ProjD);
		imwrite(img4, ProjImf);
		imwrite(img5, GI);
		imwrite(img6, GZ);
		imwrite(img7, 255 * BZ);
		imwrite(img8, 255 * BD);
		imwrite(img9, GIR);
		imwrite(img10, 255 * BI);
		imwrite(img11, 255 * BIF);
		imwrite(img12, Label);

		cout << "Image Output Done." << endl;
	}

	//hyj0728 without density
	void Imageprocess::saveimg(std::string outputFolder_base,
							   int file_index, const Mat &ProjI, const Mat &ProjZ, const Mat &ProjImf, const Mat &GI, const Mat &GZ, const Mat &BZ, const Mat &GIR, const Mat &BI, const Mat &BIF, const Mat &Label)
	{
		std::string outputFolder;

		ostringstream oss;
		// oss << "\\Images_"<< file_index;
		oss << "/Images_" << setw(4) << setfill('0') << file_index;
		// outputFolder = outputFolder_base + "\\Geo-reference Images";

		if (!boost::filesystem::exists(outputFolder_base))
		{
			boost::filesystem::create_directory(outputFolder_base);
		}

		outputFolder = outputFolder_base + oss.str();

		if (!boost::filesystem::exists(outputFolder))
		{
			boost::filesystem::create_directory(outputFolder);
		}

		string img1, img2, img3, img4, img5, img6, img7, img8, img9, img10;

		img1 = outputFolder + "/" + "1_Intensity Projection Image.jpg";
		img2 = outputFolder + "/" + "2_Elevation Projection Image.jpg";
		// img3 = outputFolder + "/" + "3_Density Projection Image.jpg";
		img3 = outputFolder + "/" + "3_Intensity Projection Image after Median Filter.jpg";
		img4 = outputFolder + "/" + "4_Intensity Gradient Image.jpg";
		img5 = outputFolder + "/" + "5_Slope Image.jpg";
		img6 = outputFolder + "/" + "6_Slope Binary Image.jpg";
		// img8 = outputFolder + "/" + "8_Density Binary Image.jpg";
		img7 = outputFolder + "/" + "7_Road Intensity Gradient Image.jpg";
		img8 = outputFolder + "/" + "8_Road Intensity Binary Image.jpg";
		img9 = outputFolder + "/" + "9_CCA Filter Road Intensity Binary Image.jpg";
		img10 = outputFolder + "/" + "10_RoadMarkings.jpg";

		imwrite(img1, ProjI);
		imwrite(img2, ProjZ);
		// imwrite(img3, ProjD);
		imwrite(img3, ProjImf);
		imwrite(img4, GI);
		imwrite(img5, GZ);
		imwrite(img6, 255 * BZ);
		// imwrite(img8, 255 * BD);
		imwrite(img7, GIR);
		imwrite(img8, 255 * BI);
		imwrite(img9, 255 * BIF);
		imwrite(img10, Label);

		cout << "Image Output Done." << endl;
	}

	void Imageprocess::saveimg(const Mat &ProjI, const Mat &ProjImf, const Mat &GI, const Mat &BI, const Mat &BIF, const Mat &Label)
	{
		imwrite("1_Intensity Projection Image.jpg", ProjI);
		imwrite("2_Intensity Projection Image after Median Filter.jpg", ProjImf);
		imwrite("3_Intensity Gradient Image.jpg", GI);
		imwrite("4_Road Intensity Binary Image.jpg", 255 * BI);
		imwrite("5_CCA Filter Road Intensity Binary Image.jpg", 255 * BIF);
		imwrite("6_RoadMarkings.jpg", Label);

		//cout << "Image Output Done." << endl;
	}

}