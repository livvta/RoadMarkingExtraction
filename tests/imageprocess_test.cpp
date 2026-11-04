#include "imageprocess.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using roadmarking::Bounds;
using roadmarking::Imageprocess;

namespace
{
void require(bool ok, const char *message)
{
    if (!ok) throw std::runtime_error(message);
}

void add(pcXYZIPtr &cloud, float x, float y, float z, float intensity)
{
    pcl::PointXYZI point;
    point.x = x; point.y = y; point.z = z; point.intensity = intensity;
    cloud->push_back(point);
}

// Independent, small-data reference for the former bucket-based projection.
std::vector<std::vector<pcl::PointXYZI>> buckets(const Imageprocess &ip, const pcXYZIPtr &cloud)
{
    std::vector<std::vector<pcl::PointXYZI>> result(ip.nx * ip.ny);
    for (const auto &p : *cloud)
    {
        const int row = (p.x - ip.minX) / ip.res;
        const int col = (p.y - ip.minY) / ip.res;
        result.at(row * ip.ny + col).push_back(p);
    }
    return result;
}

void checkProjection(Imageprocess &ip, const pcXYZIPtr &cloud, int which)
{
    const auto cells = buckets(ip, cloud);
    std::vector<float> intensities(cells.size()), elevations(cells.size());
    double mean = 0, variance = 0;
    int populated = 0;
    for (std::size_t pixel = 0; pixel < cells.size(); ++pixel)
    {
        double intensity = 0, height = ip.minZ;
        for (const auto &point : cells[pixel])
        {
            intensity += point.intensity;
            height += point.z;
        }
        intensities[pixel] = (0.001 + intensity) / (cells[pixel].size() + 1);
        elevations[pixel] = (0.1 + height) / (cells[pixel].size() + 1);
        if (intensities[pixel] > 1) { mean += intensities[pixel]; ++populated; }
    }
    mean /= populated;
    for (float value : intensities) if (value > 1) variance += (value - mean) * (value - mean);
    const double cap = mean + 3 * std::sqrt(variance / populated);
    const auto range = std::minmax_element(elevations.begin(), elevations.end());
    cv::Mat storageI(ip.nx + 2, ip.ny + 2, CV_8UC1, cv::Scalar(123));
    cv::Mat storageZ = storageI.clone(), storageD = storageI.clone();
    const cv::Rect roi(1, 1, ip.ny, ip.nx);
    cv::Mat imageI = storageI(roi), imageZ = storageZ(roi), imageD = storageD(roi);
    ip.pc2imgI(cloud, which, imageI, 3);
    ip.pc2imgZ(cloud, which, imageZ);
    ip.pc2imgD(cloud, which, imageD, 3.5f);
    for (std::size_t pixel = 0; pixel < cells.size(); ++pixel)
    {
        float value = intensities[pixel];
        if (value > cap) value = cap;
        const unsigned char expectedI = 255 * value / cap;
        const unsigned char expectedZ = 255 * (elevations[pixel] - *range.first) / (*range.second - *range.first);
        const unsigned char expectedD = cells[pixel].size() < 3 ? 255 * cells[pixel].size() / 3 : 255;
        require(imageI.at<uchar>(pixel / ip.ny, pixel % ip.ny) == expectedI, "Intensity projection changed");
        require(imageZ.at<uchar>(pixel / ip.ny, pixel % ip.ny) == expectedZ, "Elevation projection changed");
        require(imageD.at<uchar>(pixel / ip.ny, pixel % ip.ny) == expectedD, "Density projection changed");
    }
    require(storageI.at<uchar>(0, 0) == 123 && storageZ.at<uchar>(0, 0) == 123 &&
            storageD.at<uchar>(0, 0) == 123, "Projection wrote outside an output ROI");
    ip.pc2imgD(cloud, which, imageD, 0.5f);
    for (std::size_t pixel = 0; pixel < cells.size(); ++pixel)
        require(imageD.at<uchar>(pixel / ip.ny, pixel % ip.ny) == (cells[pixel].empty() ? 0 : 255), "Subpixel density lost empty cells");
}

void checkBackProjection(Imageprocess &ip, const pcXYZIPtr &cloud, bool ground)
{
    cv::Mat labels = cv::Mat::zeros(ip.nx, ip.ny, CV_32SC1);
    labels.at<int>(0, 1) = 2; // Filled empty pixel must not invent a point.
    labels.at<int>(2, 2) = 2;
    labels.at<int>(3, 3) = 3;
    ip.totallabel = 3;
    std::vector<pcXYZI> out;
    if (ground) ip.img2pclabel_g(labels, cloud, out, 0.05);
    else ip.img2pclabel_c(labels, cloud, out, 0.05);
    require(out.size() == 2 && out[0].size() == 1 && out[1].size() == 1, "Labeled point retrieval changed");
    require(out[0][0].intensity == 40 && out[1][0].intensity == 255, "Fallback order changed");
    cv::Mat mask = cv::Mat::zeros(ip.nx, ip.ny, CV_8UC3);
    mask.at<cv::Vec3b>(0, 1) = cv::Vec3b(255, 255, 255);
    mask.at<cv::Vec3b>(2, 2) = cv::Vec3b(255, 255, 255);
    pcXYZIPtr retrieved(new pcXYZI);
    if (ground) ip.img2pc_g(mask, cloud, retrieved);
    else ip.img2pc_c(mask, cloud, retrieved);
    require(retrieved->size() == 2 && (*retrieved)[0].intensity == 40 && (*retrieved)[1].intensity == 80,
            "Original point order was not retained");
}
}

int main()
{
    try
    {
        Bounds bounds;
        bounds.min_x = bounds.min_y = -1; bounds.max_x = bounds.max_y = 1;
        bounds.min_z = 0; bounds.max_z = 2;
        pcXYZIPtr cloud(new pcXYZI);
        add(cloud, -0.75, -0.75, 0.1, 20);
        add(cloud, -0.75, -0.75, 0.11, 60);
        add(cloud, 0.25, 0.25, 0.5, 40);
        add(cloud, 0.25, 0.25, 0.8, 80);
        add(cloud, 0.75, 0.75, 1.0, 255);
        Imageprocess original;
        original.savepcgrid(bounds, 0.5, cloud);
        checkProjection(original, cloud, 0);
        checkBackProjection(original, cloud, false);
        pcXYZIPtr discarded(new pcXYZI(*cloud)), ground(new pcXYZI(*cloud)), nonGround(new pcXYZI(*cloud));
        Imageprocess split;
        split.savepcgrid(bounds, 0.5, discarded, ground, nonGround);
        require(discarded->empty(), "Original cloud retained unnecessarily");
        checkProjection(split, ground, 1);
        checkProjection(split, nonGround, 2);
        checkBackProjection(split, ground, true);
        split.releaseGridIndices(2);
        split.releaseGridIndices(2);
        checkProjection(split, ground, 1);
        split.savepcgrid(bounds, 1.0, discarded, ground, nonGround);
        require(split.nx == 3 && split.ny == 3, "Grid reuse kept stale state");
        pcXYZIPtr empty(new pcXYZI);
        split.savepcgrid(bounds, 0.5, empty);
        cv::Mat image;
        split.pc2imgI(empty, 0, image, 3);
        require(cv::countNonZero(image) == 0, "Empty intensity projection is not black");
        split.pc2imgZ(empty, 0, image);
        require(cv::countNonZero(image) == 0, "Constant elevation projection is not black");
        bool rejected = false;
        try { split.pcgrid(bounds, 0); } catch (const std::invalid_argument &) { rejected = true; }
        require(rejected, "Zero resolution was accepted");
        cv::Mat binary = cv::Mat::zeros(7, 7, CV_8UC1), labels;
        binary.at<uchar>(1, 1) = binary.at<uchar>(2, 2) = binary.at<uchar>(0, 0) = 1;
        binary.at<uchar>(4, 4) = 1;
        split.CcaBySeedFill(binary, labels);
        require(split.totallabel == 3 && labels.at<int>(0, 0) == 2 && labels.at<int>(4, 4) == 3,
                "8-connected labels or border handling changed");
        std::cout << "Projection and back-projection checks passed\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
