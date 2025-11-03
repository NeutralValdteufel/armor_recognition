#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <filesystem>

int main() {

    cv::VideoCapture cap("../data/test.avi");
    if (!cap.isOpened()) {
        std::cout << "❌ 无法打开视频文件，请检查路径 ../data/test.avi" << std::endl;
        return -1;
    }

    // 视频参数
    int frame_width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps       = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 30;

    cv::VideoWriter writer("../output/result.avi",
                           cv::VideoWriter::fourcc('M','J','P','G'),
                           fps, cv::Size(frame_width, frame_height));
    if (!writer.isOpened()) {
        std::cout << "❌ 无法创建输出视频文件 ../output/result.avi" << std::endl;
        return -1;
    }

    // 相机参数
    cv::Mat cameraMatrix = (cv::Mat_<double>(3,3) <<
        928.130989, 0, 377.572945,
        0, 930.138391, 283.892859,
        0, 0, 1.0);
    cv::Mat distCoeffs = (cv::Mat_<double>(1,5) <<
        -0.254433647, 0.569431382, 0.00365405229, -0.00109433818, -1.33846840);

    cv::Mat img;
    while (true) {
        cap >> img;
        if (img.empty()) break;

        // 转 HSV 提取红色
        cv::Mat hsv;
        cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
        cv::Mat mask1, mask2, mask;
        cv::inRange(hsv, cv::Scalar(0,50,50), cv::Scalar(10,255,255), mask1);
        cv::inRange(hsv, cv::Scalar(160,50,50), cv::Scalar(180,255,255), mask2);
        cv::bitwise_or(mask1, mask2, mask);
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE,
                         cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));

        // 找轮廓
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        std::vector<cv::RotatedRect> rects;
        for (auto &c : contours) {
            if (c.size() < 5) continue;
            cv::RotatedRect rect = cv::minAreaRect(c);
            float w = rect.size.width, h = rect.size.height;
            float area = w * h;
            float aspect = std::max(h/w, w/h);
            if (aspect > 1.5 && area > 50)
                rects.push_back(rect);
        }

        if (rects.size() >= 2) {
            // 选出最亮的两个灯条
            std::vector<std::pair<double, cv::RotatedRect>> brightRects;
            for (auto &r : rects) {
                cv::Mat maskRect = cv::Mat::zeros(mask.size(), CV_8UC1);
                cv::Point2f pts[4];
                r.points(pts);
                std::vector<cv::Point> contour(pts, pts+4);
                cv::fillConvexPoly(maskRect, contour, 255);
                double meanV = cv::mean(hsv, maskRect)[2]; // V通道亮度
                brightRects.push_back({meanV, r});
            }
            std::sort(brightRects.begin(), brightRects.end(),
                      [](auto &a, auto &b){ return a.first > b.first; });
            rects = {brightRects[0].second, brightRects[1].second};

            // 按 X 排序
            if (rects[0].center.x > rects[1].center.x)
                std::swap(rects[0], rects[1]);

            // 获取每条灯条上下端点
            auto getTopBottom = [](const cv::RotatedRect &r){
                cv::Point2f pts[4];
                r.points(pts);
                cv::Point2f top = pts[0], bottom = pts[0];
                for(int i=1;i<4;i++){
                    if(pts[i].y < top.y) top = pts[i];
                    if(pts[i].y > bottom.y) bottom = pts[i];
                }
                return std::make_pair(top, bottom);
            };

            auto leftTB  = getTopBottom(rects[0]);
            auto rightTB = getTopBottom(rects[1]);

            // 构建矩形四角点（严格对应顺序保证蓝色Z轴方向稳定）
            std::vector<cv::Point2f> imagePoints = {
                leftTB.first,   // 左上
                rightTB.first,  // 右上
                rightTB.second, // 右下
                leftTB.second   // 左下
            };

            // 世界坐标（顺序对应 imagePoints）
            std::vector<cv::Point3f> objectPoints = {
                {-100,  40, 0}, {100,  40, 0},
                {100, -40, 0}, {-100, -40, 0}
            };

            // PnP解算（迭代法）
            cv::Mat rvec, tvec;
            cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs,
                         rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);

            // 坐标轴绘制
            std::vector<cv::Point3f> axes3D = {
                {0,0,0}, {100,0,0}, {0,100,0}, {0,0,100}
            };
            std::vector<cv::Point2f> axes2D;
            cv::projectPoints(axes3D, rvec, tvec, cameraMatrix, distCoeffs, axes2D);

            cv::arrowedLine(img, axes2D[0], axes2D[1], cv::Scalar(0,0,255), 2);   // X 红
            cv::arrowedLine(img, axes2D[0], axes2D[2], cv::Scalar(0,255,0), 2);   // Y 绿
            cv::arrowedLine(img, axes2D[0], axes2D[3], cv::Scalar(255,0,0), 2);   // Z 蓝

            // 绘制矩形端点
            for(auto &p : imagePoints)
                cv::circle(img, p, 5, cv::Scalar(0,255,255), -1);

            // 绘制矩形
            for(int i=0;i<4;i++)
                cv::line(img, imagePoints[i], imagePoints[(i+1)%4], cv::Scalar(255,0,255), 2);
        }

        writer.write(img);
        cv::imshow("Armor Recognition", img);
        if(cv::waitKey(10) == 'q') break;
    }

    cap.release();
    writer.release();
    cv::destroyAllWindows();

    std::cout << "✅ 输出视频保存在 ../output/result.avi" << std::endl;
    return 0;
}
