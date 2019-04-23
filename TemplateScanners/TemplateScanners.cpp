//
// Created by byrdofafeather on 2/7/19.
//

#include <utility>
#include <bits/stdc++.h>
#include <map>
#include <cassert>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/python.hpp>
#include <Python.h>

using namespace std;
using namespace cv;
using namespace boost::python;

typedef std::map<std::string, std::deque<std::string>> DescriptorToTemplatesMap;

class Timestamp {
public:
    double time;
    std::string descriptor;
    Timestamp() = default;

    Timestamp (std::string initDescriptor, double initTime) {
        time = initTime;
        descriptor = std::move(initDescriptor);
    }
};

class TemplateScanner {
public:
    std::map<std::string, std::deque<cv::Mat>> templateMats;
    double templateThreshold;

    TemplateScanner(const DescriptorToTemplatesMap &templatePaths, double threshold) {
        templateMats = build_image_threshold_hash(templatePaths);
        templateThreshold = threshold;
    }

    TemplateScanner(const boost::python::dict &templatePaths, double threshold) {
        DescriptorToTemplatesMap final;

        // Convert passed Python types to C++ map
        boost::python::list keys = templatePaths.keys();
        for (int i = 0; i < boost::python::len(keys); i++) {
            std::string currentKey = boost::python::extract<std::string>(keys[i]);
            std::deque<std::string> currentPaths;
            boost::python::list paths = boost::python::extract<boost::python::list>(templatePaths[currentKey]);
            for (int j = 0; j < boost::python::len(paths); j++) {
                currentPaths.push_front(boost::python::extract<std::string>(paths[j]));
            }
            final[currentKey] = currentPaths;
        }

        templateMats = build_image_threshold_hash(final);
        templateThreshold = threshold;
    }

    std::map<std::string, std::deque<cv::Mat>> build_image_threshold_hash(const DescriptorToTemplatesMap &templatePaths) {
        std::map<std::string, std::deque<cv::Mat>> returnMap;
        for (auto const& currentItems: templatePaths) {
            std::string currentDescriptor = currentItems.first;
            std::deque<std::string> paths = currentItems.second;
            for (const std::string &currentPath: paths) {
                // Reads the image and make sure it exists
                cv::Mat loadedImage = imread(currentPath, cv::IMREAD_COLOR);
                if (loadedImage.empty()) { cout << "FAILED TO LOAD IMAGE "; cout << currentPath << endl;}

                // Gets the reversed image
                cv::Mat reversedImage;
                cv::flip(loadedImage, reversedImage, 1);

                std::deque<cv::Mat> images = returnMap[currentDescriptor];
                images.push_front(loadedImage);
                images.push_front(reversedImage);
                returnMap[currentDescriptor] = images;
            }
        }
        return returnMap;
    }

    double match_template(cv::Mat image, cv::Mat templateToMatch, int filterProcess) {
        cv::Mat result;
        int resultCols = image.cols - templateToMatch.cols + 1;
        int resultRows = image.rows - templateToMatch.rows + 1;
        result.create(resultRows, resultCols, CV_32FC1);

        cv::matchTemplate(image, templateToMatch, result, filterProcess);

        double minVal; double maxVal; cv::Point minLoc; cv::Point maxLoc; cv::Point matchLoc;

        cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());
        return maxVal;
    }

    std::string get_best_match(const cv::Mat &image) {
        std::map<std::string, double> positive;
        for (auto const& currentValues : templateMats) {
            std::string currentDescriptor = currentValues.first;
            std::deque<cv::Mat> currentImages = currentValues.second;

            // Gets all matches above the threshold
            double currentDescriptorMaxProbability = 0;
            for (cv::Mat &currentImage: currentImages) {
                double currentMaxProbability = match_template(image, currentImage, cv::TM_CCOEFF_NORMED);
                if (currentMaxProbability >= templateThreshold) {
                    if (currentDescriptorMaxProbability < currentMaxProbability) {
                        currentDescriptorMaxProbability = currentMaxProbability;
                    }
                }
            }
            positive[currentDescriptor] = currentDescriptorMaxProbability;
        }

        // Finds best match above the threshold
        if (positive.size() > 1) {
            double mostProbableDescriptorValue = 0;
            std::string mostProbableDescriptor;
            for (auto const &currentMaxes : positive) {
                std::string currentDescriptor = currentMaxes.first;
                double currentDescriptorMaxValue = currentMaxes.second;
                if (mostProbableDescriptorValue < currentDescriptorMaxValue) {
                    mostProbableDescriptorValue = currentDescriptorMaxValue;
                    mostProbableDescriptor = currentDescriptor;
                }
            }
            return mostProbableDescriptor;
        }

        else if (positive.size() == 1) {
            for (auto const &finalValue : positive) { return finalValue.first; }
        }

        else { return ""; }
    }

    std::string scan(const cv::Mat &image) {
        std::string currentMaxDescriptor = get_best_match(image);
        return currentMaxDescriptor;
    }

    boost::python::str scanPython(const boost::python::str imagePath) {
        cv::Mat image;
        std::string cPath = boost::python::extract<std::string>(imagePath);
        image = cv::imread(cPath);
        assert(!image.empty());
        std::string currentMaxDescriptor = get_best_match(image);
        boost::python::str returnValue(currentMaxDescriptor);
        return returnValue;
    }
};

class VideoScannerThreaded: public TemplateScanner {
public:
    int frameIndexes[2]{};
    std::string videoPath;

    VideoScannerThreaded(const DescriptorToTemplatesMap &templatePaths, std::string initVideoPath,
            int frameIndexStart, int frameIndexEnd, double threshold);

    std::deque<Timestamp> get_timestamps() {
        // Load video and make sure it isn't empty
        cv::VideoCapture video(videoPath);
        assert(video.get(cv::CAP_PROP_FRAME_COUNT) != 0);

        std::deque<Timestamp> timestamps;

        // Set the video to the start of the frame index and get the FPS
        video.set(cv::CAP_PROP_POS_FRAMES, frameIndexes[0]);
        double fps = video.get(cv::CAP_PROP_FPS);

//        if (false) {
//            double windowNumber = video.get(cv::CAP_PROP_POS_FRAMES);
//            cv::namedWindow("Display window " + std::to_string(windowNumber),
//                            WINDOW_AUTOSIZE);// Create a window for display.
//        }

        while(true) {
            // Read the next frame
            cv::Mat currentFrame;
            video.read(currentFrame);

            if (video.get(cv::CAP_PROP_POS_FRAMES) < frameIndexes[1] && !currentFrame.empty()) {
                // Scan the frame with the current templates
                std::string exportDescriptor = scan(currentFrame);


                if (!exportDescriptor.empty()) {
                    cout << "EXPORTING : ";
                    cout << video.get(cv::CAP_PROP_POS_MSEC) / 1000 << endl;
                    cout << "EXPORTING : ";
                    cout << exportDescriptor << endl;

                    // Testing code for running from C++
//                    if (false) {
//                        cv::imshow("Display window " + std::to_string(windowNumber), currentFrame);
//                        cv::waitKey(0);
//                    }

                    Timestamp currentTime(exportDescriptor, video.get(cv::CAP_PROP_POS_MSEC) / 1000);
                    timestamps.push_front(currentTime);
                }

                // Move a second in the video
                double currentFrameNumber = video.get(cv::CAP_PROP_POS_FRAMES);
                video.set(cv::CAP_PROP_POS_FRAMES, currentFrameNumber + (fps / 2));
            }

            else {
                break;
            }
        }

        video.release();
        return timestamps;
    }
};

VideoScannerThreaded::VideoScannerThreaded(const DescriptorToTemplatesMap &templatePaths,
                                           std::string initVideoPath, int frameIndexStart, int frameIndexEnd,
                                           double threshold) : TemplateScanner(templatePaths, threshold) {
    frameIndexes[0] = frameIndexStart;
    frameIndexes[1] = frameIndexEnd;
    videoPath = std::move(initVideoPath);
}

class VideoThreader {
public:
    DescriptorToTemplatesMap templates;
    double threshold;

    VideoThreader(DescriptorToTemplatesMap initTemplatePaths, double initThreshold) {
        templates = initTemplatePaths;
        threshold = initThreshold;
    }

    std::deque<Timestamp> thread_scanners(const std::string &videoPath, int divisor = 1800) {
        // Get the current video to get the frame count and then release
        cv::VideoCapture video(videoPath);
        double frame_count = video.get(cv::CAP_PROP_FRAME_COUNT);
        assert(frame_count != 0);
        video.release();

        // Build the list of threads
        std::deque<std::future<std::deque<Timestamp>>> futures;
        int frameSections = 0;
        int currentFrameStart = 0;
        int currentFrameEnd = divisor;
        while (frame_count > divisor * frameSections) {
            VideoScannerThreaded currentVideoScanner(templates, videoPath, currentFrameStart, currentFrameEnd,
                                                     threshold);
            std::promise<std::deque<Timestamp>> currentReturnPromise;
            std::future<std::deque<Timestamp>> currentPromisedFuture = std::async(&VideoScannerThreaded::get_timestamps,
                    currentVideoScanner);
            futures.push_front(std::move(currentPromisedFuture));
            currentFrameStart += divisor;
            currentFrameEnd += divisor;
            frameSections += 1;
        }

        cout << "===== STARTING " + std::to_string(frameSections) + " THREADS =====" << endl;

        // Start the threads
        std::deque<Timestamp> allTimestamps;
        for (std::future<std::deque<Timestamp>> &future : futures) {
            std::deque<Timestamp> currentTimestamps = future.get();
            for (Timestamp &timestamp : currentTimestamps) {
                allTimestamps.push_front(timestamp);
            }
        }
        return allTimestamps;
    }
};

struct ThreadedVideoScan {
public:
    ThreadedVideoScan() = default;

    boost::python::list run(boost::python::dict templates, boost::python::str pythonVideoPath, double threshold) {
        DescriptorToTemplatesMap final;

        // Convert passed Python types to C++ map
        std::string videoPath = boost::python::extract<std::string>(pythonVideoPath);
        boost::python::list keys = templates.keys();
        for (int i = 0; i < boost::python::len(keys); i++) {
            std::string currentKey = boost::python::extract<std::string>(keys[i]);
            std::deque<std::string> currentPaths;
            boost::python::list paths = boost::python::extract<boost::python::list>(templates[currentKey]);
            for (int j = 0; j < boost::python::len(paths); j++) {
                currentPaths.push_front(boost::python::extract<std::string>(paths[j]));
            }
            final[currentKey] = currentPaths;
        }

        // Call the scanner to create the threads and run them
        VideoThreader scanner(final, threshold);

        // Convert scanner return values from C++ Types to Python types
        std::deque<Timestamp> cTimestamps = scanner.thread_scanners(videoPath);
        boost::python::list pythonTimestamps;
        for (Timestamp &time : cTimestamps) {
                pythonTimestamps.append(boost::python::object(Timestamp(time.descriptor, time.time)));
        }
        return pythonTimestamps;
    }

    void runTest(DescriptorToTemplatesMap templates, std::string videoPath, double threshold) {
        VideoThreader scanner(templates, threshold);
        std::deque<Timestamp> times;
        times = scanner.thread_scanners(videoPath);
        for (Timestamp &time : times) {
            cout << time.time << endl;
            cout << time.descriptor << endl;
        }
    }
};


int main() {
    // Build Test Map
    DescriptorToTemplatesMap testMap;
    std::deque<std::string> testImagePaths;
    testImagePaths.push_front("0.png");
    testMap["Jump"] = testImagePaths;

    // Build Video Path
    std::string videoPath = "mm.mp4";

    // Start test
    ThreadedVideoScan testScanner;
    testScanner.runTest(testMap, videoPath, .7);
}



BOOST_PYTHON_MODULE(templatescanners) {
    class_<TemplateScanner>("TemplateScanner", init<boost::python::dict, double>())
            .def("scan", &TemplateScanner::scanPython);

    class_<ThreadedVideoScan>("ThreadedVideoScan")
            .def("run", &ThreadedVideoScan::run);

    class_<Timestamp>("Timestamp")
            .def_readonly("time", &Timestamp::time)
            .def_readonly("marker", &Timestamp::descriptor);
}
