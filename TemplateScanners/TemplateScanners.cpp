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
typedef std::map<std::string, double> TemplateToThresholdMap;

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
    TemplateToThresholdMap thresholdMap;
    TemplateToThresholdMap orgThresholdMap; 

    TemplateScanner(const DescriptorToTemplatesMap &templatePaths) {
        templateMats = build_image_map(templatePaths, true);
    }

    TemplateScanner(const DescriptorToTemplatesMap &templatePaths, TemplateToThresholdMap initThresholdMap) {
        templateMats = build_image_map(templatePaths, true);
        thresholdMap = initThresholdMap; 
        orgThresholdMap = initThresholdMap; 
    };

    std::map<std::string, std::deque<cv::Mat>> build_image_map(const DescriptorToTemplatesMap &templatePaths, bool readGrey = false) {
        std::map<std::string, std::deque<cv::Mat>> returnMap;
        for (auto const& currentItems: templatePaths) {
            std::string currentDescriptor = currentItems.first;
            std::deque<std::string> paths = currentItems.second;
            for (const std::string &currentPath: paths) {
                // Reads the image and make sure it exists
                cv::Mat loadedImage; 
                if (readGrey) {
                    loadedImage = imread(currentPath, cv::IMREAD_GRAYSCALE);
                }

                else {
                    loadedImage = imread(currentPath, cv::IMREAD_COLOR);
                }

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

    double match_template(cv::Mat image, cv::Mat templateToMatch, std::string descriptor,
            int templateIndex, int filterProcess) {
        cv::Mat result;
        int resultCols = image.cols - templateToMatch.cols + 1;
        int resultRows = image.rows - templateToMatch.rows + 1;
        result.create(resultRows, resultCols, CV_32FC1);

        cv::matchTemplate(image, templateToMatch, result, filterProcess);

        double minVal; double maxVal; cv::Point minLoc; cv::Point maxLoc; cv::Point matchLoc;

        cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());
        cv::Mat outputImage = image.clone();
        if (maxVal >= thresholdMap[descriptor + std::to_string(templateIndex)]) {
            rectangle(outputImage, maxLoc, Point( maxLoc.x + templateToMatch.cols , maxLoc.y + templateToMatch.rows ), Scalar::all(0), 2, 8, 0);
            rectangle(result, maxLoc, Point( maxLoc.x + templateToMatch.cols , maxLoc.y + templateToMatch.rows ), Scalar::all(0), 2, 8, 0);
            cv::imwrite("results/" + descriptor + std::to_string(maxVal) + ".png", outputImage);
        }

        return maxVal;
    }

    std::string get_best_match(const cv::Mat &image) {
        std::map<std::string, double> positive;
        for (auto const& currentValues : templateMats) {
            std::string currentDescriptor = currentValues.first;
            std::deque<cv::Mat> currentImages = currentValues.second;

            // Gets all matches above the threshold
            double currentDescriptorMaxProbability = 0;
            int TemplateIndex = 0;
            std::string index = currentDescriptor + std::to_string(TemplateIndex);
            for (cv::Mat &currentImage: currentImages) {
                double currentMaxProbability = match_template(image, currentImage, currentDescriptor, TemplateIndex, cv::TM_CCOEFF_NORMED);
                if (currentMaxProbability >= thresholdMap[index]) {
                    if (currentDescriptorMaxProbability < currentMaxProbability) {
                        currentDescriptorMaxProbability = currentMaxProbability;
                        positive[currentDescriptor] = currentDescriptorMaxProbability;
                    }
                }
                else {
                    double currentDifference = thresholdMap[index] - currentMaxProbability; 
                    if (currentDifference <= .03) {
                        if (thresholdMap[index] - currentDifference >= orgThresholdMap[index]/1.3 && thresholdMap[index] - currentDifference >= .6) {
                            thresholdMap[index] -= (currentDifference);
                            cout << "This is the new probablity for " + currentDescriptor + std::to_string(TemplateIndex) << endl; 
                            cout << thresholdMap[index] << endl; 
                        }
                    }
                }

                TemplateIndex += 1;
            }
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
            for (auto const &finalValue : positive) {
                return finalValue.first;
            }
        }

        else { return ""; }
    }

    std::string scan(const cv::Mat &image) {
        std::string currentMaxDescriptor = get_best_match(image);
        return currentMaxDescriptor;
    }


};

class VideoScannerThreaded: public TemplateScanner {
public:
    int frameIndexes[2]{};
    std::string videoPath;

    VideoScannerThreaded(const DescriptorToTemplatesMap &templatePaths, std::string initVideoPath,
            int frameIndexStart, int frameIndexEnd, TemplateToThresholdMap thresholdMap);

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
                cv::Mat grayFrame; 
                cv::cvtColor(currentFrame, grayFrame, cv::COLOR_BGR2GRAY);
                std::string exportDescriptor = scan(grayFrame);

                if (!exportDescriptor.empty()) {
                    // Testing code for running from C++
//                    if (false) {
//                        cv::imshow("Display window " + std::to_string(windowNumber), currentFrame);
//                        cv::waitKey(0);
//                    }
                    cout << "EXPORTING : ";
                    cout << video.get(cv::CAP_PROP_POS_MSEC) / 1000 << endl;
                    cout << "EXPORTING : ";
                    cout << exportDescriptor << endl;
                    Timestamp currentTime(exportDescriptor, video.get(cv::CAP_PROP_POS_MSEC) / 1000);
                    timestamps.push_front(currentTime);
                }

               //  // Move a second in the video
               // double currentFrameNumber = video.get(cv::CAP_PROP_POS_FRAMES);
               // video.set(cv::CAP_PROP_POS_FRAMES, currentFrameNumber + (fps / 10));
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
                                           TemplateToThresholdMap thresholdMap)
                                           : TemplateScanner(templatePaths, thresholdMap) {
    frameIndexes[0] = frameIndexStart;
    frameIndexes[1] = frameIndexEnd;
    videoPath = std::move(initVideoPath);
}


class ThresholdFinder: public TemplateScanner {
public:
    ThresholdFinder(DescriptorToTemplatesMap templatePaths);

    using TemplateScanner::match_template; 

    double static match_template(cv::Mat image, cv::Mat templateToMatch, int filterProcess) {
        cv::Mat result;
        int resultCols = image.cols - templateToMatch.cols + 1;
        int resultRows = image.rows - templateToMatch.rows + 1;
        result.create(resultRows, resultCols, CV_32FC1);

        cv::matchTemplate(image, templateToMatch, result, filterProcess);

        double minVal; double maxVal; cv::Point minLoc; cv::Point maxLoc; cv::Point matchLoc;

        cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());

        return maxVal;
    }

    double static getThreshold(std::string descriptor, cv::Mat templateToMatch, std::string videoPath, long frameGuess, std::string specificDescriptor="Generic") {
        cv::VideoCapture video(videoPath);

        double bestMatch = 0;
        double frameCount = video.get(cv::CAP_PROP_FRAME_COUNT);

        std::random_device rd;
        std::default_random_engine generator(rd());
        std::uniform_int_distribution<long> distribution(0, (long)frameCount);

        std::deque<double> matches; 
        for (int i = 0; i <= frameGuess; i++) {

            long frame = distribution(generator);

            video.set(cv::CAP_PROP_POS_FRAMES, frame);
            cv::Mat currentFrame;
            video.read(currentFrame);

            if (!currentFrame.empty()) {
                cv::Mat grayFrame; 
                cv::cvtColor(currentFrame, grayFrame, cv::COLOR_BGR2GRAY); 
                double currentMatch = match_template(grayFrame, templateToMatch, cv::TM_CCOEFF_NORMED);
                matches.push_front(currentMatch); 
                if (bestMatch <= currentMatch) {
                    bestMatch = currentMatch;
                }
            }

            else {
                continue;
            }
        }
        cout << descriptor + specificDescriptor << endl;
        cout << std::to_string(bestMatch) << endl;
        for (double &match : matches) {
            cout << match << endl; 
        }
        video.release(); 
        double thresh = bestMatch; 
        return thresh;
    }


    TemplateToThresholdMap getThresholds(std::string videoPath) {
        cv::VideoCapture video(videoPath);

        double frameRate = video.get(cv::CAP_PROP_FPS);
        double frameCount = video.get(cv::CAP_PROP_FRAME_COUNT);
        video.release(); 

        double frameGuess = std::log(1 - .9) / std::log(1 - ((1 + frameRate) / frameCount));
        double frameGuessInt = std::floor(frameGuess);

        cout << std::to_string(frameGuessInt) << endl;

        if (frameGuessInt > frameCount) { frameGuessInt = frameCount; }

        std::map<std::string, std::future<double>> tempTemplateToThresholdMap;


        for (auto const &currentItems: templateMats) {
            std::string currentDescriptor = currentItems.first;
            std::deque<cv::Mat> currentImages = currentItems.second;
            int imageIndex = 0;
            for (cv::Mat &currentImage: currentImages) {
                std::promise<double> currentPromise;
                std::future<double> currentPromisedFuture = std::async(getThreshold, currentDescriptor, currentImage,
                                                                       videoPath, frameGuess,
                                                                       std::to_string(imageIndex));
                tempTemplateToThresholdMap[currentDescriptor + std::to_string(imageIndex)] = std::move(
                        currentPromisedFuture);
                imageIndex += 1;
                cout << "=== JUST STARTED " + currentDescriptor + " THREAD " + std::to_string(imageIndex) + " ===" << endl; 
            }
            cout << "=== STARTED " + std::to_string(imageIndex) + " THREADS ===" << endl;
        }


        TemplateToThresholdMap thresholdMap;

        for (auto  &currentItems: tempTemplateToThresholdMap) {
            std::string currentDescriptorIndex = currentItems.first;
            double currentMax = currentItems.second.get();
            cout << "Adding " + currentDescriptorIndex + " with " + std::to_string(currentMax) << endl;
            thresholdMap[currentDescriptorIndex] = currentMax;
        }
        return thresholdMap;
    }
};

ThresholdFinder::ThresholdFinder(DescriptorToTemplatesMap templatePaths): TemplateScanner(templatePaths) { }


class VideoThreader {
public:
    DescriptorToTemplatesMap templates;

    VideoThreader(DescriptorToTemplatesMap initTemplatePaths) {
        templates = initTemplatePaths;
    }

    std::deque<Timestamp> thread_scanners(const std::string &videoPath, int divisor = 1800) {
        // Get the current video to get the frame count and then release
        cv::VideoCapture video(videoPath);
        double frame_count = video.get(cv::CAP_PROP_FRAME_COUNT);
        assert(frame_count != 0);
        video.release();

        ThresholdFinder finder(templates);
        TemplateToThresholdMap thresholdMap = finder.getThresholds(videoPath);

        // Build the list of threads
        std::deque<std::future<std::deque<Timestamp>>> futures;
        int frameSections = 0;
        int currentFrameStart = 0;
        int currentFrameEnd = divisor;
        while (frame_count > divisor * frameSections) {
            VideoScannerThreaded currentVideoScanner(templates, videoPath, 
                currentFrameStart, currentFrameEnd, thresholdMap);
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

    boost::python::list run(boost::python::dict templates, boost::python::str pythonVideoPath) {
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
        VideoThreader scanner(final);

        // Convert scanner return values from C++ Types to Python types
        std::deque<Timestamp> cTimestamps = scanner.thread_scanners(videoPath);
        boost::python::list pythonTimestamps;
        for (Timestamp &time : cTimestamps) {
                pythonTimestamps.append(boost::python::object(Timestamp(time.descriptor, time.time)));
        }
        return pythonTimestamps;
    }

    void runTest(DescriptorToTemplatesMap templates, std::string videoPath) {
        VideoThreader scanner(templates);
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
    testScanner.runTest(testMap, videoPath);
}



BOOST_PYTHON_MODULE(templatescannersbeta) {
    class_<ThreadedVideoScan>("ThreadedVideoScan")
            .def("run", &ThreadedVideoScan::run);

    class_<Timestamp>("Timestamp")
            .def_readonly("time", &Timestamp::time)
            .def_readonly("marker", &Timestamp::descriptor);
}
