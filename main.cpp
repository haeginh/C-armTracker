#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc/imgproc_c.h> // cvFindContours
#include <opencv2/imgproc.hpp>
#include <opencv2/rgbd.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iterator>
#include <set>
#include <cstdio>
#include <iostream>
#include "MultiDeviceCapturer.h"
#include "transformation.h"
#include "AzureKinect.h"
#include <k4a/k4a.hpp>

void templateConvexHull(const std::vector<cv::linemod::Template>& templates,
    int num_modalities, cv::Point offset, cv::Size size,
    cv::Mat& dst);

void drawResponse(const std::vector<cv::linemod::Template>& templates,
    int num_modalities, cv::Mat& dst, cv::Point offset, int T);

cv::Mat displayQuantized(const cv::Mat& quantized);

// Copy of cv_mouse from cv_utilities
class Mouse
{
public:
    static void start(const std::string& a_img_name)
    {
        cv::setMouseCallback(a_img_name.c_str(), Mouse::cv_on_mouse, 0);
    }
    static int event(void)
    {
        int l_event = m_event;
        m_event = -1;
        return l_event;
    }
    static int x(void)
    {
        return m_x;
    }
    static int y(void)
    {
        return m_y;
    }

private:
    static void cv_on_mouse(int a_event, int a_x, int a_y, int, void*)
    {
        m_event = a_event;
        m_x = a_x;
        m_y = a_y;
    }

    static int m_event;
    static int m_x;
    static int m_y;
};
int Mouse::m_event;
int Mouse::m_x;
int Mouse::m_y;

static void help()
{
    printf("Usage: example_rgbd_linemod [templates.yml]\n\n"
        "Place your object on a planar, featureless surface. With the mouse,\n"
        "frame it in the 'color' window and right click to learn a first template.\n"
        "Then press 'l' to enter online learning mode, and move the camera around.\n"
        "When the match score falls between 90-95%% the demo will add a new template.\n\n"
        "Keys:\n"
        "\t h   -- This help page\n"
        "\t l   -- Toggle online learning\n"
        "\t m   -- Toggle printing match result\n"
        "\t t   -- Toggle printing timings\n"
        "\t w   -- Write learned templates to disk\n"
        "\t [ ] -- Adjust matching threshold: '[' down,  ']' up\n"
        "\t q   -- Quit\n\n");
}

// Adapted from cv_timer in cv_utilities
class Timer
{
public:
    Timer() : start_(0), time_(0) {}

    void start()
    {
        start_ = cv::getTickCount();
    }

    void stop()
    {
        CV_Assert(start_ != 0);
        int64 end = cv::getTickCount();
        time_ += end - start_;
        start_ = 0;
    }

    double time()
    {
        double ret = time_ / cv::getTickFrequency();
        time_ = 0;
        return ret;
    }

private:
    int64 start_, time_;
};

// Functions to store detector and templates in single XML/YAML file
static cv::Ptr<cv::linemod::Detector> readLinemod(const std::string& filename)
{
    cv::Ptr<cv::linemod::Detector> detector = cv::makePtr<cv::linemod::Detector>();
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    detector->read(fs.root());

    cv::FileNode fn = fs["classes"];
    for (cv::FileNodeIterator i = fn.begin(), iend = fn.end(); i != iend; ++i)
        detector->readClass(*i);

    return detector;
}


int main(int argc, char* argv[])
{
    // Arguments
    string path = argv[argc-1];
   
    // Various settings and flags
    bool show_match_result = true;
    bool show_timings = false;
    int num_classes = 0;
    int matching_threshold = 80;

    // Timers
    Timer match_timer;

    // Initialize HighGUI
    help();
    cv::namedWindow("color");
    cv::namedWindow("normals");
    Mouse::start("color");

    // Initialize LINEMOD data structures
    cv::Ptr<cv::linemod::Detector> detector;
    std::string filename;
    detector = readLinemod(path+"/templates.yml");

    std::vector<cv::String> ids = detector->classIds();
    num_classes = detector->numClasses();
    printf("Loaded %s with %d classes and %d templates\n",
           argv[1], num_classes, detector->numTemplates());
    if (!ids.empty())
    {
        printf("Class ids:\n");
        std::copy(ids.begin(), ids.end(), std::ostream_iterator<std::string>(std::cout, "\n"));
    }

    int num_modalities = (int)detector->getModalities().size();
    
    //**** Azure Kinect sensor
    std::vector<uint32_t> device_indices{ 0 };
    int32_t color_exposure_usec = 8000;  // somewhat reasonable default exposure time
    int32_t powerline_freq = 2;          // default to a 60 Hz powerline
    MultiDeviceCapturer capturer(device_indices, color_exposure_usec, powerline_freq);
    // Create configurations for devices
    k4a_device_configuration_t main_config = get_master_config();
    main_config.wired_sync_mode = K4A_WIRED_SYNC_MODE_STANDALONE;// no need to have a master cable if it's standalone
    k4a_device_configuration_t secondary_config = get_subordinate_config(); // not used - currently standalone mode
    // Construct all the things that we'll need whether or not we are running with 1 or 2 cameras
    k4a::calibration main_calibration = capturer.get_master_device().get_calibration(main_config.depth_mode,main_config.color_resolution);
    // Set up a transformation. DO THIS OUTSIDE OF YOUR MAIN LOOP! Constructing transformations involves time-intensive
    // hardware setup and should not change once you have a rigid setup, so only call it once or it will run very
    // slowly.
    k4a::transformation main_depth_to_main_color(main_calibration);
    capturer.start_devices(main_config, secondary_config);                                                 

    // Main loop
    cv::Mat color, depth;
    for (;;)
    {
        vector<k4a::capture> captures;
        // secondary_config isn't actually used here because there's no secondary device but the function needs it
        captures = capturer.get_synchronized_captures(secondary_config, true);
        k4a::image main_color_image = captures[0].get_color_image();
        k4a::image main_depth_image = captures[0].get_depth_image();
        // let's green screen out things that are far away.
                // first: let's get the main depth image into the color camera space
        k4a::image main_depth_in_main_color = create_depth_image_like(main_color_image);
        main_depth_to_main_color.depth_image_to_color_camera(main_depth_image, &main_depth_in_main_color);
        //cv::Mat cv_main_depth_in_main_color = depth_to_opencv(main_depth_in_main_color);
        depth = depth_to_opencv(main_depth_in_main_color);
        color = color_to_opencv(main_color_image);
        //depth = depth_to_opencv(main_depth_image);

        std::vector<cv::Mat> sources;
        sources.push_back(color);
        sources.push_back(depth);
        
        cv::Mat display = color.clone();
        // Perform matching
        std::vector<cv::linemod::Match> matches;
        std::vector<cv::String> class_ids;
        std::vector<cv::Mat> quantized_images;
        match_timer.start();
        detector->match(sources, (float)matching_threshold, matches, class_ids, quantized_images);
        match_timer.stop();

        int classes_visited = 0;
        std::set<std::string> visited;

        for (int i = 0; (i < (int)matches.size()) && (classes_visited < num_classes); ++i)
        {
            cv::linemod::Match m = matches[i];

            if (visited.insert(m.class_id).second)
            {
                ++classes_visited;

                if (show_match_result)
                {
                    printf("Similarity: %5.1f%%; x: %3d; y: %3d; class: %s; template: %3d\n",
                        m.similarity, m.x, m.y, m.class_id.c_str(), m.template_id);
                }

                // Draw matching template
                const std::vector<cv::linemod::Template>& templates = detector->getTemplates(m.class_id, m.template_id);
                drawResponse(templates, num_modalities, display, cv::Point(m.x, m.y), detector->getT(0));
            }
        }

        if (show_match_result && matches.empty())
            printf("No matches found...\n");
        if (show_timings)
        {
            printf("Matching: %.2fs\n", match_timer.time());
        }
        if (show_match_result || show_timings)
            printf("------------------------------------------------------------\n");

        cv::imshow("color", display);
        cv::imshow("normals", quantized_images[1]);

        char key = (char)cv::waitKey(10);
        if (key == 'q')
            break;

        switch (key)
        {
        case 'h':
            help();
            break;
        case 'm':
            // toggle printing match result
            show_match_result = !show_match_result;
            printf("Show match result %s\n", show_match_result ? "ON" : "OFF");
            break;
        case 't':
            // toggle printing timings
            show_timings = !show_timings;
            printf("Show timings %s\n", show_timings ? "ON" : "OFF");
            break;
        case '[':
            // decrement threshold
            matching_threshold = std::max(matching_threshold - 1, -100);
            printf("New threshold: %d\n", matching_threshold);
            break;
        case ']':
            // increment threshold
            matching_threshold = std::min(matching_threshold + 1, +100);
            printf("New threshold: %d\n", matching_threshold);
            break;
        default:
            ;
        }
    }
    return 0;
}

static void reprojectPoints(const std::vector<cv::Point3d>& proj, std::vector<cv::Point3d>& real, double f)
{
    real.resize(proj.size());
    double f_inv = 1.0 / f;

    for (int i = 0; i < (int)proj.size(); ++i)
    {
        double Z = proj[i].z;
        real[i].x = (proj[i].x - 320.) * (f_inv * Z);
        real[i].y = (proj[i].y - 240.) * (f_inv * Z);
        real[i].z = Z;
    }
}

static void filterPlane(IplImage* ap_depth, std::vector<IplImage*>& a_masks, std::vector<CvPoint>& a_chain, double f)
{
    const int l_num_cost_pts = 200;

    float l_thres = 4;

    IplImage* lp_mask = cvCreateImage(cvGetSize(ap_depth), IPL_DEPTH_8U, 1);
    cvSet(lp_mask, cvRealScalar(0));

    std::vector<CvPoint> l_chain_vector;

    float l_chain_length = 0;
    float* lp_seg_length = new float[a_chain.size()];

    for (int l_i = 0; l_i < (int)a_chain.size(); ++l_i)
    {
        float x_diff = (float)(a_chain[(l_i + 1) % a_chain.size()].x - a_chain[l_i].x);
        float y_diff = (float)(a_chain[(l_i + 1) % a_chain.size()].y - a_chain[l_i].y);
        lp_seg_length[l_i] = sqrt(x_diff * x_diff + y_diff * y_diff);
        l_chain_length += lp_seg_length[l_i];
    }
    for (int l_i = 0; l_i < (int)a_chain.size(); ++l_i)
    {
        if (lp_seg_length[l_i] > 0)
        {
            int l_cur_num = cvRound(l_num_cost_pts * lp_seg_length[l_i] / l_chain_length);
            float l_cur_len = lp_seg_length[l_i] / l_cur_num;

            for (int l_j = 0; l_j < l_cur_num; ++l_j)
            {
                float l_ratio = (l_cur_len * l_j / lp_seg_length[l_i]);

                CvPoint l_pts;

                l_pts.x = cvRound(l_ratio * (a_chain[(l_i + 1) % a_chain.size()].x - a_chain[l_i].x) + a_chain[l_i].x);
                l_pts.y = cvRound(l_ratio * (a_chain[(l_i + 1) % a_chain.size()].y - a_chain[l_i].y) + a_chain[l_i].y);

                l_chain_vector.push_back(l_pts);
            }
        }
    }
    std::vector<cv::Point3d> lp_src_3Dpts(l_chain_vector.size());

    for (int l_i = 0; l_i < (int)l_chain_vector.size(); ++l_i)
    {
        lp_src_3Dpts[l_i].x = l_chain_vector[l_i].x;
        lp_src_3Dpts[l_i].y = l_chain_vector[l_i].y;
        lp_src_3Dpts[l_i].z = CV_IMAGE_ELEM(ap_depth, unsigned short, cvRound(lp_src_3Dpts[l_i].y), cvRound(lp_src_3Dpts[l_i].x));
        //CV_IMAGE_ELEM(lp_mask,unsigned char,(int)lp_src_3Dpts[l_i].Y,(int)lp_src_3Dpts[l_i].X)=255;
    }
    //cv_show_image(lp_mask,"hallo2");

    reprojectPoints(lp_src_3Dpts, lp_src_3Dpts, f);

    CvMat* lp_pts = cvCreateMat((int)l_chain_vector.size(), 4, CV_32F);
    CvMat* lp_v = cvCreateMat(4, 4, CV_32F);
    CvMat* lp_w = cvCreateMat(4, 1, CV_32F);

    for (int l_i = 0; l_i < (int)l_chain_vector.size(); ++l_i)
    {
        CV_MAT_ELEM(*lp_pts, float, l_i, 0) = (float)lp_src_3Dpts[l_i].x;
        CV_MAT_ELEM(*lp_pts, float, l_i, 1) = (float)lp_src_3Dpts[l_i].y;
        CV_MAT_ELEM(*lp_pts, float, l_i, 2) = (float)lp_src_3Dpts[l_i].z;
        CV_MAT_ELEM(*lp_pts, float, l_i, 3) = 1.0f;
    }
    cvSVD(lp_pts, lp_w, 0, lp_v);

    float l_n[4] = { CV_MAT_ELEM(*lp_v, float, 0, 3),
                    CV_MAT_ELEM(*lp_v, float, 1, 3),
                    CV_MAT_ELEM(*lp_v, float, 2, 3),
                    CV_MAT_ELEM(*lp_v, float, 3, 3) };

    float l_norm = sqrt(l_n[0] * l_n[0] + l_n[1] * l_n[1] + l_n[2] * l_n[2]);

    l_n[0] /= l_norm;
    l_n[1] /= l_norm;
    l_n[2] /= l_norm;
    l_n[3] /= l_norm;

    float l_max_dist = 0;

    for (int l_i = 0; l_i < (int)l_chain_vector.size(); ++l_i)
    {
        float l_dist = l_n[0] * CV_MAT_ELEM(*lp_pts, float, l_i, 0) +
            l_n[1] * CV_MAT_ELEM(*lp_pts, float, l_i, 1) +
            l_n[2] * CV_MAT_ELEM(*lp_pts, float, l_i, 2) +
            l_n[3] * CV_MAT_ELEM(*lp_pts, float, l_i, 3);

        if (fabs(l_dist) > l_max_dist)
            l_max_dist = l_dist;
    }
    //std::cerr << "plane: " << l_n[0] << ";" << l_n[1] << ";" << l_n[2] << ";" << l_n[3] << " maxdist: " << l_max_dist << " end" << std::endl;
    int l_minx = ap_depth->width;
    int l_miny = ap_depth->height;
    int l_maxx = 0;
    int l_maxy = 0;

    for (int l_i = 0; l_i < (int)a_chain.size(); ++l_i)
    {
        l_minx = std::min(l_minx, a_chain[l_i].x);
        l_miny = std::min(l_miny, a_chain[l_i].y);
        l_maxx = std::max(l_maxx, a_chain[l_i].x);
        l_maxy = std::max(l_maxy, a_chain[l_i].y);
    }
    int l_w = l_maxx - l_minx + 1;
    int l_h = l_maxy - l_miny + 1;
    int l_nn = (int)a_chain.size();

    CvPoint* lp_chain = new CvPoint[l_nn];

    for (int l_i = 0; l_i < l_nn; ++l_i)
        lp_chain[l_i] = a_chain[l_i];

    cvFillPoly(lp_mask, &lp_chain, &l_nn, 1, cvScalar(255, 255, 255));

    delete[] lp_chain;

    //cv_show_image(lp_mask,"hallo1");

    std::vector<cv::Point3d> lp_dst_3Dpts(l_h * l_w);

    int l_ind = 0;

    for (int l_r = 0; l_r < l_h; ++l_r)
    {
        for (int l_c = 0; l_c < l_w; ++l_c)
        {
            lp_dst_3Dpts[l_ind].x = l_c + l_minx;
            lp_dst_3Dpts[l_ind].y = l_r + l_miny;
            lp_dst_3Dpts[l_ind].z = CV_IMAGE_ELEM(ap_depth, unsigned short, l_r + l_miny, l_c + l_minx);
            ++l_ind;
        }
    }
    reprojectPoints(lp_dst_3Dpts, lp_dst_3Dpts, f);

    l_ind = 0;

    for (int l_r = 0; l_r < l_h; ++l_r)
    {
        for (int l_c = 0; l_c < l_w; ++l_c)
        {
            float l_dist = (float)(l_n[0] * lp_dst_3Dpts[l_ind].x + l_n[1] * lp_dst_3Dpts[l_ind].y + lp_dst_3Dpts[l_ind].z * l_n[2] + l_n[3]);

            ++l_ind;

            if (CV_IMAGE_ELEM(lp_mask, unsigned char, l_r + l_miny, l_c + l_minx) != 0)
            {
                if (fabs(l_dist) < std::max(l_thres, (l_max_dist * 2.0f)))
                {
                    for (int l_p = 0; l_p < (int)a_masks.size(); ++l_p)
                    {
                        int l_col = cvRound((l_c + l_minx) / (l_p + 1.0));
                        int l_row = cvRound((l_r + l_miny) / (l_p + 1.0));

                        CV_IMAGE_ELEM(a_masks[l_p], unsigned char, l_row, l_col) = 0;
                    }
                }
                else
                {
                    for (int l_p = 0; l_p < (int)a_masks.size(); ++l_p)
                    {
                        int l_col = cvRound((l_c + l_minx) / (l_p + 1.0));
                        int l_row = cvRound((l_r + l_miny) / (l_p + 1.0));

                        CV_IMAGE_ELEM(a_masks[l_p], unsigned char, l_row, l_col) = 255;
                    }
                }
            }
        }
    }
    cvReleaseImage(&lp_mask);
    cvReleaseMat(&lp_pts);
    cvReleaseMat(&lp_w);
    cvReleaseMat(&lp_v);
}

void subtractPlane(const cv::Mat& depth, cv::Mat& mask, std::vector<CvPoint>& chain, double f)
{
    mask = cv::Mat::zeros(depth.size(), CV_8U);
    std::vector<IplImage*> tmp;
    IplImage mask_ipl = cvIplImage(mask);
    tmp.push_back(&mask_ipl);
    IplImage depth_ipl = cvIplImage(depth);
    filterPlane(&depth_ipl, tmp, chain, f);
}

std::vector<CvPoint> maskFromTemplate(const std::vector<cv::linemod::Template>& templates,
    int num_modalities, cv::Point offset, cv::Size size,
    cv::Mat& mask, cv::Mat& dst)
{
    templateConvexHull(templates, num_modalities, offset, size, mask);

    const int OFFSET = 30;
    cv::dilate(mask, mask, cv::Mat(), cv::Point(-1, -1), OFFSET);

    CvMemStorage* lp_storage = cvCreateMemStorage(0);
    CvTreeNodeIterator l_iterator;
    CvSeqReader l_reader;
    CvSeq* lp_contour = 0;

    cv::Mat mask_copy = mask.clone();
    IplImage mask_copy_ipl = cvIplImage(mask_copy);
    cvFindContours(&mask_copy_ipl, lp_storage, &lp_contour, sizeof(CvContour),
        CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

    std::vector<CvPoint> l_pts1; // to use as input to cv_primesensor::filter_plane

    cvInitTreeNodeIterator(&l_iterator, lp_contour, 1);
    while ((lp_contour = (CvSeq*)cvNextTreeNode(&l_iterator)) != 0)
    {
        CvPoint l_pt0;
        cvStartReadSeq(lp_contour, &l_reader, 0);
        CV_READ_SEQ_ELEM(l_pt0, l_reader);
        l_pts1.push_back(l_pt0);

        for (int i = 0; i < lp_contour->total; ++i)
        {
            CvPoint l_pt1;
            CV_READ_SEQ_ELEM(l_pt1, l_reader);
            /// @todo Really need dst at all? Can just as well do this outside
            cv::line(dst, l_pt0, l_pt1, CV_RGB(0, 255, 0), 2);

            l_pt0 = l_pt1;
            l_pts1.push_back(l_pt0);
        }
    }
    cvReleaseMemStorage(&lp_storage);

    return l_pts1;
}

// Adapted from cv_show_angles
cv::Mat displayQuantized(const cv::Mat& quantized)
{
    cv::Mat color(quantized.size(), CV_8UC3);
    for (int r = 0; r < quantized.rows; ++r)
    {
        const uchar* quant_r = quantized.ptr(r);
        cv::Vec3b* color_r = color.ptr<cv::Vec3b>(r);

        for (int c = 0; c < quantized.cols; ++c)
        {
            cv::Vec3b& bgr = color_r[c];
            switch (quant_r[c])
            {
            case 0:   bgr[0] = 0; bgr[1] = 0; bgr[2] = 0;    break;
            case 1:   bgr[0] = 55; bgr[1] = 55; bgr[2] = 55;    break;
            case 2:   bgr[0] = 80; bgr[1] = 80; bgr[2] = 80;    break;
            case 4:   bgr[0] = 105; bgr[1] = 105; bgr[2] = 105;    break;
            case 8:   bgr[0] = 130; bgr[1] = 130; bgr[2] = 130;    break;
            case 16:  bgr[0] = 155; bgr[1] = 155; bgr[2] = 155;    break;
            case 32:  bgr[0] = 180; bgr[1] = 180; bgr[2] = 180;    break;
            case 64:  bgr[0] = 205; bgr[1] = 205; bgr[2] = 205;    break;
            case 128: bgr[0] = 230; bgr[1] = 230; bgr[2] = 230;    break;
            case 255: bgr[0] = 0; bgr[1] = 0; bgr[2] = 255;    break;
            default:  bgr[0] = 0; bgr[1] = 255; bgr[2] = 0;    break;
            }
        }
    }

    return color;
}

// Adapted from cv_line_template::convex_hull
void templateConvexHull(const std::vector<cv::linemod::Template>& templates,
    int num_modalities, cv::Point offset, cv::Size size,
    cv::Mat& dst)
{
    std::vector<cv::Point> points;
    for (int m = 0; m < num_modalities; ++m)
    {
        for (int i = 0; i < (int)templates[m].features.size(); ++i)
        {
            cv::linemod::Feature f = templates[m].features[i];
            points.push_back(cv::Point(f.x, f.y) + offset);
        }
    }

    std::vector<cv::Point> hull;
    cv::convexHull(points, hull);

    dst = cv::Mat::zeros(size, CV_8U);
    const int hull_count = (int)hull.size();
    const cv::Point* hull_pts = &hull[0];
    cv::fillPoly(dst, &hull_pts, &hull_count, 1, cv::Scalar(255));
}

void drawResponse(const std::vector<cv::linemod::Template>& templates,
    int num_modalities, cv::Mat& dst, cv::Point offset, int T)
{
    static const cv::Scalar COLORS[5] = { CV_RGB(0, 0, 255),
                                          CV_RGB(0, 255, 0),
                                          CV_RGB(255, 255, 0),
                                          CV_RGB(255, 140, 0),
                                          CV_RGB(255, 0, 0) };

    for (int m = 0; m < num_modalities; ++m)
    {
        // NOTE: Original demo recalculated max response for each feature in the TxT
        // box around it and chose the display color based on that response. Here
        // the display color just depends on the modality.
        cv::Scalar color = COLORS[m];

        for (int i = 0; i < (int)templates[m].features.size(); ++i)
        {
            cv::linemod::Feature f = templates[m].features[i];
            cv::Point pt(f.x + offset.x, f.y + offset.y);
            cv::circle(dst, pt, T / 2, color);
        }
    }
}
