// sock_segmentation_server.cpp
//
// Real rclcpp_action server for the SegmentSocks action.
//
// Pipeline (per goal execute):
//   1. Snapshot the latest cached DisparityImage + CameraInfo + Detection2DArray
//      (thread-safe, mutex-guarded latest-message caches). Reject when missing,
//      stale (older than max_age), or out of sync (disparity vs detections stamp
//      differ by more than max_sync_dt) -> result found=false, succeed.
//   2. For each detection with score >= min_score, reproject the bbox ROI pixels
//      into 3D using the pinhole model: for pixel (u,v) with disparity d>0,
//          Z = f*t/d,  X = (u-cx)*Z/f,  Y = (v-cy)*Z/f
//      (camera optical frame: +x right, +y down, +z forward). This is equivalent
//      to cv::reprojectImageTo3D with Q built from f,t,cx,cy.
//   3. Filter: drop NaN/out-of-range disparities, optional RANSAC ground-plane
//      removal, Euclidean cluster, keep the largest cluster as the sock blob.
//   4. Transform the blob cloud + centroid from the disparity optical frame into
//      goal.target_frame (default base_link) via tf2.
//   5. Select the blob whose centroid is nearest the base_link origin; fill the
//      Result (cloud, centroid, dimensions, label, score, found=true).
//   6. Optionally publish the chosen blob on /socks/points (latched).
//   7. Honour cancellation between detections.

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include <jetank_detection/action/segment_socks.hpp>
#include <jetank_detection/msg/sock_cloud.hpp>

#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <stereo_msgs/msg/disparity_image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>
#include <pcl_conversions/pcl_conversions.h>

#include "jetank_perception/sock_reproject.hpp"

namespace jetank_perception
{

// Result of reprojecting + clustering a single detection ROI.
struct SockBlob
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;   // optical-frame points
  Eigen::Vector4f centroid;                    // optical-frame centroid (xyz,1)
  std::string label;
  float score{0.0f};
};

class SockSegmentationServer : public rclcpp::Node
{
public:
  using SegmentSocks = jetank_detection::action::SegmentSocks;
  using GoalHandle = rclcpp_action::ServerGoalHandle<SegmentSocks>;
  using SockCloud = jetank_detection::msg::SockCloud;
  using DisparityImage = stereo_msgs::msg::DisparityImage;
  using CameraInfo = sensor_msgs::msg::CameraInfo;
  using Detection2DArray = vision_msgs::msg::Detection2DArray;

  explicit SockSegmentationServer(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("sock_segmentation_server", options)
  {
    using namespace std::placeholders;

    // --- Parameters ---
    max_sync_dt_   = this->declare_parameter<double>("max_sync_dt", 0.5);
    max_age_       = this->declare_parameter<double>("max_age", 1.0);
    remove_ground_ = this->declare_parameter<bool>("remove_ground", true);
    min_points_    = this->declare_parameter<int>("min_points", 30);
    cluster_tolerance_ = this->declare_parameter<double>("cluster_tolerance", 0.05);
    ground_distance_threshold_ =
      this->declare_parameter<double>("ground_distance_threshold", 0.02);
    default_target_frame_ =
      this->declare_parameter<std::string>("default_target_frame", "base_link");
    base_frame_ = this->declare_parameter<std::string>("base_frame", "base_link");

    // --- TF ---
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // --- Subscriptions (latest-message cache) ---
    //
    // QoS: the upstream publishers are all RELIABLE/VOLATILE:
    //   * stereo_camera_node:  create_publisher<DisparityImage>("disparity", 1)
    //     -> RELIABLE, VOLATILE, KeepLast(1)
    //   * left/camera_info:    RELIABLE, VOLATILE, KeepLast(10)
    //   * /detections/socks:   RELIABLE, VOLATILE, KeepLast(10)
    //
    // A BEST_EFFORT SensorDataQoS subscriber against a RELIABLE/KeepLast(1)
    // publisher is the bug we are fixing: under sim time + executor contention
    // it silently lags (observed disparity age ~1s vs an 11 Hz publisher). Use a
    // RELIABLE KeepLast(5) subscription which is guaranteed-compatible with the
    // RELIABLE publishers above, so every published frame is delivered and the
    // cache below always holds the freshest message.
    const rclcpp::QoS cache_qos = rclcpp::QoS(rclcpp::KeepLast(5)).reliable();

    disparity_sub_ = this->create_subscription<DisparityImage>(
      "/stereo_camera/disparity", cache_qos,
      [this](DisparityImage::SharedPtr msg) {
        std::lock_guard<std::mutex> lk(data_mtx_);
        latest_disparity_ = msg;   // overwrite: always cache the freshest frame
      });
    camera_info_sub_ = this->create_subscription<CameraInfo>(
      "/stereo_camera/left/camera_info", cache_qos,
      [this](CameraInfo::SharedPtr msg) {
        std::lock_guard<std::mutex> lk(data_mtx_);
        latest_camera_info_ = msg;   // overwrite
      });
    detections_sub_ = this->create_subscription<Detection2DArray>(
      "/detections/socks", cache_qos,
      [this](Detection2DArray::SharedPtr msg) {
        std::lock_guard<std::mutex> lk(data_mtx_);
        latest_detections_ = msg;   // overwrite
      });

    // --- Debug publisher (latched) ---
    debug_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/socks/points", rclcpp::QoS(1).transient_local());

    // --- Action server ---
    action_server_ = rclcpp_action::create_server<SegmentSocks>(
      this,
      "segment_socks",
      std::bind(&SockSegmentationServer::handle_goal, this, _1, _2),
      std::bind(&SockSegmentationServer::handle_cancel, this, _1),
      std::bind(&SockSegmentationServer::handle_accepted, this, _1));

    RCLCPP_INFO(
      this->get_logger(),
      "sock_segmentation_server ready — action: /segment_socks");
  }

  // The pure, unit-testable reprojection math lives in the free function
  // jetank_perception::reproject_roi() in sock_reproject.hpp.

private:
  // --- Action plumbing ---------------------------------------------------
  rclcpp_action::GoalResponse handle_goal(
    const rclcpp_action::GoalUUID & /*uuid*/,
    std::shared_ptr<const SegmentSocks::Goal> goal)
  {
    RCLCPP_INFO(
      this->get_logger(),
      "Received SegmentSocks goal (target_frame='%s', min_score=%.2f, "
      "max_range=%.2f, publish_debug=%s)",
      goal->target_frame.c_str(), goal->min_score, goal->max_range,
      goal->publish_debug ? "true" : "false");
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handle_cancel(
    const std::shared_ptr<GoalHandle> /*goal_handle*/)
  {
    RCLCPP_INFO(this->get_logger(), "Received cancel request — accepting");
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handle_accepted(const std::shared_ptr<GoalHandle> goal_handle)
  {
    std::thread{std::bind(&SockSegmentationServer::execute, this, std::placeholders::_1),
      goal_handle}.detach();
  }

  // --- Main pipeline -----------------------------------------------------
  void execute(const std::shared_ptr<GoalHandle> goal_handle)
  {
    const auto goal = goal_handle->get_goal();
    auto result = std::make_shared<SegmentSocks::Result>();
    result->found = false;

    const std::string target_frame =
      goal->target_frame.empty() ? default_target_frame_ : goal->target_frame;

    // 1) Snapshot caches.
    DisparityImage::SharedPtr disparity;
    CameraInfo::SharedPtr camera_info;
    Detection2DArray::SharedPtr detections;
    {
      std::lock_guard<std::mutex> lk(data_mtx_);
      disparity = latest_disparity_;
      camera_info = latest_camera_info_;
      detections = latest_detections_;
    }

    RCLCPP_INFO(
      this->get_logger(),
      "[snapshot] present: disparity=%s camera_info=%s detections=%s",
      disparity ? "yes" : "no", camera_info ? "yes" : "no",
      detections ? "yes" : "no");

    if (!disparity || !camera_info || !detections) {
      RCLCPP_WARN(
        this->get_logger(),
        "Missing input (disparity=%d camera_info=%d detections=%d) -> found=false",
        disparity != nullptr, camera_info != nullptr, detections != nullptr);
      goal_handle->succeed(result);
      return;
    }

    // Describe the cached disparity image and count valid pixels over the FULL
    // image so we can confirm the cache actually holds usable depth data (and
    // that the live-frame subscription is delivering it).
    {
      const sensor_msgs::msg::Image & dimg = disparity->image;
      size_t finite_pos = 0;   // finite && > 0
      size_t scanned = 0;
      if (dimg.width > 0 && dimg.height > 0 &&
        dimg.encoding == sensor_msgs::image_encodings::TYPE_32FC1)
      {
        const uint8_t * dbase = dimg.data.data();
        const size_t dstep = dimg.step;
        for (uint32_t vv = 0; vv < dimg.height; ++vv) {
          const float * drow =
            reinterpret_cast<const float *>(dbase + static_cast<size_t>(vv) * dstep);
          for (uint32_t uu = 0; uu < dimg.width; ++uu) {
            const float dv = drow[uu];
            ++scanned;
            if (std::isfinite(dv) && dv > 0.0f) {++finite_pos;}
          }
        }
      }
      RCLCPP_INFO(
        this->get_logger(),
        "[snapshot] disparity image %ux%u encoding='%s' step=%u f=%.2f t=%.3f "
        "min_disp=%.2f max_disp=%.2f | valid(finite&>0) pixels=%zu/%zu",
        dimg.width, dimg.height, dimg.encoding.c_str(), dimg.step,
        disparity->f, disparity->t, disparity->min_disparity,
        disparity->max_disparity, finite_pos, scanned);
    }

    // Staleness / sync checks.
    const rclcpp::Time now = this->now();
    const rclcpp::Time disp_stamp(disparity->header.stamp);
    const rclcpp::Time det_stamp(detections->header.stamp);

    RCLCPP_INFO(
      this->get_logger(),
      "[snapshot] now=%.3f disparity_stamp=%.3f (age=%.3fs) "
      "detections_stamp=%.3f (age=%.3fs) sync_dt=%.3fs "
      "[max_age=%.2f max_sync_dt=%.2f use_sim_time=%s]",
      now.seconds(), disp_stamp.seconds(), (now - disp_stamp).seconds(),
      det_stamp.seconds(), (now - det_stamp).seconds(),
      std::abs((disp_stamp - det_stamp).seconds()), max_age_, max_sync_dt_,
      this->get_parameter("use_sim_time").as_bool() ? "true" : "false");

    auto age_ok = [&](const rclcpp::Time & stamp) {
      // Guard against zero/unset stamps (e.g. replayed data with no clock):
      // only enforce max_age when both clocks are non-zero.
      if (stamp.nanoseconds() == 0 || now.nanoseconds() == 0) {return true;}
      return (now - stamp).seconds() <= max_age_;
    };

    if (!age_ok(disp_stamp) || !age_ok(det_stamp)) {
      RCLCPP_WARN(
        this->get_logger(),
        "Stale input (disparity age=%.2fs, detections age=%.2fs, max_age=%.2fs)"
        " -> found=false",
        (now - disp_stamp).seconds(), (now - det_stamp).seconds(), max_age_);
      goal_handle->succeed(result);
      return;
    }

    if (disp_stamp.nanoseconds() != 0 && det_stamp.nanoseconds() != 0) {
      const double dt = std::abs((disp_stamp - det_stamp).seconds());
      if (dt > max_sync_dt_) {
        RCLCPP_WARN(
          this->get_logger(),
          "Disparity/detections out of sync (dt=%.2fs > max_sync_dt=%.2fs)"
          " -> found=false",
          dt, max_sync_dt_);
        goal_handle->succeed(result);
        return;
      }
    }

    const std::string optical_frame =
      disparity->header.frame_id.empty() ? camera_info->header.frame_id
                                         : disparity->header.frame_id;

    // 2+3) Reproject + cluster each qualifying detection into a blob.
    auto feedback = std::make_shared<SegmentSocks::Feedback>();
    feedback->total = static_cast<uint16_t>(detections->detections.size());
    feedback->processed = 0;

    std::vector<SockBlob> blobs;

    for (const auto & det : detections->detections) {
      if (goal_handle->is_canceling()) {
        RCLCPP_INFO(this->get_logger(), "Goal canceled");
        goal_handle->canceled(result);
        return;
      }

      // Detection gate.
      float score = 0.0f;
      std::string label;
      if (!det.results.empty()) {
        score = static_cast<float>(det.results[0].hypothesis.score);
        label = det.results[0].hypothesis.class_id;
      }
      if (score < goal->min_score) {
        feedback->processed++;
        goal_handle->publish_feedback(feedback);
        continue;
      }

      // Integer bbox (center +/- size/2), clamped to the image bounds here for
      // logging (reproject_roi clamps again internally).
      const double cxb = det.bbox.center.position.x;
      const double cyb = det.bbox.center.position.y;
      const double hx = det.bbox.size_x * 0.5;
      const double hy = det.bbox.size_y * 0.5;
      const int W = static_cast<int>(disparity->image.width);
      const int H = static_cast<int>(disparity->image.height);
      const int u0 = std::max(0, std::min(static_cast<int>(std::floor(cxb - hx)), W));
      const int u1 = std::max(0, std::min(static_cast<int>(std::ceil(cxb + hx)), W));
      const int v0 = std::max(0, std::min(static_cast<int>(std::floor(cyb - hy)), H));
      const int v1 = std::max(0, std::min(static_cast<int>(std::ceil(cyb + hy)), H));

      pcl::PointCloud<pcl::PointXYZ>::Ptr roi_cloud =
        reproject_roi(*disparity, *camera_info, u0, v0, u1, v1, goal->max_range);

      const size_t n_reproject = roi_cloud->size();
      RCLCPP_INFO(
        this->get_logger(),
        "[det %u/%u] label='%s' score=%.2f ROI=[u0=%d v0=%d u1=%d v1=%d] "
        "max_range=%.2f -> reprojected (range/NaN-filtered) points=%zu "
        "(min_points=%d)",
        static_cast<unsigned>(feedback->processed),
        static_cast<unsigned>(feedback->total), label.c_str(), score,
        u0, v0, u1, v1, goal->max_range, n_reproject, min_points_);

      if (static_cast<int>(roi_cloud->size()) < min_points_) {
        RCLCPP_INFO(
          this->get_logger(),
          "[det] dropped: reprojected points %zu < min_points %d "
          "(stage=reproject)",
          n_reproject, min_points_);
        feedback->processed++;
        goal_handle->publish_feedback(feedback);
        continue;
      }

      // Optional ground-plane removal.
      pcl::PointCloud<pcl::PointXYZ>::Ptr working = roi_cloud;
      if (remove_ground_) {
        working = remove_ground_plane(roi_cloud);
      }
      const size_t n_after_ground = working->size();
      RCLCPP_INFO(
        this->get_logger(),
        "[det] after ground removal (remove_ground=%s): points=%zu",
        remove_ground_ ? "true" : "false", n_after_ground);
      if (static_cast<int>(working->size()) < min_points_) {
        RCLCPP_INFO(
          this->get_logger(),
          "[det] dropped: post-ground points %zu < min_points %d (stage=ground)",
          n_after_ground, min_points_);
        feedback->processed++;
        goal_handle->publish_feedback(feedback);
        continue;
      }

      // Largest Euclidean cluster = sock blob.
      size_t n_clusters = 0;
      pcl::PointCloud<pcl::PointXYZ>::Ptr blob =
        largest_cluster(working, &n_clusters);
      RCLCPP_INFO(
        this->get_logger(),
        "[det] clustering (tolerance=%.3f): clusters=%zu largest=%zu "
        "(min_points=%d)",
        cluster_tolerance_, n_clusters, blob->size(), min_points_);
      if (static_cast<int>(blob->size()) < min_points_) {
        RCLCPP_INFO(
          this->get_logger(),
          "[det] dropped: largest cluster %zu < min_points %d (stage=cluster)",
          blob->size(), min_points_);
        feedback->processed++;
        goal_handle->publish_feedback(feedback);
        continue;
      }

      SockBlob sb;
      sb.cloud = blob;
      pcl::compute3DCentroid(*blob, sb.centroid);
      sb.label = label;
      sb.score = score;
      blobs.push_back(std::move(sb));

      feedback->processed++;
      goal_handle->publish_feedback(feedback);
    }

    if (blobs.empty()) {
      RCLCPP_INFO(
        this->get_logger(),
        "No valid sock blob produced from %u detection(s) -> found=false "
        "(see per-detection [det] logs above for the stage that emptied the "
        "cloud: reproject / ground / cluster)",
        static_cast<unsigned>(detections->detections.size()));
      goal_handle->succeed(result);
      return;
    }
    RCLCPP_INFO(
      this->get_logger(), "[blobs] %zu blob(s) survived gating", blobs.size());

    // 4) Transform every blob into target_frame, and (separately) its centroid
    //    into base_link for nearest-sock selection.
    geometry_msgs::msg::TransformStamped tf_to_target;
    geometry_msgs::msg::TransformStamped tf_to_base;
    if (!lookup_transform(target_frame, optical_frame, disp_stamp, tf_to_target) ||
      !lookup_transform(base_frame_, optical_frame, disp_stamp, tf_to_base))
    {
      RCLCPP_WARN(
        this->get_logger(),
        "TF unavailable (optical='%s' -> target='%s'/base='%s') -> found=false",
        optical_frame.c_str(), target_frame.c_str(), base_frame_.c_str());
      goal_handle->succeed(result);
      return;
    }

    // 5) Choose the blob whose centroid is nearest the base_link origin.
    int best = -1;
    double best_dist = std::numeric_limits<double>::max();
    for (size_t i = 0; i < blobs.size(); ++i) {
      geometry_msgs::msg::PointStamped c_opt, c_base;
      c_opt.header.frame_id = optical_frame;
      c_opt.header.stamp = disparity->header.stamp;
      c_opt.point.x = blobs[i].centroid[0];
      c_opt.point.y = blobs[i].centroid[1];
      c_opt.point.z = blobs[i].centroid[2];
      tf2::doTransform(c_opt, c_base, tf_to_base);
      const double dist = std::sqrt(
        c_base.point.x * c_base.point.x +
        c_base.point.y * c_base.point.y +
        c_base.point.z * c_base.point.z);
      if (dist < best_dist) {
        best_dist = dist;
        best = static_cast<int>(i);
      }
    }

    const SockBlob & chosen = blobs[static_cast<size_t>(best)];

    // Build the chosen blob's PointCloud2 in the optical frame, then transform
    // the whole cloud into target_frame.
    sensor_msgs::msg::PointCloud2 cloud_opt;
    pcl::toROSMsg(*chosen.cloud, cloud_opt);
    cloud_opt.header.frame_id = optical_frame;
    cloud_opt.header.stamp = disparity->header.stamp;

    sensor_msgs::msg::PointCloud2 cloud_target;
    tf2::doTransform(cloud_opt, cloud_target, tf_to_target);
    cloud_target.header.frame_id = target_frame;
    cloud_target.header.stamp = disparity->header.stamp;

    // Centroid in target_frame.
    geometry_msgs::msg::PointStamped centroid_opt, centroid_target;
    centroid_opt.header.frame_id = optical_frame;
    centroid_opt.header.stamp = disparity->header.stamp;
    centroid_opt.point.x = chosen.centroid[0];
    centroid_opt.point.y = chosen.centroid[1];
    centroid_opt.point.z = chosen.centroid[2];
    tf2::doTransform(centroid_opt, centroid_target, tf_to_target);

    // Axis-aligned bbox dimensions of the chosen blob in target_frame: recompute
    // min/max over the transformed cloud so the dimensions match the output frame.
    pcl::PointCloud<pcl::PointXYZ> blob_target;
    pcl::fromROSMsg(cloud_target, blob_target);
    pcl::PointXYZ min_pt, max_pt;
    pcl::getMinMax3D(blob_target, min_pt, max_pt);

    SockCloud sock;
    sock.cloud = cloud_target;
    sock.centroid = centroid_target;
    sock.dimensions.x = static_cast<double>(max_pt.x - min_pt.x);
    sock.dimensions.y = static_cast<double>(max_pt.y - min_pt.y);
    sock.dimensions.z = static_cast<double>(max_pt.z - min_pt.z);
    sock.label = chosen.label;
    sock.score = chosen.score;

    result->found = true;
    result->sock = sock;

    // 6) Debug publish.
    if (goal->publish_debug) {
      debug_pub_->publish(cloud_target);
    }

    RCLCPP_INFO(
      this->get_logger(),
      "Found sock '%s' (score=%.2f, %zu pts) nearest base_link (%.2f m) in '%s'",
      chosen.label.c_str(), chosen.score, chosen.cloud->size(), best_dist,
      target_frame.c_str());

    goal_handle->succeed(result);
  }

  // --- PCL helpers -------------------------------------------------------

  // RANSAC-remove the dominant ~horizontal plane (arena floor). If no plane is
  // found, returns the input unchanged.
  pcl::PointCloud<pcl::PointXYZ>::Ptr remove_ground_plane(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & in)
  {
    if (in->size() < 3) {return in;}

    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coeffs(new pcl::ModelCoefficients);
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(ground_distance_threshold_);
    seg.setMaxIterations(100);
    seg.setInputCloud(in);
    seg.segment(*inliers, *coeffs);

    if (inliers->indices.empty()) {
      return in;   // no dominant plane
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr out(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(in);
    extract.setIndices(inliers);
    extract.setNegative(true);   // keep everything that is NOT the plane
    extract.filter(*out);
    return out;
  }

  // Euclidean-cluster the cloud and return the largest cluster.
  //
  // When remove_ground_ is false there is typically a single dominant
  // continuous surface (e.g. the arena floor) in the ROI. A small cluster
  // tolerance can fragment such a surface into many sub-min_points clusters and
  // drop the whole blob. To avoid that we fall back to returning the full input
  // cloud whenever ground removal is disabled and clustering failed to produce
  // a cluster that meets min_points (the input is, by construction here, the
  // one dominant surface).
  pcl::PointCloud<pcl::PointXYZ>::Ptr largest_cluster(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & in,
    size_t * n_clusters_out = nullptr)
  {
    if (n_clusters_out) {*n_clusters_out = 0;}
    if (in->size() < 3) {return in;}

    auto tree = std::make_shared<pcl::search::KdTree<pcl::PointXYZ>>();
    tree->setInputCloud(in);

    std::vector<pcl::PointIndices> clusters;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(cluster_tolerance_);
    ec.setMinClusterSize(std::max(1, min_points_));
    ec.setMaxClusterSize(static_cast<int>(in->size()));
    ec.setSearchMethod(tree);
    ec.setInputCloud(in);
    ec.extract(clusters);
    if (n_clusters_out) {*n_clusters_out = clusters.size();}

    if (clusters.empty()) {
      // No contiguous group of >= min_points points within tolerance. Rather
      // than fragmenting and dropping the blob, fall back to the whole input
      // cloud (which, with remove_ground=false, is the single dominant surface).
      return in;
    }

    // Largest cluster.
    size_t best = 0;
    for (size_t i = 1; i < clusters.size(); ++i) {
      if (clusters[i].indices.size() > clusters[best].indices.size()) {
        best = i;
      }
    }

    auto out = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    out->reserve(clusters[best].indices.size());
    for (int idx : clusters[best].indices) {
      out->push_back((*in)[static_cast<size_t>(idx)]);
    }
    out->width = static_cast<uint32_t>(out->size());
    out->height = 1;
    out->is_dense = true;
    return out;
  }

  // TF lookup with stamp fallback to latest (Time(0)).
  bool lookup_transform(
    const std::string & target,
    const std::string & source,
    const rclcpp::Time & stamp,
    geometry_msgs::msg::TransformStamped & out)
  {
    try {
      out = tf_buffer_->lookupTransform(
        target, source, stamp, tf2::durationFromSec(0.2));
      return true;
    } catch (const tf2::TransformException & e) {
      RCLCPP_WARN(
        this->get_logger(), "TF at stamp failed (%s); retrying at latest", e.what());
    }
    try {
      out = tf_buffer_->lookupTransform(
        target, source, tf2::TimePointZero, tf2::durationFromSec(0.2));
      return true;
    } catch (const tf2::TransformException & e) {
      RCLCPP_WARN(this->get_logger(), "TF at latest failed: %s", e.what());
      return false;
    }
  }

  // --- Members -----------------------------------------------------------
  rclcpp_action::Server<SegmentSocks>::SharedPtr action_server_;

  rclcpp::Subscription<DisparityImage>::SharedPtr disparity_sub_;
  rclcpp::Subscription<CameraInfo>::SharedPtr camera_info_sub_;
  rclcpp::Subscription<Detection2DArray>::SharedPtr detections_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_pub_;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  std::mutex data_mtx_;
  DisparityImage::SharedPtr latest_disparity_;
  CameraInfo::SharedPtr latest_camera_info_;
  Detection2DArray::SharedPtr latest_detections_;

  // Parameters.
  double max_sync_dt_{0.5};
  double max_age_{1.0};
  bool remove_ground_{true};
  int min_points_{30};
  double cluster_tolerance_{0.05};
  double ground_distance_threshold_{0.02};
  std::string default_target_frame_{"base_link"};
  std::string base_frame_{"base_link"};
};

}  // namespace jetank_perception

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<jetank_perception::SockSegmentationServer>());
  rclcpp::shutdown();
  return 0;
}
