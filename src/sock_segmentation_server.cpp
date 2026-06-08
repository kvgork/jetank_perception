// sock_segmentation_server.cpp
//
// STUB / skeleton rclcpp_action server for the SegmentSocks action.
//
// This is a Phase 2 placeholder: it hosts the action so the rest of the
// system (clients, launch wiring, behaviour trees) can be developed against a
// real, reachable server. The execute callback does the minimum to drive a
// goal to a terminal SUCCEEDED state: it emits a single feedback message and
// returns a Result with found == false.
//
// TODO(Phase 3): implement the real segmentation pipeline. On goal execute the
//   server should:
//     - Synchronise the latest stereo_msgs/DisparityImage, the matching
//       sensor_msgs/CameraInfo (for the reprojection matrix Q), and the most
//       recent /detections/socks (vision_msgs/Detection2DArray) using a
//       message_filters TimeSynchronizer (or an approximate-time policy).
//     - For each detection whose score >= goal.min_score, reproject the
//       detection's ROI pixels into 3D using cv::reprojectImageTo3D / the Q
//       matrix from CameraInfo, producing a per-detection point set.
//     - Filter the reprojected points: drop invalid/NaN/inf disparities and
//       clip by goal.max_range (z clip, skip when max_range <= 0), then
//       cluster (e.g. Euclidean clustering) to isolate the sock blob from
//       background.
//     - Compute each blob's centroid and axis-aligned bounding-box dimensions,
//       package as a jetank_detection/SockCloud (cloud + centroid + label +
//       score).
//     - Select the sock whose centroid is nearest base_link.
//     - TF-transform the chosen blob into goal.target_frame (default base_link
//       when target_frame is empty) using tf2_ros, set result.found = true and
//       result.sock accordingly.
//     - Publish feedback (processed / total) as detections are reprojected, and
//       optionally publish the chosen blob on /socks/points when
//       goal.publish_debug is true.
//     - Honour cancellation: abort the goal cleanly if the client cancels.

#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include <jetank_detection/action/segment_socks.hpp>

namespace jetank_perception
{

class SockSegmentationServer : public rclcpp::Node
{
public:
  using SegmentSocks = jetank_detection::action::SegmentSocks;
  using GoalHandle = rclcpp_action::ServerGoalHandle<SegmentSocks>;

  explicit SockSegmentationServer(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("sock_segmentation_server", options)
  {
    using namespace std::placeholders;

    action_server_ = rclcpp_action::create_server<SegmentSocks>(
      this,
      "segment_socks",
      std::bind(&SockSegmentationServer::handle_goal, this, _1, _2),
      std::bind(&SockSegmentationServer::handle_cancel, this, _1),
      std::bind(&SockSegmentationServer::handle_accepted, this, _1));

    RCLCPP_INFO(
      this->get_logger(),
      "sock_segmentation_server ready (STUB) — action: /segment_socks");
  }

private:
  rclcpp_action::Server<SegmentSocks>::SharedPtr action_server_;

  rclcpp_action::GoalResponse handle_goal(
    const rclcpp_action::GoalUUID & /*uuid*/,
    std::shared_ptr<const SegmentSocks::Goal> goal)
  {
    RCLCPP_INFO(
      this->get_logger(),
      "Received SegmentSocks goal (target_frame='%s', min_score=%.2f, "
      "max_range=%.2f, publish_debug=%s) — accepting (STUB)",
      goal->target_frame.c_str(), goal->min_score, goal->max_range,
      goal->publish_debug ? "true" : "false");
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handle_cancel(
    const std::shared_ptr<GoalHandle> /*goal_handle*/)
  {
    RCLCPP_INFO(this->get_logger(), "Received cancel request — accepting (STUB)");
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handle_accepted(const std::shared_ptr<GoalHandle> goal_handle)
  {
    // Run execution on a detached thread so the executor stays responsive.
    std::thread{std::bind(&SockSegmentationServer::execute, this, std::placeholders::_1),
      goal_handle}.detach();
  }

  void execute(const std::shared_ptr<GoalHandle> goal_handle)
  {
    RCLCPP_INFO(this->get_logger(), "Executing SegmentSocks goal (STUB)");

    // Emit one feedback message. Phase 3 will update processed/total as it
    // reprojects detections; the stub has nothing to process.
    auto feedback = std::make_shared<SegmentSocks::Feedback>();
    feedback->processed = 0;
    feedback->total = 0;
    goal_handle->publish_feedback(feedback);

    // STUB result: no sock found, default SockCloud left in place.
    auto result = std::make_shared<SegmentSocks::Result>();
    result->found = false;

    goal_handle->succeed(result);
    RCLCPP_INFO(this->get_logger(), "SegmentSocks goal succeeded (found=false, STUB)");
  }
};

}  // namespace jetank_perception

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<jetank_perception::SockSegmentationServer>());
  rclcpp::shutdown();
  return 0;
}
