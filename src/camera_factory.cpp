#include "jetank_perception/camera_interface.hpp"

namespace jetson_stereo_camera {

// Example implementation for create_camera
std::unique_ptr<CameraInterface> CameraFactory::create_camera(CameraType type) {
    switch(type) {
        case CameraType::JETSON_CSI:
            // return std::make_unique<JetsonCsiCamera>();
            break;
        case CameraType::USB_CAMERA:
            // return std::make_unique<UsbCamera>();
            break;
        case CameraType::VIRTUAL_CAMERA:
            // return std::make_unique<VirtualCamera>();
            break;
        default:
            return nullptr;
    }
    return nullptr;
}

std::vector<std::string> CameraFactory::get_available_cameras() {
    // Return a list of camera names or IDs
    return {"jetson_csi", "usb_camera", "virtual_camera"};
}

}  // namespace jetson_stereo_camera
