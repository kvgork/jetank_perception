#!/usr/bin/env python3
"""
Simple camera launch file for quick testing
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    
    # Basic configuration arguments
    resolution_arg = DeclareLaunchArgument(
        'resolution',
        default_value='640x480',
        choices=['640x480', '1280x720', '1920x1080'],
        description='Camera resolution preset'
    )
    
    fps_arg = DeclareLaunchArgument(
        'fps',
        default_value='30',
        description='Camera frame rate'
    )
    
    # Parse resolution into width/height
    # This is a bit advanced - alternatively you could just use separate width/height args
    
    camera_node = Node(
        package='jetank_perception',  # Change this to your package name
        executable='camera_node',
        name='camera_node',
        output='screen',
        parameters=[{
            # IMX219 typical configurations
            'camera_width': 640,      # or use LaunchConfiguration for dynamic
            'camera_height': 480,
            'camera_fps': LaunchConfiguration('fps'),
            'camera_format': 'NV12',  # Good for IMX219
            'sensor_id': 0,           # First CSI camera
            'use_hardware_acceleration': True,  # Essential for Jetson
            'publish_rate_hz': LaunchConfiguration('fps'),  # Match camera FPS
        }],
        respawn=True,
    )
    
    return LaunchDescription([
        fps_arg,
        camera_node,
    ])