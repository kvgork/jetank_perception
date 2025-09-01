#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, GroupAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node, LoadComposableNodes
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Package containing the stereo camera node
    stereo_package_name = 'jetank_perception'
    
    # Get package share directory
    pkg_share = FindPackageShare(package=stereo_package_name).find(stereo_package_name)
    
    # ============================================================================
    # LAUNCH ARGUMENTS (for runtime overrides)
    # ============================================================================
    
    # Configuration file argument
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([
            FindPackageShare(stereo_package_name),
            'config',
            'stereo_camera_config.yaml'
        ]),
        description='Path to the stereo camera configuration YAML file'
    )
    
    # Namespace argument
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='stereo_camera',
        description='Namespace for the stereo camera node'
    )
    
    # Log level argument
    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        choices=['debug', 'info', 'warn', 'error', 'fatal'],
        description='Log level for the node'
    )
    
    # Camera resolution overrides (optional - will override YAML if provided)
    camera_width_arg = DeclareLaunchArgument(
        'camera_width',
        default_value='',
        description='Override camera width from config file (leave empty to use config file value)'
    )
    
    camera_height_arg = DeclareLaunchArgument(
        'camera_height', 
        default_value='',
        description='Override camera height from config file (leave empty to use config file value)'
    )
    
    camera_fps_arg = DeclareLaunchArgument(
        'camera_fps',
        default_value='',
        description='Override camera FPS from config file (leave empty to use config file value)'
    )
    
    # Stereo algorithm override
    stereo_algorithm_arg = DeclareLaunchArgument(
        'stereo_algorithm',
        default_value='',
        choices=['', 'GPU_BM', 'CPU_BM', 'GPU_SGBM', 'CPU_SGBM'],
        description='Override stereo algorithm from config file (leave empty to use config file value)'
    )
    
    # Transform publishing argument
    publish_camera_transforms_arg = DeclareLaunchArgument(
        'publish_camera_transforms',
        default_value='false',
        description='Publish static transforms for camera frames'
    )
    
    # Calibration file overrides
    left_camera_info_url_arg = DeclareLaunchArgument(
        'left_camera_info_url',
        default_value='',
        description='Override left camera calibration file URL (leave empty to use config file value)'
    )
    
    right_camera_info_url_arg = DeclareLaunchArgument(
        'right_camera_info_url',
        default_value='',
        description='Override right camera calibration file URL (leave empty to use config file value)'
    )
    
    # ============================================================================
    # PARAMETER OVERRIDE LOGIC
    # ============================================================================
    
    def build_parameter_overrides():
        """Build parameter overrides that are not empty"""
        overrides = {}
        
        # Only add overrides if launch arguments are provided (not empty)
        width_val = LaunchConfiguration('camera_width')
        height_val = LaunchConfiguration('camera_height')
        fps_val = LaunchConfiguration('camera_fps')
        algorithm_val = LaunchConfiguration('stereo_algorithm')
        left_url_val = LaunchConfiguration('left_camera_info_url')
        right_url_val = LaunchConfiguration('right_camera_info_url')
        
        return overrides
    
    # ============================================================================
    # NODE CONFIGURATION
    # ============================================================================
    
    stereo_camera_node = Node(
        package=stereo_package_name,
        executable='stereo_camera_node',
        name='stereo_camera_node',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare(stereo_package_name),
                'config',
                'stereo_camera_config.yaml'
            ]),
            # Direct parameter overrides for key runtime settings
            {
                'calibration.transforms.publish_camera_transforms': LaunchConfiguration('publish_camera_transforms'),
            }
        ],
        output='screen',
        emulate_tty=True,
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        remappings=[
            # Add any topic remappings here if needed
            # ('left/image_raw', 'stereo/left/image_raw'),
            # ('right/image_raw', 'stereo/right/image_raw'),
        ]
    )
    
    # ============================================================================
    # STATIC TRANSFORM PUBLISHERS (Optional)
    # ============================================================================
    
    # Left camera transform
    left_camera_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='left_camera_tf_publisher',
        namespace=LaunchConfiguration('namespace'),
        arguments=[
            '0', '0', '0',  # x, y, z
            '0', '0', '0', '1',  # qx, qy, qz, qw (quaternion)
            'base_link',  # parent frame
            'camera_left_link'  # child frame
        ],
        condition=IfCondition(LaunchConfiguration('publish_camera_transforms'))
    )
    
    # Right camera transform (6cm baseline)
    right_camera_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='right_camera_tf_publisher', 
        namespace=LaunchConfiguration('namespace'),
        arguments=[
            '0.06', '0', '0',  # x, y, z (6cm baseline)
            '0', '0', '0', '1',  # qx, qy, qz, qw (quaternion)
            'base_link',  # parent frame
            'camera_right_link'  # child frame
        ],
        condition=IfCondition(LaunchConfiguration('publish_camera_transforms'))
    )
    
    # ============================================================================
    # LAUNCH INFORMATION
    # ============================================================================
    
    launch_info = LogInfo(
        msg=[
            'Launching Jetson Stereo Camera Node:\n',
            '  Package: ', stereo_package_name, '\n',
            '  Namespace: ', LaunchConfiguration('namespace'), '\n',
            '  Config File: ', LaunchConfiguration('config_file'), '\n',
            '  Log Level: ', LaunchConfiguration('log_level'), '\n',
            '  Transform Publishing: ', LaunchConfiguration('publish_camera_transforms')
        ]
    )
    
    # ============================================================================
    # RETURN LAUNCH DESCRIPTION  
    # ============================================================================
    
    return LaunchDescription([
        # Launch arguments
        config_file_arg,
        namespace_arg,
        log_level_arg,
        camera_width_arg,
        camera_height_arg,
        camera_fps_arg,
        stereo_algorithm_arg,
        publish_camera_transforms_arg,
        left_camera_info_url_arg,
        right_camera_info_url_arg,
        
        # Log launch info
        launch_info,
        
        # Nodes
        stereo_camera_node,
        left_camera_tf,
        right_camera_tf,
    ])