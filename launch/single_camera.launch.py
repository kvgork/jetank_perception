#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch.conditions import IfCondition

def generate_launch_description():
    # Declare launch arguments with default values
    # These can be overridden from command line: ros2 launch pkg launch_file.py camera_width:=1280
    
    camera_width_arg = DeclareLaunchArgument(
        'camera_width',
        default_value='640',
        description='Camera frame width in pixels'
    )
    
    camera_height_arg = DeclareLaunchArgument(
        'camera_height', 
        default_value='480',
        description='Camera frame height in pixels'
    )
    
    camera_fps_arg = DeclareLaunchArgument(
        'camera_fps',
        default_value='30',
        description='Camera capture frame rate'
    )
    
    publish_rate_arg = DeclareLaunchArgument(
        'publish_rate_hz',
        default_value='30.0',  
        description='Rate at which to publish images (Hz)'
    )
    
    sensor_id_arg = DeclareLaunchArgument(
        'sensor_id',
        default_value='0',
        description='Camera sensor ID (0 or 1 for dual CSI setup)'
    )
    
    camera_format_arg = DeclareLaunchArgument(
        'camera_format',
        default_value='NV12',
        description='Camera pixel format (NV12, YUYV, etc.)'
    )
    
    use_hw_accel_arg = DeclareLaunchArgument(
        'use_hardware_acceleration',
        default_value='true',
        description='Enable hardware acceleration (nvvidconv)'
    )
    
    node_name_arg = DeclareLaunchArgument(
        'node_name',
        default_value='camera_node',
        description='Name for the camera node'
    )
    
    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='',
        description='Namespace for the camera node'
    )
    
    enable_debug_arg = DeclareLaunchArgument(
        'debug',
        default_value='false',
        description='Enable debug logging'
    )
    
    # Create the camera node
    camera_node = Node(
        package='jetank_perception',  # Replace with your actual package name
        executable='camera_node',     # Replace with your executable name
        name=LaunchConfiguration('node_name'),
        namespace=LaunchConfiguration('namespace'),
        output='screen',
        parameters=[{
            'camera_width': LaunchConfiguration('camera_width'),
            'camera_height': LaunchConfiguration('camera_height'), 
            'camera_fps': LaunchConfiguration('camera_fps'),
            'camera_format': LaunchConfiguration('camera_format'),
            'sensor_id': LaunchConfiguration('sensor_id'),
            'use_hardware_acceleration': LaunchConfiguration('use_hardware_acceleration'),
            'publish_rate_hz': LaunchConfiguration('publish_rate_hz'),
        }],
        # Conditional debug logging
        arguments=['--ros-args', '--log-level', 
                  PythonExpression(['"DEBUG" if "', LaunchConfiguration('debug'), '" == "true" else "INFO"'])],
        respawn=True,  # Restart node if it crashes
        respawn_delay=2.0,  # Wait 2 seconds before restart
    )
    
    # Optional: Launch RViz2 for visualization
    rviz_arg = DeclareLaunchArgument(
        'launch_rviz',
        default_value='false',
        description='Launch RViz2 for image visualization'
    )
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', '/path/to/your/rviz/config.rviz'],  # Optional: specify config file
        condition=IfCondition(LaunchConfiguration('launch_rviz')),
        output='screen'
    )
    
    # Optional: Launch image view for quick testing
    image_view_arg = DeclareLaunchArgument(
        'launch_image_view',
        default_value='false', 
        description='Launch image_view for quick image display'
    )
    
    image_view_node = Node(
        package='image_view',
        executable='image_view',
        name='image_view',
        remappings=[
            ('image', 'camera/image_raw')
        ],
        condition=IfCondition(LaunchConfiguration('launch_image_view')),
        output='screen'
    )
    
    # Log the configuration being used
    config_info = LogInfo(
        msg=[
            'Starting camera with configuration:\n',
            '  Resolution: ', LaunchConfiguration('camera_width'), 'x', LaunchConfiguration('camera_height'), '\n',
            '  FPS: ', LaunchConfiguration('camera_fps'), '\n', 
            '  Publish Rate: ', LaunchConfiguration('publish_rate_hz'), ' Hz\n',
            '  Sensor ID: ', LaunchConfiguration('sensor_id'), '\n',
            '  Format: ', LaunchConfiguration('camera_format'), '\n',
            '  Hardware Acceleration: ', LaunchConfiguration('use_hardware_acceleration')
        ]
    )
    
    return LaunchDescription([
        # Launch arguments
        camera_width_arg,
        camera_height_arg,
        camera_fps_arg,
        publish_rate_arg,
        sensor_id_arg,
        camera_format_arg,
        use_hw_accel_arg,
        node_name_arg,
        namespace_arg,
        enable_debug_arg,
        rviz_arg,
        image_view_arg,
        
        # Log configuration
        config_info,
        
        # Nodes
        camera_node,
        rviz_node,
        image_view_node,
    ])