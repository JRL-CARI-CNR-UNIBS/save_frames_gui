from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution([
        FindPackageShare('save_frames_gui'),
        'config',
        'save_frames_gui.yaml',
    ])

    config_arg = DeclareLaunchArgument(
        'config_file',
        default_value=default_config,
        description='Path to save_frames_gui YAML configuration file.',
    )

    gui_node = Node(
        package='save_frames_gui',
        executable='save_frames_gui_node',
        name='save_frames_gui_node',
        output='screen',
        parameters=[{'config_file': LaunchConfiguration('config_file')}],
    )

    return LaunchDescription([config_arg, gui_node])
