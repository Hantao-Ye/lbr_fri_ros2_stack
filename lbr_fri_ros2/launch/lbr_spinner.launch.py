from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    lbr_spinner_node = Node(
        package="lbr_fri_ros2",
        executable="lbr_spinner",
        emulate_tty=True,
        output="screen"
    )

    return LaunchDescription([
        lbr_spinner_node
    ])