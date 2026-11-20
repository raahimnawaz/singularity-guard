#!/usr/bin/env python
"""Launch the arm in Webots under ros2_control.

    ros2 launch armctl_ros2_control webots_sim.launch.py
    ros2 launch armctl_ros2_control webots_sim.launch.py mode:=fast
"""

import os

import launch
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from webots_ros2_driver.wait_for_controller_connection import WaitForControllerConnection
from webots_ros2_driver.webots_controller import WebotsController
from webots_ros2_driver.webots_launcher import WebotsLauncher


def get_ros2_nodes(*args):
    package_dir = get_package_share_directory('armctl_ros2_control')
    robot_description_path = os.path.join(package_dir, 'urdf', 'singularity_arm.urdf')
    controllers_yaml = os.path.join(package_dir, 'config', 'controllers.yaml')

    with open(robot_description_path, 'r') as f:
        robot_description_content = f.read()

    use_sim_time = LaunchConfiguration('use_sim_time', default=True)
    # Generous timeout: on a loaded machine Webots can take over a minute to attach.
    controller_manager_timeout = ['--controller-manager-timeout', '180']
    controller_manager_prefix = 'python.exe' if os.name == 'nt' else ''

    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        output='screen',
        prefix=controller_manager_prefix,
        arguments=['joint_state_broadcaster'] + controller_manager_timeout,
    )
    singularity_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        output='screen',
        prefix=controller_manager_prefix,
        arguments=['singularity_controller'] + controller_manager_timeout,
    )
    ros_control_spawners = [joint_state_broadcaster_spawner, singularity_controller_spawner]

    arm_driver = WebotsController(
        robot_name='singularity_arm',
        parameters=[
            {
                'robot_description': robot_description_path,
                'use_sim_time': use_sim_time,
                'set_robot_state_publisher': True,
            },
            controllers_yaml,
        ],
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description_content,
            'use_sim_time': use_sim_time,
        }],
    )

    # Spawn controllers only once the Webots driver reports ready.
    waiting_nodes = WaitForControllerConnection(
        target_driver=arm_driver,
        nodes_to_start=ros_control_spawners,
    )

    return [
        robot_state_publisher,
        arm_driver,
        waiting_nodes,
    ]


def generate_launch_description():
    package_dir = get_package_share_directory('armctl_ros2_control')
    world = LaunchConfiguration('world')
    mode = LaunchConfiguration('mode')

    webots = WebotsLauncher(
        world=PathJoinSubstitution([package_dir, 'worlds', world]),
        mode=mode,
        ros2_supervisor=True,
    )

    # Respawns the ROS 2 nodes if the simulation resets (supervisor exits).
    reset_handler = launch.actions.RegisterEventHandler(
        event_handler=launch.event_handlers.OnProcessExit(
            target_action=webots._supervisor,
            on_exit=get_ros2_nodes,
        )
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'world',
            default_value='singularity_arm.wbt',
            description='World file from the `worlds` directory',
        ),
        DeclareLaunchArgument(
            'mode',
            default_value='realtime',
            description='Webots startup mode: realtime | fast | pause',
        ),
        webots,
        webots._supervisor,

        # Shuts everything down once Webots itself exits.
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit(
                target_action=webots,
                on_exit=[
                    launch.actions.UnregisterEventHandler(event_handler=reset_handler.event_handler),
                    launch.actions.EmitEvent(event=launch.events.Shutdown()),
                ],
            )
        ),
        reset_handler,
    ] + get_ros2_nodes())
