#!/usr/bin/env python
"""Launch the singularity-robust-control arm in Webots under ros2_control.

Structure follows Cyberbotics' own webots_ros2_husarion launch pattern:
WebotsLauncher starts the simulator (with a Ros2Supervisor for spawn/reset
events), WebotsController attaches our robot's <extern> controller process
(which loads webots_ros2_control::Ros2Control -> Ros2ControlSystem -> our
armctl_ros2_control/SingularityController via controller_manager), and
controller_manager spawner nodes activate the controllers once the driver
reports ready.

    ros2 launch armctl_ros2_control webots_sim.launch.py
    ros2 launch armctl_ros2_control webots_sim.launch.py mode:=fast   # headless-friendly
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
    # 60s is enough on an idle machine and not enough on a busy one. Webots has
    # to load the world and attach the extern controller before controller_manager
    # offers its services at all, and under memory pressure that alone can exceed
    # a minute -- at which point the spawners die and the symptom is "controllers
    # never activated", which looks nothing like "the host was short on RAM".
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

    # Waits for the Webots controller to report ready before spawning
    # controllers, so controller_manager doesn't try to claim interfaces from
    # a hardware plugin that hasn't attached yet.
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
