from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    return LaunchDescription([
        Node(
            package='semantic_segmentation',
            executable='semantic_segmentation_node',
            name='semantic_segmentation_node',
            output='screen',
            #parameters=['src/semantic_segmentation/config/mercator_params.yaml']
            parameters=['/home/tedusar/aart25_mercator_ws/src/navigation/perception/segmentation/semantic_segmentation/config/mercator_params.yaml']
        )
    ])
