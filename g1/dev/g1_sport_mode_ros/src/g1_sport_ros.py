#!/usr/bin/env python3
import sys, rclpy, math, json, time
from rclpy.node import Node
from geometry_msgs.msg import Twist
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient


class G1SportRosNode(Node):
    def __init__(self):
        super().__init__('g1_sport_ros_node')
        
        # Declare and get interface parameter
        self.declare_parameter('interface', 'eth0')
        self.interface = self.get_parameter('interface').get_parameter_value().string_value
        
        # Initialize message counter for throttled logging
        self.message_count = 0
        self.log_every_n_messages = 100  # Log every 100 messages for performance
        
        self.get_logger().info(f'Using network interface: {self.interface}')
        
        try:
            # Initialize channel factory with network interface
            ChannelFactoryInitialize(0, self.interface)
            
            # Create and initialize LocoClient
            self.client = LocoClient()
            self.client.SetTimeout(10.0)
            self.client.Init()
            
            self.get_logger().info('G1 SDK initialized successfully')
            
            # Subscribe to cmd_vel topic
            self.cmd_vel_subscription = self.create_subscription(
                Twist,
                'cmd_vel',
                self.cmd_vel_callback,
                10
            )
            
            self.get_logger().info('Subscribed to cmd_vel topic')
            
        except Exception as e:
            self.get_logger().error(f'Failed to initialize G1 client: {str(e)}')
            raise
    
    def cmd_vel_callback(self, msg):
        """
        Callback function for cmd_vel messages.
        Maps Twist message to G1 movement commands.
        Applies minimum velocity threshold for G1's movement constraints.
        """
        # Extract velocity components
        vx = msg.linear.x   # Forward/backward velocity (m/s)
        vy = msg.linear.y   # Left/right velocity (m/s) 
        omega = msg.angular.z  # Rotational velocity (rad/s)
        
        # Apply minimum velocity threshold while preserving direction
        def apply_min_threshold(vel, min_threshold=0.2):
            if abs(vel) > 0.0 and abs(vel) < min_threshold:
                # Scale to minimum while preserving sign/direction
                return math.copysign(min_threshold, vel)
            return vel
        
        # Apply threshold to each axis independently
        vx_adjusted = apply_min_threshold(vx)
        vy_adjusted = apply_min_threshold(vy)
        # Omega doesn't need adjustment - rotation works at low speeds
        
        # Increment message counter
        self.message_count += 1
        
        # Log when adjustments are made (but still throttled)
        velocity_adjusted = (vx != vx_adjusted or vy != vy_adjusted)
        if velocity_adjusted and self.message_count % 50 == 0:  # Log adjustments more frequently
            self.get_logger().info(
                f"Velocity adjusted: vx {vx:.3f}->{vx_adjusted:.3f}, vy {vy:.3f}->{vy_adjusted:.3f}, omega: {omega:.3f}"
            )
        
        try:
            # Send movement command to G1
            self.client.Move(vx_adjusted, vy_adjusted, omega)
            
            # Throttled logging for performance - only log every N messages
            if self.message_count % self.log_every_n_messages == 0:
                self.get_logger().info(
                    f'Processed {self.message_count} commands. Latest: vx: {vx_adjusted:.3f}, vy: {vy_adjusted:.3f}, omega: {omega:.3f}'
                )
            
        except Exception as e:
            self.get_logger().error(f'Failed to send movement command: {str(e)}')
    
    def shutdown(self):
        """Graceful shutdown of the node."""
        try:
            self.get_logger().info('Stopping G1 movement...')
            self.client.StopMove()
            self.get_logger().info('G1 movement stopped')
        except Exception as e:
            self.get_logger().error(f'Error during shutdown: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = None
    
    try:
        node = G1SportRosNode()
        node.get_logger().info('G1 Sport ROS node started')
        
        # Spin the node
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        print('\nReceived keyboard interrupt, shutting down...')
        
    except Exception as e:
        print(f'Error in main: {str(e)}')
        
    finally:
        # Graceful shutdown - ensure StopMove() is called to prevent continued motion
        if node is not None:
            try:
                node.get_logger().info('Ensuring robot stops moving...')
                node.client.StopMove()
                node.get_logger().info('Robot movement stopped')
            except Exception as e:
                print(f'Error stopping robot movement: {str(e)}')
            
            try:
                node.shutdown()
            except Exception as e:
                print(f'Error during node shutdown: {str(e)}')
        
        try:
            rclpy.shutdown()
        except Exception as e:
            print(f'Error during ROS shutdown: {str(e)}')
            
        print('G1 Sport ROS node shut down')


if __name__ == '__main__':
    main()
