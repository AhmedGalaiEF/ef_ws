#!/usr/bin/env python3
import multiprocessing as mp
import signal
import time
from multiprocessing import Process, Queue
from queue import Empty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


def sdk_process(interface, command_queue, status_queue):
    """Separate process for SDK operations"""
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
    
    client = None
    running = True
    
    def signal_handler(signum, frame):
        nonlocal running
        running = False
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Initialize SDK in this separate process
        ChannelFactoryInitialize(0, interface)
        client = LocoClient()
        client.SetTimeout(10.0)
        client.Init()
        status_queue.put(('initialized', True))
        print("SDK process: G1 SDK initialized successfully")
        
        # Process commands
        while running:
            try:
                # Check for commands with timeout
                cmd = command_queue.get(timeout=0.1)
                
                if cmd[0] == 'move':
                    _, vx, vy, omega = cmd
                    client.Move(vx, vy, omega)
                elif cmd[0] == 'stop':
                    client.StopMove()
                elif cmd[0] == 'shutdown':
                    break
                    
            except Empty:
                continue
            except Exception as e:
                status_queue.put(('error', str(e)))
                
    except Exception as e:
        status_queue.put(('init_error', str(e)))
        print(f"SDK process: Failed to initialize - {e}")
    finally:
        if client:
            try:
                client.StopMove()
            except:
                pass
        print("SDK process: Shutting down")

class G1SportRosNode(Node):
    def __init__(self):
        super().__init__('g1_sport_ros_node')
        
        # ROS parameters
        self.declare_parameter('interface', 'eth0')
        self.interface = self.get_parameter('interface').value
        self.get_logger().info(f'Using network interface: {self.interface}')
        
        # Multiprocessing setup
        self.command_queue = Queue()
        self.status_queue = Queue()
        self.sdk_process = None
        self.sdk_ready = False
        self.sdk_exit_logged = False
        
        # Start SDK in separate process
        self.start_sdk_process()
        
        # Set up ROS subscription
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            'cmd_vel', 
            self.cmd_vel_callback,
            10
        )
        self.get_logger().info('Subscribed to cmd_vel topic')
        
        # Timer to check SDK status
        self.status_timer = self.create_timer(0.1, self.check_sdk_status)
        
    def start_sdk_process(self):
        """Start the SDK in a separate process"""
        self.sdk_process = Process(
            target=sdk_process, 
            args=(self.interface, self.command_queue, self.status_queue)
        )
        self.sdk_process.start()
        self.get_logger().info('Started SDK process')
        
    def check_sdk_status(self):
        """Check for status updates from SDK process"""
        while True:
            try:
                status_type, data = self.status_queue.get_nowait()
            except Empty:
                break

            if status_type == 'initialized':
                self.sdk_ready = True
                self.get_logger().info('SDK process initialized successfully')
            elif status_type == 'init_error':
                self.get_logger().error(f'SDK initialization failed: {data}')
            elif status_type == 'error':
                self.get_logger().error(f'SDK error: {data}')

        if (
            self.sdk_process is not None
            and not self.sdk_process.is_alive()
            and not self.sdk_ready
            and not self.sdk_exit_logged
        ):
            self.sdk_exit_logged = True
            self.get_logger().error('SDK process exited before becoming ready.')
    
    def cmd_vel_callback(self, msg):
        """Handle velocity commands"""
        if not self.sdk_ready:
            self.get_logger().warn('SDK not ready yet, ignoring cmd_vel')
            return
            
        try:
            # Extract velocities
            vx = msg.linear.x  # Forward/backward
            vy = msg.linear.y  # Left/right
            omega = msg.angular.z  # Rotation
            
            # Apply minimum threshold to avoid small movements
            threshold = 0.2
            vx_adjusted = vx if abs(vx) >= threshold else 0.0
            vy_adjusted = vy if abs(vy) >= threshold else 0.0
            
            # Log the command
            self.get_logger().debug(f'Cmd_vel: vx={vx_adjusted:.2f}, vy={vy_adjusted:.2f}, omega={omega:.2f}')
            
            # Send movement command to SDK process
            self.command_queue.put(('move', vx_adjusted, vy_adjusted, omega))
            
        except Exception as e:
            self.get_logger().error(f'Error in cmd_vel_callback: {e}')
    
    def shutdown(self):
        """Clean shutdown"""
        try:
            # Stop robot movement
            if self.sdk_ready and self.sdk_process and self.sdk_process.is_alive():
                self.command_queue.put(('stop',))
                time.sleep(0.1)
            
            # Shutdown SDK process
            if self.sdk_process and self.sdk_process.is_alive():
                self.command_queue.put(('shutdown',))
            
            if self.sdk_process and self.sdk_process.is_alive():
                self.sdk_process.join(timeout=2)
                if self.sdk_process.is_alive():
                    self.sdk_process.terminate()
                    
            self.get_logger().info('Shutdown complete')
        except Exception as e:
            self.get_logger().error(f'Error during shutdown: {e}')

def main(args=None):
    try:
        rclpy.init(args=args)
        node = G1SportRosNode()
        
        node.get_logger().info('G1 Sport ROS node started (multiprocess mode)')
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        print('\nKeyboard interrupt detected')
    except Exception as e:
        print(f'Error in main: {e}')
    finally:
        if 'node' in locals():
            node.shutdown()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    # Required for multiprocessing
    mp.set_start_method('spawn', force=True)
    main()
