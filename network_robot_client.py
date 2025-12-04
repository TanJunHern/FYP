#!/usr/bin/env python3
"""
Network Robot Client - Runs on MacBook
Sends robot commands to RPI server via TCP

Usage:
    from network_robot_client import NetworkRobotClient

    client = NetworkRobotClient(rpi_ip="192.168.2.1")
    client.connect()
    client.move_forward(50)
    client.disconnect()
"""

import socket
import json
import time
import logging
from typing import Optional, Dict, Any

class NetworkRobotClient:
    """
    Network client for sending robot commands to RPI server
    """

    def __init__(self, rpi_ip: str = "192.168.2.1", rpi_port: int = 8888, timeout: float = 5.0):
        """
        Initialize Network Robot Client

        Args:
            rpi_ip: IP address of Raspberry Pi
            rpi_port: Port of robot server on RPI
            timeout: Socket timeout in seconds
        """
        self.rpi_ip = rpi_ip
        self.rpi_port = rpi_port
        self.timeout = timeout
        self.socket = None
        self.connected = False

        # Setup logging
        self.logger = logging.getLogger(__name__)

        print(f"Network Robot Client initialized")
        print(f"RPI IP: {rpi_ip}, Port: {rpi_port}")
        print(f"Timeout: {timeout}s")

    def connect(self) -> bool:
        """
        Establish connection to RPI server

        Returns:
            True if connection successful, False otherwise
        """
        try:
            print(f"Connecting to RPI robot server at {self.rpi_ip}:{self.rpi_port}...")

            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.rpi_ip, self.rpi_port))

            self.connected = True
            print("Connected to RPI robot server successfully!")

            # Test connection with status command
            status_response = self.get_status()
            if status_response and status_response.get("status") == "success":
                print("Robot server communication test: PASSED")
                return True
            else:
                print("Robot server communication test: FAILED")
                self.disconnect()
                return False

        except socket.timeout:
            print(f"Connection timeout - RPI server not responding")
            return False
        except ConnectionRefusedError:
            print(f"Connection refused - RPI server not running on {self.rpi_ip}:{self.rpi_port}")
            return False
        except Exception as e:
            print(f"Failed to connect to RPI server: {e}")
            return False

    def disconnect(self):
        """Disconnect from RPI server"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None

        self.connected = False
        print("Disconnected from RPI robot server")

    def _send_command(self, command: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Send command to RPI server

        Args:
            command: Command dictionary

        Returns:
            Response dictionary or None if failed
        """
        if not self.connected or not self.socket:
            print("Error: Not connected to RPI server")
            return None

        try:
            # Send command
            command_data = json.dumps(command).encode('utf-8')
            self.socket.send(command_data)

            # Receive response
            response_data = self.socket.recv(1024)
            response = json.loads(response_data.decode('utf-8'))

            return response

        except socket.timeout:
            print("Command timeout - no response from RPI server")
            return None
        except ConnectionResetError:
            print("Connection lost to RPI server")
            self.connected = False
            return None
        except json.JSONDecodeError as e:
            print(f"Invalid response format from RPI server: {e}")
            return None
        except Exception as e:
            print(f"Error sending command: {e}")
            return None

    def move_forward(self, speed: int = 50) -> bool:
        """
        Move robot forward

        Args:
            speed: Forward speed in mm/sec (1-1000)

        Returns:
            True if command successful
        """
        command = {"action": "move_forward", "speed": speed}
        response = self._send_command(command)

        if response and response.get("status") == "success":
            return True
        else:
            error_msg = response.get("message", "Unknown error") if response else "No response"
            print(f"Move forward failed: {error_msg}")
            return False

    def move_backward(self, speed: int = 50) -> bool:
        """
        Move robot backward

        Args:
            speed: Backward speed in mm/sec (1-1000)

        Returns:
            True if command successful
        """
        command = {"action": "move_backward", "speed": speed}
        response = self._send_command(command)

        if response and response.get("status") == "success":
            return True
        else:
            error_msg = response.get("message", "Unknown error") if response else "No response"
            print(f"Move backward failed: {error_msg}")
            return False

    def turn_left(self, angle: int = 800, speed: int = 50) -> bool:
        """
        Turn robot left while moving forward

        Args:
            angle: Steering angle (1-1000)
            speed: Forward speed in mm/sec

        Returns:
            True if command successful
        """
        command = {"action": "turn_left", "angle": angle, "speed": speed}
        response = self._send_command(command)

        if response and response.get("status") == "success":
            return True
        else:
            error_msg = response.get("message", "Unknown error") if response else "No response"
            print(f"Turn left failed: {error_msg}")
            return False

    def turn_right(self, angle: int = 800, speed: int = 50) -> bool:
        """
        Turn robot right while moving forward

        Args:
            angle: Steering angle (1-1000)
            speed: Forward speed in mm/sec

        Returns:
            True if command successful
        """
        command = {"action": "turn_right", "angle": angle, "speed": speed}
        response = self._send_command(command)

        if response and response.get("status") == "success":
            return True
        else:
            error_msg = response.get("message", "Unknown error") if response else "No response"
            print(f"Turn right failed: {error_msg}")
            return False

    def stop(self) -> bool:
        """
        Stop the robot

        Returns:
            True if command successful
        """
        command = {"action": "stop"}
        response = self._send_command(command)

        if response and response.get("status") == "success":
            return True
        else:
            error_msg = response.get("message", "Unknown error") if response else "No response"
            print(f"Stop failed: {error_msg}")
            return False

    def pivot_left(self, angle: int = 800) -> bool:
        """
        Pivot robot left in place

        Args:
            angle: Steering angle (1-1000)

        Returns:
            True if command successful
        """
        command = {"action": "pivot_left", "angle": angle}
        response = self._send_command(command)

        if response and response.get("status") == "success":
            return True
        else:
            error_msg = response.get("message", "Unknown error") if response else "No response"
            print(f"Pivot left failed: {error_msg}")
            return False

    def pivot_right(self, angle: int = 800) -> bool:
        """
        Pivot robot right in place

        Args:
            angle: Steering angle (1-1000)

        Returns:
            True if command successful
        """
        command = {"action": "pivot_right", "angle": angle}
        response = self._send_command(command)

        if response and response.get("status") == "success":
            return True
        else:
            error_msg = response.get("message", "Unknown error") if response else "No response"
            print(f"Pivot right failed: {error_msg}")
            return False

    def get_status(self) -> Optional[Dict[str, Any]]:
        """
        Get robot status

        Returns:
            Status dictionary or None if failed
        """
        command = {"action": "status"}
        response = self._send_command(command)
        return response

    def __enter__(self):
        """Context manager entry"""
        if self.connect():
            return self
        else:
            raise ConnectionError("Failed to connect to RPI robot server")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.connected:
            self.stop()  # Stop robot before disconnecting
        self.disconnect()

def test_client():
    """Test function for network robot client"""
    print("=" * 50)
    print("Network Robot Client Test")
    print("=" * 50)

    rpi_ip = input("Enter RPI IP address (default: 192.168.2.1): ").strip()
    if not rpi_ip:
        rpi_ip = "192.168.2.1"

    try:
        with NetworkRobotClient(rpi_ip=rpi_ip) as client:
            print("\nConnection successful! Testing commands...")

            # Test status
            status = client.get_status()
            print(f"Robot status: {status}")

            # Test movements
            print("\n1. Moving forward...")
            client.move_forward(100)
            time.sleep(2)

            print("2. Stopping...")
            client.stop()
            time.sleep(1)

            print("3. Turning left...")
            client.turn_left(500, 100)
            time.sleep(2)

            print("4. Stopping...")
            client.stop()
            time.sleep(1)

            print("5. Moving backward...")
            client.move_backward(100)
            time.sleep(2)

            print("6. Final stop...")
            client.stop()

            print("\nTest completed successfully!")

    except ConnectionError as e:
        print(f"Connection error: {e}")
        print("Make sure the RPI robot server is running.")
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test error: {e}")

if __name__ == "__main__":
    test_client()
