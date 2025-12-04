#!/usr/bin/env python3
"""
RPI Robot Server - Runs on Raspberry Pi
Receives robot commands from MacBook and forwards to STM32 via serial

Usage on RPI:
    python3 rpi_robot_server.py --port 8485 --serial_port /dev/ttyUSB0

Transfer this file to your RPI and run it there.
"""

import socket
import json
import serial
import time
import threading
import argparse
import logging
import struct
from typing import Optional, Dict, Any

class STM32Controller:
    """
    Controller for STM32 board via serial/UART
    """

    def __init__(self, serial_port='/dev/ttyUSB0', baudrate=115200, timeout=1.0):
        """
        Initialize STM32 serial connection

        Args:
            serial_port: Serial port path (e.g., /dev/ttyUSB0, /dev/ttyACM0)
            baudrate: Serial baud rate
            timeout: Serial timeout in seconds
        """
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.connected = False

        print(f"STM32 Controller initialized:")
        print(f"  Port: {serial_port}")
        print(f"  Baudrate: {baudrate}")
        print(f"  Timeout: {timeout}s")

    def connect(self) -> bool:
        """
        Connect to STM32 via serial

        Returns:
            True if connection successful
        """
        try:
            print(f"Connecting to STM32 on {self.serial_port}...")
            self.ser = serial.Serial(
                port=self.serial_port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            time.sleep(2)  # Wait for connection to stabilize
            self.connected = True
            print("STM32 connected successfully!")
            return True
        except serial.SerialException as e:
            print(f"Failed to connect to STM32: {e}")
            self.connected = False
            return False
        except Exception as e:
            print(f"Unexpected error connecting to STM32: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """Disconnect from STM32"""
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
            except:
                pass
        self.connected = False
        print("STM32 disconnected")

    def build_packet(self, x_speed: int, z_steering: int) -> bytes:
        """
        Build 12-byte binary packet for STM32FC controller

        Packet format (from NanoCar User Guide):
        - Header: 0x5a
        - Packet length: 0x0c
        - ID: 0x01
        - Function: 0x01
        - X speed (2 bytes MSB, LSB): forward/backward speed in mm/sec
        - Y value (2 bytes MSB, LSB): always 0x0000 (not used)
        - Z steering (2 bytes MSB, LSB): steering angle
        - Reserve: 0x00
        - CRC: 0xff (disabled)

        Args:
            x_speed: Speed in mm/sec. Positive=forward, Negative=backward, 0=stop
            z_steering: Steering angle. Positive=left, Negative=right, 0=straight

        Returns:
            12-byte packet as bytes
        """
        # Clamp values to valid ranges
        x_speed = max(-1000, min(1000, x_speed))
        z_steering = max(-1000, min(1000, z_steering))

        # Convert to 16-bit signed integers (big-endian, MSB first)
        x_bytes = struct.pack('>h', x_speed)  # signed short, big-endian
        y_bytes = struct.pack('>h', 0)        # Y always 0
        z_bytes = struct.pack('>h', z_steering)

        # Build complete packet
        packet = bytes([
            0x5a,           # Header
            0x0c,           # Packet length (12 bytes)
            0x01,           # ID
            0x01,           # Function
        ]) + x_bytes + y_bytes + z_bytes + bytes([
            0x00,           # Reserve byte
            0xff            # CRC (disabled)
        ])

        return packet

    def send_packet(self, x_speed: int, z_steering: int) -> bool:
        """
        Send binary packet to STM32

        Args:
            x_speed: Speed in mm/sec (positive=forward, negative=backward)
            z_steering: Steering angle (positive=left, negative=right)

        Returns:
            True if packet sent successfully
        """
        if not self.connected or not self.ser or not self.ser.is_open:
            print("Error: STM32 not connected")
            return False

        try:
            packet = self.build_packet(x_speed, z_steering)
            self.ser.write(packet)
            self.ser.flush()
            print(f"Sent to STM32: speed={x_speed}, steering={z_steering} | packet={packet.hex()}")
            return True
        except serial.SerialException as e:
            print(f"Serial error sending packet: {e}")
            return False
        except Exception as e:
            print(f"Error sending packet: {e}")
            return False

    def move_forward(self, speed: int) -> bool:
        """
        Move robot forward

        Args:
            speed: Forward speed in mm/sec (1-1000)

        Returns:
            True if command successful
        """
        return self.send_packet(x_speed=speed, z_steering=0)

    def move_backward(self, speed: int) -> bool:
        """
        Move robot backward

        Args:
            speed: Backward speed in mm/sec (1-1000)

        Returns:
            True if command successful
        """
        return self.send_packet(x_speed=-speed, z_steering=0)

    def turn_left(self, angle: int, speed: int) -> bool:
        """
        Turn robot left while moving forward

        Args:
            angle: Steering angle (1-1000)
            speed: Forward speed in mm/sec (1-1000)

        Returns:
            True if command successful
        """
        return self.send_packet(x_speed=speed, z_steering=angle)

    def turn_right(self, angle: int, speed: int) -> bool:
        """
        Turn robot right while moving forward

        Args:
            angle: Steering angle (1-1000)
            speed: Forward speed in mm/sec (1-1000)

        Returns:
            True if command successful
        """
        return self.send_packet(x_speed=speed, z_steering=-angle)

    def stop(self) -> bool:
        """
        Stop the robot

        Returns:
            True if command successful
        """
        return self.send_packet(x_speed=0, z_steering=0)

    def pivot_left(self, angle: int) -> bool:
        """
        Pivot robot left in place

        Args:
            angle: Steering angle (1-1000)

        Returns:
            True if command successful
        """
        return self.send_packet(x_speed=0, z_steering=angle)

    def pivot_right(self, angle: int) -> bool:
        """
        Pivot robot right in place

        Args:
            angle: Steering angle (1-1000)

        Returns:
            True if command successful
        """
        return self.send_packet(x_speed=0, z_steering=-angle)


class RobotServer:
    """
    TCP Server that receives commands from MacBook and forwards to STM32
    """

    def __init__(self, host='0.0.0.0', port=8485, stm32_port='/dev/ttyUSB0'):
        """
        Initialize robot server

        Args:
            host: Server host address (0.0.0.0 for all interfaces)
            port: Server port
            stm32_port: Serial port for STM32
        """
        self.host = host
        self.port = port
        self.stm32 = STM32Controller(serial_port=stm32_port)
        self.running = False
        self.server_socket = None

        print(f"Robot Server initialized:")
        print(f"  Server: {host}:{port}")
        print(f"  STM32 Port: {stm32_port}")

    def start(self):
        """Start the robot server"""
        # Connect to STM32
        if not self.stm32.connect():
            print("Failed to connect to STM32 - server will not start")
            return False

        # Create server socket
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.running = True

            print(f"\n{'='*60}")
            print(f"Robot Server started on {self.host}:{self.port}")
            print(f"Waiting for connections from MacBook...")
            print(f"{'='*60}\n")

            while self.running:
                try:
                    # Accept client connection
                    client_socket, client_address = self.server_socket.accept()
                    print(f"Connection from: {client_address}")

                    # Handle client in separate thread
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, client_address),
                        daemon=True
                    )
                    client_thread.start()

                except KeyboardInterrupt:
                    print("\nServer interrupted by user")
                    break
                except Exception as e:
                    if self.running:
                        print(f"Error accepting connection: {e}")

        except Exception as e:
            print(f"Failed to start server: {e}")
            return False
        finally:
            self.stop()

        return True

    def handle_client(self, client_socket: socket.socket, client_address):
        """
        Handle client connection

        Args:
            client_socket: Client socket
            client_address: Client address tuple
        """
        print(f"Handling client: {client_address}")

        try:
            while self.running:
                # Receive command from client
                data = client_socket.recv(1024)
                if not data:
                    break

                # Parse JSON command
                try:
                    command = json.loads(data.decode('utf-8'))
                    print(f"Received command: {command}")

                    # Process command and get response
                    response = self.process_command(command)

                    # Send response back to client
                    response_data = json.dumps(response).encode('utf-8')
                    client_socket.send(response_data)

                except json.JSONDecodeError as e:
                    print(f"Invalid JSON received: {e}")
                    error_response = {
                        "status": "error",
                        "message": f"Invalid JSON: {str(e)}"
                    }
                    client_socket.send(json.dumps(error_response).encode('utf-8'))

        except ConnectionResetError:
            print(f"Client {client_address} disconnected")
        except Exception as e:
            print(f"Error handling client {client_address}: {e}")
        finally:
            client_socket.close()
            print(f"Connection closed: {client_address}")

    def process_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process robot command and forward to STM32

        Args:
            command: Command dictionary from client

        Returns:
            Response dictionary
        """
        action = command.get('action', '')

        try:
            if action == 'move_forward':
                speed = command.get('speed', 50)
                success = self.stm32.move_forward(speed)

            elif action == 'move_backward':
                speed = command.get('speed', 50)
                success = self.stm32.move_backward(speed)

            elif action == 'turn_left':
                angle = command.get('angle', 800)
                speed = command.get('speed', 50)
                success = self.stm32.turn_left(angle, speed)

            elif action == 'turn_right':
                angle = command.get('angle', 800)
                speed = command.get('speed', 50)
                success = self.stm32.turn_right(angle, speed)

            elif action == 'stop':
                success = self.stm32.stop()

            elif action == 'pivot_left':
                angle = command.get('angle', 800)
                success = self.stm32.pivot_left(angle)

            elif action == 'pivot_right':
                angle = command.get('angle', 800)
                success = self.stm32.pivot_right(angle)

            elif action == 'status':
                # Return status
                return {
                    "status": "success",
                    "connected": self.stm32.connected,
                    "message": "Robot server operational"
                }

            else:
                return {
                    "status": "error",
                    "message": f"Unknown action: {action}"
                }

            # Return result
            if success:
                return {
                    "status": "success",
                    "action": action,
                    "message": f"Command {action} executed"
                }
            else:
                return {
                    "status": "error",
                    "action": action,
                    "message": f"Command {action} failed"
                }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Error processing command: {str(e)}"
            }

    def stop(self):
        """Stop the server"""
        print("\nStopping robot server...")
        self.running = False

        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass

        self.stm32.disconnect()
        print("Robot server stopped")


def parse_args():
    parser = argparse.ArgumentParser(
        description='RPI Robot Server - Forward commands to STM32')

    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Server host address (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8485,
                        help='Server port (default: 8485)')
    parser.add_argument('--serial_port', type=str, default='/dev/ttyUSB0',
                        help='STM32 serial port (default: /dev/ttyS0)')
    parser.add_argument('--baudrate', type=int, default=115200,
                        help='Serial baudrate (default: 115200)')

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    print("="*60)
    print("RPI Robot Server")
    print("="*60)
    print(f"Server: {args.host}:{args.port}")
    print(f"STM32: {args.serial_port} @ {args.baudrate} baud")
    print("="*60)

    # Create and start server
    server = RobotServer(
        host=args.host,
        port=args.port,
        stm32_port=args.serial_port
    )

    try:
        server.start()
    except KeyboardInterrupt:
        print("\nServer interrupted by user")
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        server.stop()


if __name__ == '__main__':
    main()
