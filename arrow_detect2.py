#!/usr/bin/env python3
"""
UDP Stream Arrow Detection with Robot Centering Control
Receives video stream via UDP and keeps detected arrows centered
"""

from __future__ import absolute_import, division, print_function

import os
import sys
import argparse
import numpy as np
import cv2
import threading
import socket
import pickle
import struct
import time

# Import arrow detector and robot client (local copies)
from arrow_detector import ArrowDetector
from network_robot_client import NetworkRobotClient


class ArrowCenteringController:
    """Controller to keep detected arrow centered in camera frame"""

    def __init__(self, image_width=1280, image_height=720):
        """
        Initialize arrow centering controller

        Args:
            image_width: Width of camera frame
            image_height: Height of camera frame
        """
        # Control parameters
        self.Kp = 0.15  # Proportional gain (very slight corrections)
        self.DEADBAND_PIXELS = 40  # Tolerance zone (pixels)
        self.MAX_STEERING_ANGLE = 200  # Very slight steering (200 out of 1000)
        self.BASE_SPEED = 50  # Base forward speed

        # Frame dimensions
        self.image_width = image_width
        self.image_height = image_height
        self.center_x = image_width // 2
        self.center_y = image_height // 2

        # State tracking
        self.last_error = 0
        self.no_arrow_count = 0

        print(f"Arrow Centering Controller initialized:")
        print(f"  Kp: {self.Kp}")
        print(f"  Deadband: {self.DEADBAND_PIXELS} pixels")
        print(f"  Max steering angle: {self.MAX_STEERING_ANGLE}")
        print(f"  Base speed: {self.BASE_SPEED}")

    def calculate_control(self, arrow_bbox):
        """
        Calculate steering control to center arrow

        Args:
            arrow_bbox: [x1, y1, x2, y2] bounding box of arrow

        Returns:
            (steering_angle, speed, status_text)
        """
        if arrow_bbox is None:
            # No arrow detected - continue straight
            self.no_arrow_count += 1
            return 0, self.BASE_SPEED, "NO ARROW - STRAIGHT"

        # Reset no-arrow counter
        self.no_arrow_count = 0

        # Calculate arrow center
        arrow_center_x = (arrow_bbox[0] + arrow_bbox[2]) / 2

        # Calculate horizontal error (positive = arrow is to the right)
        error_x = arrow_center_x - self.center_x

        # Apply deadband (tolerance zone)
        if abs(error_x) < self.DEADBAND_PIXELS:
            # Arrow is centered - go straight
            self.last_error = 0
            return 0, self.BASE_SPEED, "CENTERED"

        # Calculate proportional steering
        # Positive error = arrow right -> turn right
        # Negative error = arrow left -> turn left
        steering = self.Kp * error_x

        # Clamp to maximum steering angle
        steering = np.clip(steering, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

        # Store error for next iteration
        self.last_error = error_x

        # Determine status
        if error_x > self.DEADBAND_PIXELS:
            status = f"ADJUST RIGHT ({error_x:.0f}px)"
        else:
            status = f"ADJUST LEFT ({error_x:.0f}px)"

        return int(steering), self.BASE_SPEED, status


class UDPStreamCapture:
    def __init__(self, host, port=8485, timeout=5.0):
        """UDP video stream receiver for RPI camera"""
        self.host = host
        self.port = port
        self.frame = None
        self.ret = False
        self.running = True

        # Create UDP socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 131072)

        # Connect to server
        self.server_address = (host, port)
        print(f"[UDP] Connecting to {host}:{port}...")

        try:
            self.socket.settimeout(timeout)
            # Send connection request
            for attempt in range(3):
                self.socket.sendto(b'CONNECT', self.server_address)
                try:
                    data, _ = self.socket.recvfrom(1024)
                    if data == b'ACK':
                        print("[UDP] Connected to stream server")
                        break
                except socket.timeout:
                    if attempt < 2:
                        print(f"[UDP] Retry {attempt + 2}/3...")
                    continue
            else:
                raise Exception("Connection timeout")

            self.socket.settimeout(2.0)

            # Start receiving thread
            self.thread = threading.Thread(target=self._update_frame)
            self.thread.daemon = True
            self.thread.start()

        except Exception as e:
            print(f"[UDP] Connection failed: {e}")
            self.ret = False

    def _update_frame(self):
        """Continuously receive frames from UDP stream"""
        timeout_count = 0
        MAX_TIMEOUTS = 10

        try:
            while self.running:
                try:
                    # Receive frame header
                    header_data, _ = self.socket.recvfrom(6)

                    if len(header_data) != 6:
                        continue

                    frame_size, num_chunks = struct.unpack('!LH', header_data)

                    # Sanity check
                    if num_chunks > 100 or num_chunks == 0:
                        continue

                    # Receive all chunks
                    chunks = {}
                    for _ in range(num_chunks):
                        chunk_data, _ = self.socket.recvfrom(65535)
                        if len(chunk_data) < 2:
                            continue
                        chunk_id = struct.unpack('!H', chunk_data[:2])[0]
                        chunks[chunk_id] = chunk_data[2:]

                    # Check if we got all chunks
                    if len(chunks) != num_chunks:
                        continue

                    # Reassemble frame
                    try:
                        frame_data = b''.join([chunks[i] for i in range(num_chunks)])
                    except KeyError:
                        continue

                    # Deserialize and decode
                    try:
                        encoded_frame = pickle.loads(frame_data)
                        frame = cv2.imdecode(encoded_frame, cv2.IMREAD_COLOR)

                        if frame is not None:
                            self.frame = frame
                            self.ret = True
                            timeout_count = 0
                    except (pickle.UnpicklingError, EOFError):
                        continue

                except socket.timeout:
                    timeout_count += 1
                    if timeout_count >= MAX_TIMEOUTS:
                        print("[UDP] Stream lost")
                        self.ret = False
                        break
                    continue

        except Exception as e:
            print(f"[UDP] Stream error: {e}")
            self.ret = False

    def read(self):
        return self.ret, self.frame

    def isOpened(self):
        return self.running and self.ret

    def release(self):
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.socket.close()


def plot_one_box(xyxy, img, color=None, label=None, line_thickness=2, text_color=None):
    """Plot one bounding box on image"""
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [255, 0, 0]
    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 6, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        # Dark semi-transparent background for better text visibility
        cv2.rectangle(img, c1, c2, (0, 0, 0), -1, cv2.LINE_AA)
        # Use specified text color (colored text on dark background)
        final_text_color = text_color if text_color is not None else color
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 6, final_text_color, thickness=tf, lineType=cv2.LINE_AA)


def draw_centering_overlay(frame, center_x, center_y, deadband, arrow_bbox=None, is_centered=False):
    """
    Draw crosshair and tolerance zone overlay

    Args:
        frame: Image to draw on
        center_x: Center X coordinate
        center_y: Center Y coordinate
        deadband: Tolerance zone size in pixels
        arrow_bbox: Current arrow bounding box (if any)
        is_centered: Whether arrow is currently centered
    """
    # Draw crosshair at center
    crosshair_size = 30
    crosshair_color = (0, 255, 0) if is_centered else (0, 255, 255)  # Green if centered, Yellow otherwise

    # Horizontal line
    cv2.line(frame, (center_x - crosshair_size, center_y),
             (center_x + crosshair_size, center_y), crosshair_color, 2)
    # Vertical line
    cv2.line(frame, (center_x, center_y - crosshair_size),
             (center_x, center_y + crosshair_size), crosshair_color, 2)

    # Draw tolerance zone (rectangle around center)
    zone_color = (0, 255, 0) if is_centered else (255, 165, 0)  # Green if centered, Orange otherwise
    zone_x1 = center_x - deadband
    zone_x2 = center_x + deadband
    zone_y1 = int(center_y - deadband * 0.75)  # Make it slightly rectangular
    zone_y2 = int(center_y + deadband * 0.75)

    cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), zone_color, 2)

    # Draw line from arrow center to screen center (if arrow exists)
    if arrow_bbox is not None:
        arrow_center_x = int((arrow_bbox[0] + arrow_bbox[2]) / 2)
        arrow_center_y = int((arrow_bbox[1] + arrow_bbox[3]) / 2)

        # Draw line
        line_color = (0, 255, 0) if is_centered else (0, 0, 255)  # Green if centered, Red otherwise
        cv2.line(frame, (arrow_center_x, arrow_center_y),
                 (center_x, center_y), line_color, 2)

        # Draw small circle at arrow center
        cv2.circle(frame, (arrow_center_x, arrow_center_y), 5, (255, 0, 255), -1)


def parse_args():
    parser = argparse.ArgumentParser(
        description='UDP Stream Arrow Detection with Robot Centering')

    parser.add_argument('--host', type=str, required=True,
                        help='UDP stream host IP address')
    parser.add_argument('--port', type=int, default=8485,
                        help='UDP stream port (default: 8485)')
    parser.add_argument('--robot_ip', type=str, default='192.168.2.1',
                        help='Robot RPI IP address (default: 192.168.2.1)')
    parser.add_argument('--robot_port', type=int, default=8888,
                        help='Robot RPI port (default: 8888)')
    parser.add_argument('--no_robot',
                        help='Run without robot control (visualization only)',
                        action='store_true')
    parser.add_argument("--show_fps",
                        help='if set, displays FPS counter',
                        action='store_true')
    parser.add_argument("--save_video",
                        help='if set, saves output video',
                        action='store_true')
    parser.add_argument("--output_path", type=str,
                        default='output_arrow_centering.mp4',
                        help='path to save output video')

    return parser.parse_args()


def run_arrow_centering(args):
    """Run arrow detection with centering control on UDP stream"""

    # Load arrow detector with same model as main2.py
    print("Loading arrow detection model...")
    arrow_detector = ArrowDetector(
        '/Users/tanjunhern/Documents/GitHub/pyson3/SC2079-MDP-Group-29/image_recognition/runs/detect/train task_2/weights/best.pt',
        optimize_for_distance=True
    )
    print("Arrow detector loaded successfully!")

    # Arrow colors - Left: orange, Right: green
    arrow_colors = {
        'Left': [0, 165, 255],    # Orange
        'Right': [0, 255, 0],     # Green
        'Bullseye': [0, 255, 255] # Yellow
    }

    # Connect to UDP stream
    print(f"\nConnecting to UDP stream at {args.host}:{args.port}")
    cap = UDPStreamCapture(args.host, args.port)

    # Wait for first frame
    timeout = 10
    start_time = time.time()
    while not cap.ret and (time.time() - start_time) < timeout:
        time.sleep(0.1)

    if not cap.ret:
        print("Failed to connect to stream!")
        sys.exit(1)

    print("Stream connected successfully!")

    # Get frame dimensions
    ret, first_frame = cap.read()
    if ret and first_frame is not None:
        frame_height, frame_width = first_frame.shape[:2]
    else:
        frame_width, frame_height = 1280, 720  # Default

    # Initialize centering controller
    controller = ArrowCenteringController(frame_width, frame_height)

    # Initialize robot client (if enabled)
    robot = None
    if not args.no_robot:
        try:
            print(f"\nConnecting to robot at {args.robot_ip}:{args.robot_port}")
            robot = NetworkRobotClient(rpi_ip=args.robot_ip, rpi_port=args.robot_port)
            if robot.connect():
                print("Robot connected successfully!")
            else:
                print("Failed to connect to robot - continuing in visualization mode")
                robot = None
        except Exception as e:
            print(f"Robot connection error: {e}")
            print("Continuing in visualization mode")
            robot = None
    else:
        print("\nRunning in visualization-only mode (no robot control)")

    # Setup video writer if saving
    video_writer = None
    if args.save_video:
        if ret and first_frame is not None:
            h, w = first_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(args.output_path, fourcc, 10.0, (w, h))
            print(f"Saving output to: {args.output_path}")

    # FPS calculation
    fps_counter = []
    frame_count = 0

    print("\nStarting arrow centering control...")
    print("Press 'q' to quit, 's' to save current frame, 'SPACE' to emergency stop")

    try:
        while cap.isOpened():
            loop_start_time = time.time()

            ret, frame = cap.read()

            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            original_height, original_width = frame.shape[:2]

            # Detect arrows using same settings as main2.py
            arrow_detections = arrow_detector.detect_arrows(frame, enhanced=True, detection_size=1280)
            # pyson3 task_2 model has correct labels - do NOT swap
            arrow_info_list = arrow_detector.get_arrow_info(
                arrow_detections,
                apply_confidence_boost=False,
                swap_left_right=False
            )

            # Find the largest/closest arrow for centering
            target_arrow = None
            target_bbox = None

            if len(arrow_info_list) > 0:
                # Select arrow with largest area (closest)
                largest_area = 0
                for arrow_info in arrow_info_list:
                    xyxy = arrow_info['bbox']
                    area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
                    if area > largest_area:
                        largest_area = area
                        target_arrow = arrow_info
                        target_bbox = xyxy

            # Calculate control based on target arrow
            steering, speed, status = controller.calculate_control(target_bbox)

            # Send robot command
            if robot:
                if steering == 0:
                    # Go straight
                    robot.move_forward(speed)
                elif steering > 0:
                    # Turn right (arrow is to the right)
                    robot.turn_right(abs(steering), speed)
                else:
                    # Turn left (arrow is to the left)
                    robot.turn_left(abs(steering), speed)

            # Draw all detected arrows
            for arrow_info in arrow_info_list:
                xyxy = arrow_info['bbox']
                arrow_type = arrow_info['arrow_type']
                confidence = arrow_info['confidence']

                # Determine if this is the target arrow
                is_target = (target_arrow is not None and
                           arrow_info['bbox'] == target_arrow['bbox'])

                # task_2 model uses class names: 'Left', 'Right', 'Bullseye'
                if 'left' in arrow_type.lower():
                    actual_direction = "LEFT"
                    color_key = "Left"
                elif 'right' in arrow_type.lower():
                    actual_direction = "RIGHT"
                    color_key = "Right"
                elif 'bullseye' in arrow_type.lower():
                    actual_direction = "BULLSEYE"
                    color_key = "Bullseye"
                else:
                    actual_direction = arrow_type.upper()
                    color_key = arrow_type

                label = f"{actual_direction} ({confidence:.2f})"
                if is_target:
                    label += " [TRACKING]"

                color = arrow_colors.get(color_key, [255, 255, 255])

                # Highlight target arrow with thicker border
                thickness = 5 if is_target else 3

                plot_one_box(xyxy, frame, color=color, label=label,
                           line_thickness=thickness, text_color=color)

            # Draw centering overlay
            is_centered = (status == "CENTERED")
            draw_centering_overlay(frame, controller.center_x, controller.center_y,
                                 controller.DEADBAND_PIXELS, target_bbox, is_centered)

            # Calculate FPS
            end_time = time.time()
            fps = 1.0 / (end_time - loop_start_time)
            fps_counter.append(fps)
            if len(fps_counter) > 30:
                fps_counter.pop(0)
            avg_fps = np.mean(fps_counter)

            # Add status overlay
            status_y = 30
            if args.show_fps:
                cv2.putText(frame, f"FPS: {avg_fps:.1f}",
                           (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                status_y += 30

            # Control status
            status_color = (0, 255, 0) if is_centered else (0, 165, 255)
            cv2.putText(frame, f"Status: {status}",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            status_y += 30

            # Steering info
            if robot:
                steering_text = f"Steering: {steering:+4d} | Speed: {speed}"
                cv2.putText(frame, steering_text,
                           (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                status_y += 30

            # Arrow count
            cv2.putText(frame, f"Arrows: {len(arrow_info_list)}",
                       (10, original_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Save video if enabled
            if video_writer is not None:
                video_writer.write(frame)

            # Display
            cv2.imshow('Arrow Centering Control', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"centering_frame_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved frame to: {filename}")
            elif key == ord(' '):
                # Emergency stop
                if robot:
                    robot.stop()
                    print("EMERGENCY STOP!")

            # Print status every 30 frames
            if frame_count % 30 == 0:
                print(f"Frames: {frame_count} | FPS: {avg_fps:.1f} | {status} | Steering: {steering:+4d}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Cleanup
        print("\nCleaning up...")
        if robot:
            robot.stop()
            robot.disconnect()
        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        print("Done!")


if __name__ == '__main__':
    args = parse_args()
    run_arrow_centering(args)
