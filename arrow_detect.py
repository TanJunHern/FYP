#!/usr/bin/env python3
"""
UDP Stream Arrow Detection
Receives video stream via UDP and displays arrow detection
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

# Import arrow detector (local copy)
from arrow_detector import ArrowDetector


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
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        # Dark semi-transparent background for better text visibility
        cv2.rectangle(img, c1, c2, (0, 0, 0), -1, cv2.LINE_AA)
        # Use specified text color (colored text on dark background)
        final_text_color = text_color if text_color is not None else color
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, final_text_color, thickness=tf, lineType=cv2.LINE_AA)


def parse_args():
    parser = argparse.ArgumentParser(
        description='UDP Stream Arrow Detection')

    parser.add_argument('--host', type=str, required=True,
                        help='UDP stream host IP address')
    parser.add_argument('--port', type=int, default=8485,
                        help='UDP stream port (default: 8485)')
    parser.add_argument("--show_fps",
                        help='if set, displays FPS counter',
                        action='store_true')
    parser.add_argument("--save_video",
                        help='if set, saves output video',
                        action='store_true')
    parser.add_argument("--output_path", type=str,
                        default='output_arrows.mp4',
                        help='path to save output video')

    return parser.parse_args()


def run_arrow_detection(args):
    """Run arrow detection on UDP stream"""

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

    # Setup video writer if saving
    video_writer = None
    if args.save_video:
        # Get frame size from first frame
        ret, frame = cap.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(args.output_path, fourcc, 10.0, (w, h))
            print(f"Saving output to: {args.output_path}")

    # FPS calculation
    fps_counter = []
    frame_count = 0

    print("\nStarting arrow detection...")
    print("Press 'q' to quit, 's' to save current frame")

    while cap.isOpened():
        start_time = time.time()

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

        # Draw arrow detections
        for arrow_info in arrow_info_list:
            xyxy = arrow_info['bbox']
            arrow_type = arrow_info['arrow_type']
            confidence = arrow_info['confidence']
            class_id = arrow_info['class_id']

            # task_2 model uses class names: 'Left', 'Right', 'Bullseye'
            # Extract the direction from the class name
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
            color = arrow_colors.get(color_key, [255, 255, 255])

            # Use the arrow color for both box and text
            plot_one_box(xyxy, frame, color=color, label=label, line_thickness=3, text_color=color)

        # Calculate FPS
        end_time = time.time()
        fps = 1.0 / (end_time - start_time)
        fps_counter.append(fps)
        if len(fps_counter) > 30:
            fps_counter.pop(0)
        avg_fps = np.mean(fps_counter)

        # Add FPS counter
        if args.show_fps:
            cv2.putText(frame, f"FPS: {avg_fps:.1f}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Add arrow count
        cv2.putText(frame, f"Arrows: {len(arrow_info_list)}",
                   (10, original_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Save video if enabled
        if video_writer is not None:
            video_writer.write(frame)

        # Display
        cv2.imshow('UDP Stream - Arrow Detection', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('s'):
            # Save current frame
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"arrow_frame_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved frame to: {filename}")

        # Print status every 30 frames
        if frame_count % 30 == 0:
            print(f"Frames processed: {frame_count} | FPS: {avg_fps:.1f} | Arrows detected: {len(arrow_info_list)}")

    # Cleanup
    print("\nCleaning up...")
    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
    print("Done!")


if __name__ == '__main__':
    args = parse_args()
    run_arrow_detection(args)
