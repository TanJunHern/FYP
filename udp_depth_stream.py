#!/usr/bin/env python3
"""
UDP Stream Depth Estimation for Monodepth2
Receives video stream via UDP and displays depth map alongside camera feed
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

import torch
from torchvision import transforms
import matplotlib as mpl
import matplotlib.cm as cm

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist


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

# To test diff models
def parse_args():
    parser = argparse.ArgumentParser(
        description='UDP Stream Depth Estimation using Monodepth2')

    parser.add_argument('--host', type=str, required=True,
                        help='UDP stream host IP address')
    parser.add_argument('--port', type=int, default=8485,
                        help='UDP stream port (default: 8485)')
    parser.add_argument('--model_name', type=str,
                        default='mono+stereo_640x192',
                        help='name of a pretrained model to use',
                        choices=[
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320"])
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    parser.add_argument("--show_fps",
                        help='if set, displays FPS counter',
                        action='store_true')
    parser.add_argument("--save_video",
                        help='if set, saves output video',
                        action='store_true')
    parser.add_argument("--output_path", type=str,
                        default='output_depth.mp4',
                        help='path to save output video')

    return parser.parse_args()


def run_depth_stream(args):
    """Run depth estimation on UDP stream"""

    # Setup device
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Download and load model
    print(f"Loading model: {args.model_name}")
    download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("models", args.model_name)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # Load encoder
    print("Loading pretrained encoder...")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    # Load depth decoder
    print("Loading pretrained decoder...")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))
    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)
    depth_decoder.to(device)
    depth_decoder.eval()

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
            # Output will be side-by-side: original + depth map
            video_writer = cv2.VideoWriter(args.output_path, fourcc, 10.0, (w * 2, h))
            print(f"Saving output to: {args.output_path}")

    # FPS calculation
    fps_counter = []
    frame_count = 0

    print("\nStarting depth estimation...")
    print("Press 'q' to quit, 's' to save current frame")

    with torch.no_grad():
        while cap.isOpened():
            start_time = time.time()

            ret, frame = cap.read()

            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            original_frame = frame.copy()

            # Prepare input for model
            input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_height, original_width = input_image.shape[:2]

            # Resize for model input
            input_image = cv2.resize(input_image, (feed_width, feed_height))
            input_tensor = transforms.ToTensor()(input_image).unsqueeze(0)
            input_tensor = input_tensor.to(device)

            # Run depth prediction
            features = encoder(input_tensor)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]

            # Resize disparity to original size
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Convert to numpy and create colormap
            disp_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_depth = (mapper.to_rgba(disp_np)[:, :, :3] * 255).astype(np.uint8)
            colormapped_depth = cv2.cvtColor(colormapped_depth, cv2.COLOR_RGB2BGR)

            # Calculate depth in meters (approximate)
            scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
            depth_meters = depth.squeeze().cpu().numpy()

            # Add distance annotation to depth map
            # Use the actual displayed depth map dimensions
            display_height, display_width = colormapped_depth.shape[:2]
            center_x = display_width // 2
            # Move crosshair down by 1/4 from center (to 3/4 from top)
            center_y = display_height // 2 + display_height // 4

            # Get depth value from depth_meters (need to map coordinates)
            depth_h, depth_w = depth_meters.shape
            depth_sample_y = int(center_y * depth_h / display_height)
            depth_sample_x = int(center_x * depth_w / display_width)
            center_depth = depth_meters[depth_sample_y, depth_sample_x]

            cv2.putText(colormapped_depth, f"Center: {center_depth:.2f}m",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Add crosshair (horizontally centered, down by 1/4)
            cv2.line(colormapped_depth, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 2)
            cv2.line(colormapped_depth, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 2)

            # Calculate FPS
            end_time = time.time()
            fps = 1.0 / (end_time - start_time)
            fps_counter.append(fps)
            if len(fps_counter) > 30:
                fps_counter.pop(0)
            avg_fps = np.mean(fps_counter)

            # Add FPS to original frame
            if args.show_fps:
                cv2.putText(original_frame, f"FPS: {avg_fps:.1f}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Add labels
            cv2.putText(original_frame, "Camera Feed",
                       (10, original_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(colormapped_depth, "Depth Map",
                       (10, original_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Combine side by side
            combined = np.hstack([original_frame, colormapped_depth])

            # Save video if enabled
            if video_writer is not None:
                video_writer.write(combined)

            # Display
            cv2.imshow('UDP Stream - Depth Estimation', combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"depth_frame_{timestamp}.jpg"
                cv2.imwrite(filename, combined)
                print(f"Saved frame to: {filename}")

            # Print status every 30 frames
            if frame_count % 30 == 0:
                print(f"Frames processed: {frame_count} | FPS: {avg_fps:.1f} | Center depth: {center_depth:.2f}m")

    # Cleanup
    print("\nCleaning up...")
    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()
    print("Done!")


if __name__ == '__main__':
    args = parse_args()
    run_depth_stream(args)
