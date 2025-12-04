# RPI Robot Server Setup Guide

This guide explains how to set up the RPI robot server to enable arrow_detect2.py on your MacBook to control the robot via RPI → STM32.

## Architecture

```
MacBook (arrow_detect2.py)
    ↓ TCP/IP (WiFi)
RPI (rpi_robot_server.py)
    ↓ Serial/UART
STM32 (Motor Controller)
    ↓
Robot Motors
```

## Step 1: Transfer Server to RPI

### Option A: Using SCP
```bash
# From your MacBook, in the monodepth2 directory
scp rpi_robot_server.py pi@192.168.2.1:~/
```

### Option B: Using rsync
```bash
rsync -avz rpi_robot_server.py pi@192.168.2.1:~/
```

### Option C: Manual Copy
1. Open the file in a text editor
2. Copy all contents
3. SSH to RPI: `ssh pi@192.168.2.1`
4. Create file: `nano rpi_robot_server.py`
5. Paste contents and save (Ctrl+X, Y, Enter)

## Step 2: Install Dependencies on RPI

SSH into your RPI:
```bash
ssh pi@192.168.2.1
```

Install required packages:
```bash
# Update package list
sudo apt-get update

# Install Python3 and pip (if not already installed)
sudo apt-get install python3 python3-pip

# Install pyserial for STM32 communication
pip3 install pyserial
```

## Step 3: Find Your STM32 Serial Port

Connect your STM32 to the RPI and find the serial port:

```bash
# List all USB serial devices
ls /dev/ttyUSB* /dev/ttyACM*

# Or use dmesg to see recently connected devices
dmesg | grep tty

# Common options:
# /dev/ttyUSB0  - USB-to-serial adapter
# /dev/ttyACM0  - STM32 USB CDC
# /dev/serial0  - RPI GPIO UART
```

## Step 4: Set Serial Port Permissions

Grant permission to access the serial port:

```bash
# Add your user to dialout group
sudo usermod -a -G dialout $USER

# Or change permissions directly (temporary)
sudo chmod 666 /dev/ttyUSB0  # Replace with your port

# Reboot for group changes to take effect
sudo reboot
```

## Step 5: Test STM32 Connection (Optional)

Test if STM32 is responding:

```bash
# Install screen if needed
sudo apt-get install screen

# Connect to serial port
screen /dev/ttyUSB0 115200

# Type commands and see if STM32 responds
# Press Ctrl+A then K to exit
```

## Step 6: Start the Robot Server on RPI

### Basic Start (default settings)
```bash
python3 rpi_robot_server.py
```

### With Custom Settings
```bash
# Specify serial port and baudrate
python3 rpi_robot_server.py --serial_port /dev/ttyACM0 --baudrate 115200

# Specify server port
python3 rpi_robot_server.py --port 8485

# Full example
python3 rpi_robot_server.py --host 0.0.0.0 --port 8485 --serial_port /dev/ttyUSB0 --baudrate 115200
```

### Run in Background (Systemd Service)

Create a systemd service for automatic startup:

```bash
# Create service file
sudo nano /etc/systemd/system/robot-server.service
```

Add this content (adjust paths):
```ini
[Unit]
Description=RPI Robot Server
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi
ExecStart=/usr/bin/python3 /home/pi/rpi_robot_server.py --serial_port /dev/ttyUSB0
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable robot-server
sudo systemctl start robot-server

# Check status
sudo systemctl status robot-server

# View logs
sudo journalctl -u robot-server -f
```

## Step 7: Run arrow_detect2.py on MacBook

From your MacBook:

```bash
cd /Users/tanjunhern/Documents/GitHub/monodepth2

# Run with robot control
python arrow_detect2.py --host 192.168.2.1 --robot_ip 192.168.2.1 --robot_port 8485 --show_fps
```

## STM32 Command Protocol

The server sends these commands to STM32 via serial:

| Command | Format | Example | Description |
|---------|--------|---------|-------------|
| Move Forward | `FW####` | `FW0050` | Speed 50 |
| Move Backward | `BW####` | `BW0050` | Speed 50 |
| Turn Left | `TL####,####` | `TL0800,0050` | Angle 800, Speed 50 |
| Turn Right | `TR####,####` | `TR0800,0050` | Angle 800, Speed 50 |
| Stop | `STOP` | `STOP` | Stop motors |
| Pivot Left | `PL####` | `PL0800` | Angle 800 |
| Pivot Right | `PR####` | `PR0800` | Angle 800 |

**Note:** Adjust the command format in the `STM32Controller` class if your STM32 uses a different protocol.

## Troubleshooting

### Connection Refused Error
```
Connection refused - RPI server not running on 192.168.2.1:8485
```

**Solutions:**
1. Make sure RPI server is running: `python3 rpi_robot_server.py`
2. Check firewall: `sudo ufw allow 8485` (if using firewall)
3. Verify RPI IP address: `hostname -I` on RPI
4. Test connection: `telnet 192.168.2.1 8485` from MacBook

### Serial Port Permission Denied
```
PermissionError: [Errno 13] Permission denied: '/dev/ttyUSB0'
```

**Solutions:**
1. Add user to dialout group: `sudo usermod -a -G dialout $USER`
2. Reboot: `sudo reboot`
3. Or: `sudo chmod 666 /dev/ttyUSB0`

### STM32 Not Responding
```
Failed to connect to STM32
```

**Solutions:**
1. Check cable connection
2. Verify serial port: `ls /dev/tty*`
3. Check baudrate matches STM32 (default: 115200)
4. Test with screen: `screen /dev/ttyUSB0 115200`

### Wrong Command Format

If your STM32 expects different commands, edit the `STM32Controller` class in `rpi_robot_server.py`:

```python
def move_forward(self, speed: int) -> bool:
    # Change this to match your STM32 protocol
    command = f"FW{speed:04d}"  # Modify format here
    return self.send_command(command)
```

## Testing the Setup

1. **Test RPI Server:**
   ```bash
   # On RPI
   python3 rpi_robot_server.py
   ```

2. **Test Connection from MacBook:**
   ```bash
   # On MacBook
   telnet 192.168.2.1 8485
   # Or use NetworkRobotClient test
   python -c "from network_robot_client import NetworkRobotClient; client = NetworkRobotClient('192.168.2.1', 8485); client.connect()"
   ```

3. **Test Full System:**
   ```bash
   # On MacBook
   python arrow_detect2.py --host 192.168.2.1 --robot_ip 192.168.2.1 --show_fps
   ```

## Command-Line Arguments

### rpi_robot_server.py (RPI)

| Argument | Default | Description |
|----------|---------|-------------|
| `--host` | 0.0.0.0 | Server host (0.0.0.0 = all interfaces) |
| `--port` | 8485 | Server TCP port |
| `--serial_port` | /dev/ttyUSB0 | STM32 serial port |
| `--baudrate` | 115200 | Serial baud rate |

### arrow_detect2.py (MacBook)

| Argument | Default | Description |
|----------|---------|-------------|
| `--host` | Required | UDP video stream IP |
| `--port` | 8485 | UDP stream port |
| `--robot_ip` | 192.168.2.1 | RPI robot server IP |
| `--robot_port` | 8485 | RPI robot server port |
| `--no_robot` | False | Run without robot control |
| `--show_fps` | False | Show FPS counter |

## System Overview

```
┌─────────────────────────────────────┐
│         MacBook                     │
│                                     │
│  arrow_detect2.py                   │
│  ├─ Arrow Detection (YOLO)          │
│  ├─ Centering Controller            │
│  └─ NetworkRobotClient              │
│      │                               │
│      │ TCP: Robot commands           │
│      ↓                               │
└──────────────────────────────────────┘
           │
           │ WiFi (192.168.2.x)
           ↓
┌─────────────────────────────────────┐
│         Raspberry Pi                │
│                                     │
│  rpi_robot_server.py                │
│  ├─ TCP Server (port 8485)          │
│  ├─ Command Parser                  │
│  └─ STM32Controller                 │
│      │                               │
│      │ Serial: Motor commands        │
│      ↓                               │
└──────────────────────────────────────┘
           │
           │ UART/USB
           ↓
┌─────────────────────────────────────┐
│         STM32                       │
│                                     │
│  Motor Control Firmware             │
│  └─ Drives robot motors             │
│                                     │
└──────────────────────────────────────┘
```

## Quick Start Checklist

- [ ] Copy `rpi_robot_server.py` to RPI
- [ ] Install pyserial on RPI: `pip3 install pyserial`
- [ ] Find STM32 serial port: `ls /dev/tty*`
- [ ] Set serial permissions: `sudo usermod -a -G dialout $USER`
- [ ] Start server on RPI: `python3 rpi_robot_server.py --serial_port /dev/ttyUSB0`
- [ ] Run on MacBook: `python arrow_detect2.py --host 192.168.2.1 --robot_ip 192.168.2.1 --show_fps`
- [ ] Point camera at arrow and watch it center!

## Additional Notes

- The server supports multiple simultaneous clients (one active connection at a time)
- Commands are sent at ~10Hz from arrow_detect2.py
- Emergency stop: Press SPACE key in arrow_detect2.py window
- The server will automatically reconnect if STM32 connection is lost
- Adjust `Kp`, `DEADBAND`, and `MAX_STEERING_ANGLE` in arrow_detect2.py for tuning
