#!/usr/bin/env bash
# setup_ros2.sh -- Install ROS2 jazzy on Ubuntu 22.04 WSL2 (Windows 11)
#
# Run once inside your WSL terminal:
#   chmod +x perception/setup_ros2.sh && ./perception/setup_ros2.sh
#
# After install, attach your USB camera from a Windows terminal (PowerShell admin):
#   usbipd list                        # find camera BUSID (e.g. 2-3)
#   usbipd bind --busid 2-3
#   usbipd attach --wsl --busid 2-3
# Then verify in WSL:
#   ls /dev/video*
set -euo pipefail

# ── 1. Verify Ubuntu 22.04 ────────────────────────────────────────────────────
# . /etc/os-release
# if [[ "$VERSION_ID" != "22.04" ]]; then
#     echo "ERROR: This script targets Ubuntu 22.04 (detected $PRETTY_NAME)."
#     echo "For Ubuntu 24.04, replace 'humble' with 'jazzy' throughout."
#     exit 1
# fi
# echo "==> Ubuntu $VERSION_ID OK"

# ── 2. Add ROS2 apt repository ────────────────────────────────────────────────
echo "==> Adding ROS2 apt repository"
sudo apt install -y software-properties-common curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu jammy main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# ── 3. Install ROS2 jazzy + perception deps ─────────────────────────────────
echo "==> Installing ROS2 jazzy (this may take a few minutes)"
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
    ros-jazzy-desktop \
    ros-jazzy-cv-bridge \
    ros-jazzy-image-transport \
    python3-rosdep \
    python3-colcon-common-extensions \
    v4l-utils

# ── 4. Initialize rosdep ──────────────────────────────────────────────────────
echo "==> Initializing rosdep"
sudo rosdep init 2>/dev/null || echo "(rosdep already initialized)"
rosdep update

# ── 5. Source ROS2 in shell profile ──────────────────────────────────────────
SETUP_LINE="source /opt/ros/jazzy/setup.bash"
if ! grep -qxF "$SETUP_LINE" ~/.bashrc; then
    echo "$SETUP_LINE" >> ~/.bashrc
    echo "==> Added ROS2 source to ~/.bashrc"
fi

# ── 6. Install usbipd-win hint ───────────────────────────────────────────────
echo ""
echo "=========================================================="
echo " ROS2 jazzy installed successfully."
echo "=========================================================="
echo ""
echo " Next steps:"
echo ""
echo " 1. Restart your WSL shell (or run): source ~/.bashrc"
echo ""
echo " 2. Install usbipd-win on Windows to pass the USB camera to WSL:"
echo "      https://github.com/dorssel/usbipd-win/releases"
echo "    Then in a Windows PowerShell (Administrator):"
echo "      usbipd list"
echo "      usbipd bind --busid <BUSID>"
echo "      usbipd attach --wsl --busid <BUSID>"
echo "    Verify in WSL:  ls /dev/video*"
echo ""
echo " 3. Run the camera node (in one terminal):"
echo "      source /opt/ros/jazzy/setup.bash"
echo "      cd perception && python camera_node.py"
echo ""
echo " 4. Run the perception node (in another terminal):"
echo "      source /opt/ros/jazzy/setup.bash"
echo "      cd perception && python rosmain.py"
echo ""
echo " 5. Inspect topics:"
echo "      ros2 topic list"
echo "      ros2 run rqt_image_view rqt_image_view  # view annotated feed"
echo ""
