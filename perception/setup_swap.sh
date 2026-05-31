#!/usr/bin/env bash
# setup_swap.sh -- Configure swap on Jetson Nano
#
# Jetson Nano ships with no swap by default. Without it, CPU allocations
# (ReID, ROS2, Python libs) have no overflow and OOM-killer fires instead.
#
# This script sets up two layers:
#   1. zram  -- compressed in-RAM swap (~2x effective RAM, fast, low latency)
#   2. swapfile -- disk-backed overflow (~4 GB on SD/NVMe, slow but deep)
#
# Run once on the Jetson:
#   chmod +x perception/setup_swap.sh && sudo ./perception/setup_swap.sh
set -euo pipefail

SWAPFILE=/var/swapfile
SWAPFILE_SIZE_GB=4

# ── Existing swap report ──────────────────────────────────────────────────────
echo "==> Current swap:"
free -h | grep -E "Mem|Swap"
echo ""

# ── 1. zram (NVIDIA ships a service for this) ─────────────────────────────────
if systemctl list-unit-files | grep -q nvzramconfig; then
    echo "==> Enabling NVIDIA zram service (nvzramconfig)"
    systemctl enable nvzramconfig
    systemctl start  nvzramconfig
    echo "    zram active"
else
    echo "==> nvzramconfig not found -- setting up zram manually"
    # Allocate half of physical RAM as compressed swap
    PHYS_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    ZRAM_BYTES=$(( PHYS_KB * 1024 / 2 ))

    modprobe zram
    echo "$ZRAM_BYTES" > /sys/block/zram0/disksize
    mkswap /dev/zram0
    swapon --priority 100 /dev/zram0
    echo "    zram0: $(( ZRAM_BYTES / 1024 / 1024 )) MB at priority 100"

    # Persist via rc.local
    cat >> /etc/rc.local <<'EOF'
modprobe zram
PHYS_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
echo $(( PHYS_KB * 1024 / 2 )) > /sys/block/zram0/disksize
mkswap /dev/zram0
swapon --priority 100 /dev/zram0
EOF
fi

# ── 2. Disk-backed swapfile (overflow) ───────────────────────────────────────
if swapon --show | grep -q "$SWAPFILE"; then
    echo "==> Swapfile $SWAPFILE already active, skipping"
else
    echo "==> Creating ${SWAPFILE_SIZE_GB} GB swapfile at $SWAPFILE"
    fallocate -l "${SWAPFILE_SIZE_GB}G" "$SWAPFILE" || \
        dd if=/dev/zero of="$SWAPFILE" bs=1M count=$(( SWAPFILE_SIZE_GB * 1024 )) status=progress
    chmod 600 "$SWAPFILE"
    mkswap "$SWAPFILE"
    swapon --priority 10 "$SWAPFILE"   # lower priority than zram -- disk is last resort
    echo "    $SWAPFILE: ${SWAPFILE_SIZE_GB} GB at priority 10"

    # Persist across reboots
    if ! grep -q "$SWAPFILE" /etc/fstab; then
        echo "$SWAPFILE swap swap defaults 0 0" >> /etc/fstab
    fi
fi

# ── 3. Tune swappiness ────────────────────────────────────────────────────────
# Default is 60 (aggressive). 80 tells the kernel to prefer swap over OOM-kill,
# which is what we want -- we'd rather be slow than crash.
echo "==> Setting vm.swappiness=80"
sysctl -w vm.swappiness=80
if ! grep -q "vm.swappiness" /etc/sysctl.conf; then
    echo "vm.swappiness=80" >> /etc/sysctl.conf
fi

# ── Report ────────────────────────────────────────────────────────────────────
echo ""
echo "==> Done. Swap layout:"
swapon --show
echo ""
free -h | grep -E "Mem|Swap"
echo ""
echo "  zram  -- fast compressed swap (fits in RAM, ~2x density)"
echo "  disk  -- slow overflow on SD/NVMe (deep, last resort)"
echo ""
echo "  ReID and other CPU allocations can now spill to swap."
echo "  CUDA allocations (YOLO) remain pinned -- keep YOLO_IMGSZ=416"
echo "  and use a TRT engine (python export_trt.py) to keep CUDA footprint low."
