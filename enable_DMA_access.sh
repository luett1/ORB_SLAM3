# Allow user-space ORB-SLAM3 hardware acceleration to access DMA control
# devices and shared DMA buffers for this boot session.
#
# Instance 0 (left / mono / RGBD) is always present. Instance 1 (right; F.12
# dual-accelerator) exists only when the dual overlay + udmabuf2/3 are loaded,
# so its chmods are guarded -- this script stays correct on single-instance builds.

# Instance 0: DMA regs (uio4), TOP regs (uio5), in/out buffers (udmabuf0/1).
sudo chmod a+rw /dev/uio4 /dev/uio5 /dev/udmabuf0 /dev/udmabuf1
sudo chmod a+rw /sys/class/u-dma-buf/udmabuf0/sync_* /sys/class/u-dma-buf/udmabuf1/sync_*

# Instance 1: DMA regs (uio8), TOP regs (uio9), in/out buffers (udmabuf2/3).
if [ -e /dev/uio8 ]; then
    sudo chmod a+rw /dev/uio8 /dev/uio9 /dev/udmabuf2 /dev/udmabuf3
    sudo chmod a+rw /sys/class/u-dma-buf/udmabuf2/sync_* /sys/class/u-dma-buf/udmabuf3/sync_*
fi

echo "ORBSLAM commands can now run without sudo"