# Allow user-space ORB-SLAM3 hardware acceleration to access DMA control
# devices and shared DMA buffers for this boot session.
sudo chmod a+rw /dev/uio4 /dev/uio5 /dev/udmabuf0 /dev/udmabuf1
sudo chmod a+rw /sys/class/u-dma-buf/udmabuf0/sync_* /sys/class/u-dma-buf/udmabuf1/sync_*
echo "ORBSLAM commands can now run without sudo"