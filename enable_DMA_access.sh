# Allow user-space ORB-SLAM3 hardware acceleration to access DMA control
# devices and shared DMA buffers for this boot session.
sudo chmod a+rw /dev/uio4 /dev/uio5 /dev/udmabuf0 /dev/udmabuf1
echo "ORBSLAM commands can now run without sudo"