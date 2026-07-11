/**
* This file is part of a hardware-accelerated fork of ORB-SLAM3
* (github.com/luett1/ORB_SLAM3) developed for the Master's thesis
* "On-device 3D Mapping" (M. Schieber, SDSU/UniBw, 2026). Licensed under
* GPLv3, like ORB-SLAM3 itself (see the original notice in ORBextractor.h).
*
* Public interface to the KR260 PL ORB feature-extraction accelerator.
* The whole userspace driver (uio/u-dma-buf device tables, AXI DMA and TOP
* register maps, transfer sequencing, keypoint decoding and the per-cell
* strict/permissive selection) lives in ORBHwAccelerator.cc; this header
* only exposes the one call ORBextractor needs per pyramid level.
*/

#ifndef ORBHWACCELERATOR_H
#define ORBHWACCELERATOR_H

#ifdef USE_HW_ACCEL

#include <cstdint>
#include <vector>

#include <opencv2/core/core.hpp>

namespace ORB_SLAM3
{
namespace orb_hw
{

// Per-level hardware stats returned by RunLevel (beyond the keypoints).
struct HwLevelStats
{
    double dmaResetUs;      // time spent waiting on the two DMA channel resets
    uint32_t stallCycles;   // STALLCNT: PL cycles the pixel stream was held off
                            // by corner-FIFO backpressure (0 unless corner-dense)
};

// Push one pyramid level image (CV_8UC1) through hardware accelerator
// instance `instanceIndex` (0 = left/mono/RGBD, 1 = right; folded modulo the
// number of instances in the build, see ORB_HW_NUM_ACCELERATORS). Returns the
// selected corners in the (absolute - minBorder) frame DistributeOctTree
// expects, with the hardware orientation already in kp.angle (so the HW path
// skips computeOrientation) and the ranking response in kp.response.
// `level` is only used for diagnostics. Thread-safe: each instance has its
// own lock, so left/right threads driving different instances run truly in
// parallel. Throws std::runtime_error on any device or protocol error.
HwLevelStats RunLevel(int instanceIndex, const cv::Mat& image,
                      std::vector<cv::KeyPoint>& vToDistributeKeys, int level);

}  // namespace orb_hw
}  // namespace ORB_SLAM3

#endif // USE_HW_ACCEL

#endif // ORBHWACCELERATOR_H
