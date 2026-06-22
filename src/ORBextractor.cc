/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

/**
* Software License Agreement (BSD License)
*
*  Copyright (c) 2009, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
*/


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <vector>
#include <iostream>

#include "ORBextractor.h"

//includes for timing
#include <chrono>
#include <fstream>

#ifdef USE_HW_ACCEL
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#endif

using namespace cv;
using namespace std;

namespace ORB_SLAM3
{

    const int PATCH_SIZE = 31;
    const int HALF_PATCH_SIZE = 15;
    const int EDGE_THRESHOLD = 19;

#ifdef USE_HW_ACCEL
namespace {

    const char* kDmaDev = "/dev/uio4";
    const char* kTopDev = "/dev/uio5";
    const char* kInDev = "/dev/udmabuf0";
    const char* kOutDev = "/dev/udmabuf1";
    const char* kInPhysPath = "/sys/class/u-dma-buf/udmabuf0/phys_addr";
    const char* kOutPhysPath = "/sys/class/u-dma-buf/udmabuf1/phys_addr";
    const char* kInSizePath = "/sys/class/u-dma-buf/udmabuf0/size";
    const char* kOutSizePath = "/sys/class/u-dma-buf/udmabuf1/size";
    const char* kInSyncDir = "/sys/class/u-dma-buf/udmabuf0";

    const size_t kDmaMapSize = 0x10000;
    const size_t kTopMapSize = 0x1000;
    const size_t kHwInputMapBytes = 512 * 1024;
    const size_t kHwOutputMapBytes = 256 * 1024;
    const size_t kRecordBytes = 16;
    const uint32_t kExpectedTopId = 0xC0DE0001u;

namespace dma {
    const uint32_t MM2S_DMACR = 0x00;
    const uint32_t MM2S_DMASR = 0x04;
    const uint32_t MM2S_SA = 0x18;
    const uint32_t MM2S_SA_MSB = 0x1C;
    const uint32_t MM2S_LENGTH = 0x28;

    const uint32_t S2MM_DMACR = 0x30;
    const uint32_t S2MM_DMASR = 0x34;
    const uint32_t S2MM_DA = 0x48;
    const uint32_t S2MM_DA_MSB = 0x4C;
    const uint32_t S2MM_LENGTH = 0x58;

    const uint32_t DmacrRunStop = 0x00000001;
    const uint32_t DmacrReset = 0x00000004;
    const uint32_t DmacrIocIrqEn = 0x00001000;
    const uint32_t DmasrIocIrq = 0x00001000;
    const uint32_t DmasrErrMask = 0x00000070;
}  // namespace dma

namespace top {
    const uint32_t CTRL = 0x00;
    const uint32_t STATUS = 0x04;
    const uint32_t WIDTH = 0x08;
    const uint32_t HEIGHT = 0x0C;
    const uint32_t KPCOUNT = 0x14;
    const uint32_t DROPCNT = 0x18;
    const uint32_t ID = 0x1C;

    const uint32_t CtrlEnable = 0x00000001;
    const uint32_t CtrlSoftReset = 0x00000002;
    const uint32_t StatusBusy = 0x00000001;
    const uint32_t StatusDone = 0x00000002;
    const uint32_t StatusOverflow = 0x00000004;
    const uint32_t StatusCfgError = 0x00000008;
}  // namespace top

    class Fd
    {
    public:
        explicit Fd(const char* path, int flags) : fd_(::open(path, flags))
        {
            if(fd_ < 0)
                throw runtime_error(string("open failed for ") + path + ": " + strerror(errno));
        }

        ~Fd()
        {
            if(fd_ >= 0)
                ::close(fd_);
        }

        Fd(const Fd&) = delete;
        Fd& operator=(const Fd&) = delete;

        int get() const { return fd_; }

    private:
        int fd_;
    };

    class Mapping
    {
    public:
        Mapping() : data_(nullptr), size_(0) {}

        ~Mapping()
        {
            if(data_)
                ::munmap(data_, size_);
        }

        Mapping(const Mapping&) = delete;
        Mapping& operator=(const Mapping&) = delete;

        void map(int fd, size_t size)
        {
            void* ptr = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            if(ptr == MAP_FAILED)
                throw runtime_error(string("mmap failed: ") + strerror(errno));

            data_ = static_cast<uint8_t*>(ptr);
            size_ = size;
        }

        uint8_t* bytes() { return data_; }
        const uint8_t* bytes() const { return data_; }
        size_t size() const { return size_; }

        uint32_t read32(uint32_t byteOffset) const
        {
            const volatile uint32_t* regs = reinterpret_cast<const volatile uint32_t*>(data_);
            return regs[byteOffset / 4];
        }

        void write32(uint32_t byteOffset, uint32_t value)
        {
            volatile uint32_t* regs = reinterpret_cast<volatile uint32_t*>(data_);
            regs[byteOffset / 4] = value;
        }

    private:
        uint8_t* data_;
        size_t size_;
    };

    // u-dma-buf manual cache control via sysfs. The buffers are mapped CACHEABLE
    // (opened without O_SYNC) and the DMA path is non-coherent, so the CPU must
    // flush before the DMA reads (sync_for_device) and invalidate before reading
    // what the DMA wrote (sync_for_cpu). Works because sync_mode=1, dma_coherent=0.
    class UdmabufSync
    {
    public:
        explicit UdmabufSync(const string& dir) : base_(dir + "/") {}

        void forDevice(uint64_t size)   // flush CPU cache -> DDR, before MM2S
        {
            writeAttr("sync_offset", 0);
            writeAttr("sync_size", size);
            writeAttr("sync_direction", 1);   // DMA_TO_DEVICE
            writeAttr("sync_for_device", 1);
        }

        void forCpu(uint64_t size)      // invalidate CPU cache, after S2MM completes
        {
            writeAttr("sync_offset", 0);
            writeAttr("sync_size", size);
            writeAttr("sync_direction", 2);   // DMA_FROM_DEVICE
            writeAttr("sync_for_cpu", 1);
        }

    private:
        void writeAttr(const char* attr, uint64_t value)
        {
            ofstream out(base_ + attr);
            if(!out || !(out << value))
                throw runtime_error("u-dma-buf sysfs write failed: " + base_ + attr);
        }
        string base_;
    };

    uint64_t ReadU64Auto(const string& path)
    {
        ifstream input(path.c_str());
        if(!input)
            throw runtime_error("failed to open " + path);

        string value;
        input >> value;
        return stoull(value, nullptr, 0);
    }

    string Hex32(uint32_t value)
    {
        ostringstream stream;
        stream << "0x" << hex << value;
        return stream.str();
    }

    double WaitForDmaResetClear(const Mapping& regs, uint32_t dmacrOffset,
                                const char* channelName)
    {
        const chrono::steady_clock::time_point start = chrono::steady_clock::now();
        uint32_t dmacr = regs.read32(dmacrOffset);
        while((dmacr & dma::DmacrReset) != 0)
        {
            const chrono::steady_clock::time_point now = chrono::steady_clock::now();
            if(now - start > chrono::milliseconds(10))
                throw runtime_error(string("AXI DMA reset timed out: ") + channelName +
                                    " DMACR=" + Hex32(dmacr));
            this_thread::yield();
            dmacr = regs.read32(dmacrOffset);
        }

        const chrono::steady_clock::time_point end = chrono::steady_clock::now();
        return chrono::duration<double, std::micro>(end - start).count();
    }

    void WaitForDmaRunStopAccepted(const Mapping& regs, uint32_t dmacrOffset,
                                   const char* channelName)
    {
        const chrono::steady_clock::time_point start = chrono::steady_clock::now();
        uint32_t dmacr = regs.read32(dmacrOffset);
        while((dmacr & dma::DmacrRunStop) == 0)
        {
            const chrono::steady_clock::time_point now = chrono::steady_clock::now();
            if(now - start > chrono::milliseconds(10))
                throw runtime_error(string("AXI DMA run/stop not accepted: ") + channelName +
                                    " DMACR=" + Hex32(dmacr));
            this_thread::yield();
            dmacr = regs.read32(dmacrOffset);
        }
    }

    void PulseTopSoftReset(Mapping& regs)
    {
        const chrono::steady_clock::time_point start = chrono::steady_clock::now();

        regs.write32(top::CTRL, top::CtrlSoftReset);
        uint32_t ctrl = regs.read32(top::CTRL);
        while((ctrl & top::CtrlSoftReset) == 0)
        {
            const chrono::steady_clock::time_point now = chrono::steady_clock::now();
            if(now - start > chrono::milliseconds(10))
                throw runtime_error("TOP soft reset assert timed out: CTRL=" + Hex32(ctrl));
            this_thread::yield();
            ctrl = regs.read32(top::CTRL);
        }

        regs.write32(top::CTRL, 0);
        while(true)
        {
            ctrl = regs.read32(top::CTRL);
            const uint32_t status = regs.read32(top::STATUS);
            const bool resetCleared = (ctrl & top::CtrlSoftReset) == 0;
            const bool statusCleared =
                (status & (top::StatusBusy | top::StatusDone |
                           top::StatusOverflow | top::StatusCfgError)) == 0;
            if(resetCleared && statusCleared)
                return;

            const chrono::steady_clock::time_point now = chrono::steady_clock::now();
            if(now - start > chrono::milliseconds(10))
                throw runtime_error("TOP soft reset clear timed out: CTRL=" + Hex32(ctrl) +
                                    " STATUS=" + Hex32(status));
            this_thread::yield();
        }
    }

    uint16_t ReadLe16(const uint8_t* data)
    {
        return static_cast<uint16_t>(data[0]) |
               static_cast<uint16_t>(static_cast<uint16_t>(data[1]) << 8);
    }

    uint32_t ReadLe32(const uint8_t* data)
    {
        return static_cast<uint32_t>(data[0]) |
               (static_cast<uint32_t>(data[1]) << 8) |
               (static_cast<uint32_t>(data[2]) << 16) |
               (static_cast<uint32_t>(data[3]) << 24);
    }

    void FillMappedBytes(uint8_t* dst, size_t len, uint8_t value)
    {
        volatile uint8_t* p = dst;
        for(size_t i = 0; i < len; ++i)
            p[i] = value;
    }

    void CopyImageToMapped(uint8_t* dst, const Mat& image)
    {
        // Plain memcpy is now legal+fast: the buffer is mapped CACHEABLE (Normal
        // memory), not the non-cacheable Device mapping that made memcpy SIGBUS.
        // Coherency with the DMA is handled by the explicit forDevice() flush.
        const size_t cols = static_cast<size_t>(image.cols);
        if(image.isContinuous())
            memcpy(dst, image.data, cols * static_cast<size_t>(image.rows));
        else
            for(int row = 0; row < image.rows; ++row)
                memcpy(dst + static_cast<size_t>(row) * cols, image.ptr<uchar>(row), cols);
    }

    float HardwareAngleToDegrees(int32_t angleQ24)
    {
        float degrees = static_cast<float>(angleQ24) * (360.0f / 8388608.0f);
        if(degrees < 0.0f)
            degrees += 360.0f;
        if(degrees >= 360.0f)
            degrees -= 360.0f;
        return degrees;
    }

    // One PL keypoint as delivered for a single pyramid level (raw, absolute coords).
    struct PLKeypoint
    {
        float    x;             // level-image absolute coords (window center)
        float    y;
        uint16_t score;         // FAST response
        int32_t  angleQ24;      // raw 24b sign-extended HW angle
        bool     is_brighter;
        bool     passed_strict; // score >= iniThFAST(20)
    };

    // PS-side per-cell strict/permissive filter == ORB-SLAM3's per-cell double-FAST.
    // The PL detects at THRESHOLD_PERMISSIVE(7) over the whole image and tags every
    // corner with passed_strict (score>=20). Per W=35 cell: if ANY corner is strict,
    // keep only the strict ones (== FAST at iniThFAST); else keep all permissive
    // (== the empty-cell minThFAST fallback). Validated bit-exact vs OpenCV FAST
    // (TYPE_9_16) in select_per_cell.cpp. Without this the octree was fed the full
    // permissive flood, producing weak, non-repeatable keypoints (see F.10).
    // Output is in the (absolute - minBorder) frame DistributeOctTree expects, with
    // the HW angle carried in kp.angle (so the HW path skips computeOrientation).
    vector<KeyPoint> SelectPerCellHW(int cols, int rows, const vector<PLKeypoint>& plKps)
    {
        const int   EDGE        = EDGE_THRESHOLD;   // 19
        const float W           = 35.0f;
        const int   FAST_BORDER = 3;                // empirically confirmed for TYPE_9_16

        // identical to ComputeKeyPointsOctTree
        const int minBorderX = EDGE - 3;
        const int minBorderY = minBorderX;
        const int maxBorderX = cols - EDGE + 3;
        const int maxBorderY = rows - EDGE + 3;

        const float width  = static_cast<float>(maxBorderX - minBorderX);
        const float height = static_cast<float>(maxBorderY - minBorderY);
        const int nCols = static_cast<int>(width  / W);
        const int nRows = static_cast<int>(height / W);
        const int wCell = static_cast<int>(ceil(width  / nCols));
        const int hCell = static_cast<int>(ceil(height / nRows));

        vector<KeyPoint> vToDistributeKeys;
        vToDistributeKeys.reserve(plKps.size());

        // Each PL corner lands in at most one cell (regions are contiguous; the
        // 'used' guard is defensive against the 6px cell overlap).
        vector<char> used(plKps.size(), 0);

        for(int i = 0; i < nRows; ++i)
        {
            const int iniY = minBorderY + i * hCell;
            int maxY = iniY + hCell + 6;
            if(iniY >= maxBorderY - 3) continue;
            if(maxY > maxBorderY) maxY = maxBorderY;

            for(int j = 0; j < nCols; ++j)
            {
                const int iniX = minBorderX + j * wCell;
                int maxX = iniX + wCell + 6;
                if(iniX >= maxBorderX - 6) continue;
                if(maxX > maxBorderX) maxX = maxBorderX;

                // Cell detection region in absolute level coords.
                const float xLo = static_cast<float>(iniX + FAST_BORDER);
                const float xHi = static_cast<float>(maxX - FAST_BORDER - 1);
                const float yLo = static_cast<float>(iniY + FAST_BORDER);
                const float yHi = static_cast<float>(maxY - FAST_BORDER - 1);

                vector<size_t> cellIdx;
                bool anyStrict = false;
                for(size_t k = 0; k < plKps.size(); ++k)
                {
                    if(used[k]) continue;
                    const PLKeypoint& p = plKps[k];
                    if(p.x >= xLo && p.x <= xHi && p.y >= yLo && p.y <= yHi)
                    {
                        cellIdx.push_back(k);
                        if(p.passed_strict) anyStrict = true;
                    }
                }
                if(cellIdx.empty()) continue;

                // if-any-strict-else-all  (== iniThFAST else minThFAST fallback)
                for(size_t k : cellIdx)
                {
                    const PLKeypoint& p = plKps[k];
                    if(anyStrict && !p.passed_strict) continue;
                    used[k] = 1;

                    KeyPoint kp;
                    kp.pt.x     = p.x - minBorderX;
                    kp.pt.y     = p.y - minBorderY;
                    kp.angle    = HardwareAngleToDegrees(p.angleQ24);  // HW orientation
                    kp.response = static_cast<float>(p.score);
                    vToDistributeKeys.push_back(kp);
                }
            }
        }
        return vToDistributeKeys;
    }

    class OrbHwAccelerator
    {
    public:
        OrbHwAccelerator()
            : dmaFd_(kDmaDev, O_RDWR | O_SYNC),    // MMIO regs: uncached
              topFd_(kTopDev, O_RDWR | O_SYNC),    // MMIO regs: uncached
              inFd_(kInDev, O_RDWR),               // input buffer: CACHEABLE (fast memcpy)
              outFd_(kOutDev, O_RDWR | O_SYNC),    // output buffer: uncached (small reads, always correct)
              inPhys_(ReadU64Auto(kInPhysPath)),
              outPhys_(ReadU64Auto(kOutPhysPath)),
              inSize_(static_cast<size_t>(ReadU64Auto(kInSizePath))),
              outSize_(static_cast<size_t>(ReadU64Auto(kOutSizePath))),
              inMapSize_(min(inSize_, kHwInputMapBytes)),
              outMapSize_(min(outSize_, kHwOutputMapBytes))
        {
            if(inMapSize_ == 0 || outMapSize_ < kRecordBytes)
                throw runtime_error("invalid u-dma-buf size");
            if((outMapSize_ % kRecordBytes) != 0)
                throw runtime_error("hardware output map size must be a multiple of 16 bytes");
            if(inPhys_ > 0xFFFFFFFFull || outPhys_ > 0xFFFFFFFFull)
                throw runtime_error("u-dma-buf physical address is above the 32-bit DMA range");

            dmaRegs_.map(dmaFd_.get(), kDmaMapSize);
            topRegs_.map(topFd_.get(), kTopMapSize);
            inMap_.map(inFd_.get(), inMapSize_);
            outMap_.map(outFd_.get(), outMapSize_);

            const uint32_t topId = topRegs_.read32(top::ID);
            if(topId != kExpectedTopId)
                throw runtime_error("unexpected TOP build ID: " + Hex32(topId));
        }

        double RunLevel(const Mat& image, vector<KeyPoint>& rawKeypoints, int level,
                        int minBorderX, int maxBorderX, int minBorderY, int maxBorderY)
        {
            lock_guard<mutex> lock(mutex_);

            if(image.type() != CV_8UC1)
                throw runtime_error("hardware accelerator expects CV_8UC1 images");

            const size_t inLen = static_cast<size_t>(image.cols) * static_cast<size_t>(image.rows);
            const size_t outLen = outMapSize_;
            if(inLen > inMapSize_)
                throw runtime_error("pyramid level exceeds mapped udmabuf0 size");
            if(outLen < kRecordBytes)
                throw runtime_error("mapped udmabuf1 is too small for one output record");

#ifdef USE_HW_ACCEL_TRACE
            if(traceCount_ < 20)
                cerr << "[USE_HW_ACCEL] begin level=" << level
                     << ", image=" << image.cols << "x" << image.rows
                     << ", inLen=" << inLen
                     << ", outLen=" << outLen << endl;
#endif

            PulseTopSoftReset(topRegs_);

            dmaRegs_.write32(dma::MM2S_DMACR, dma::DmacrReset);
            dmaRegs_.write32(dma::S2MM_DMACR, dma::DmacrReset);
            const double dmaResetUs =
                WaitForDmaResetClear(dmaRegs_, dma::MM2S_DMACR, "MM2S") +
                WaitForDmaResetClear(dmaRegs_, dma::S2MM_DMACR, "S2MM");

#ifdef USE_HW_ACCEL_TRACE
            if(traceCount_ < 20)
                cerr << "[USE_HW_ACCEL] reset done level=" << level << endl;
#endif

            CopyImageToMapped(inMap_.bytes(), image);
            inSync_.forDevice(inLen);   // flush CPU cache -> DDR before MM2S reads it

#ifdef USE_HW_ACCEL_TRACE
            if(traceCount_ < 20)
                cerr << "[USE_HW_ACCEL] input copied level=" << level << endl;
#endif

            topRegs_.write32(top::WIDTH, static_cast<uint32_t>(image.cols));
            topRegs_.write32(top::HEIGHT, static_cast<uint32_t>(image.rows));

            dmaRegs_.write32(dma::S2MM_DMACR, dma::DmacrRunStop | dma::DmacrIocIrqEn);
            dmaRegs_.write32(dma::MM2S_DMACR, dma::DmacrRunStop | dma::DmacrIocIrqEn);
            WaitForDmaRunStopAccepted(dmaRegs_, dma::S2MM_DMACR, "S2MM");
            WaitForDmaRunStopAccepted(dmaRegs_, dma::MM2S_DMACR, "MM2S");

            dmaRegs_.write32(dma::S2MM_DA, static_cast<uint32_t>(outPhys_));
            dmaRegs_.write32(dma::S2MM_DA_MSB, 0);
            dmaRegs_.write32(dma::S2MM_LENGTH, static_cast<uint32_t>(outLen));

            topRegs_.write32(top::CTRL, top::CtrlEnable);

            dmaRegs_.write32(dma::MM2S_SA, static_cast<uint32_t>(inPhys_));
            dmaRegs_.write32(dma::MM2S_SA_MSB, 0);
            dmaRegs_.write32(dma::MM2S_LENGTH, static_cast<uint32_t>(inLen));

#ifdef USE_HW_ACCEL_TRACE
            if(traceCount_ < 20)
                cerr << "[USE_HW_ACCEL] transfer started level=" << level << endl;
#endif

            const chrono::steady_clock::time_point start = chrono::steady_clock::now();
            bool dmaError = false;
            bool timedOut = false;
            uint32_t errorMm2sStatus = 0;
            uint32_t errorS2mmStatus = 0;
            while(true)
            {
                const uint32_t mm2sStatus = dmaRegs_.read32(dma::MM2S_DMASR);
                const uint32_t s2mmStatus = dmaRegs_.read32(dma::S2MM_DMASR);
                const uint32_t topStatus = topRegs_.read32(top::STATUS);

                if(((mm2sStatus | s2mmStatus) & dma::DmasrErrMask) != 0)
                {
                    dmaError = true;
                    errorMm2sStatus = mm2sStatus;
                    errorS2mmStatus = s2mmStatus;
                    break;
                }

                const bool mm2sDone = (mm2sStatus & dma::DmasrIocIrq) != 0;
                const bool s2mmDone = (s2mmStatus & dma::DmasrIocIrq) != 0;
                const bool topDone = (topStatus & top::StatusDone) != 0;
                if(mm2sDone && s2mmDone && topDone)
                    break;

                if(chrono::steady_clock::now() - start > chrono::seconds(2))
                {
                    timedOut = true;
                    break;
                }

                this_thread::sleep_for(chrono::microseconds(50));
            }

            const uint32_t finalStatus = topRegs_.read32(top::STATUS);
            const uint32_t kpCount = topRegs_.read32(top::KPCOUNT);
#ifdef USE_HW_ACCEL_TRACE
            const uint32_t dropCount = topRegs_.read32(top::DROPCNT);
#endif

            topRegs_.write32(top::CTRL, 0);

            if(dmaError)
                throw runtime_error("AXI DMA error: MM2S=" + Hex32(errorMm2sStatus) +
                                    " S2MM=" + Hex32(errorS2mmStatus));
            if(timedOut)
                throw runtime_error("hardware accelerator timed out");
            if((finalStatus & top::StatusCfgError) != 0)
                throw runtime_error("TOP cfg_error set");
            if((finalStatus & top::StatusOverflow) != 0)
                throw runtime_error("TOP overflow set");
#ifdef USE_HW_ACCEL_TRACE
            if(dropCount != 0)
            {
                if(dropWarningCount_ < 20)
                {
                    cerr << "[USE_HW_ACCEL] warning: TOP DROPCNT=" << dropCount
                         << ", KPCOUNT=" << kpCount
                         << ", level=" << level
                         << ", image=" << image.cols << "x" << image.rows << endl;
                    if(dropWarningCount_ == 19)
                        cerr << "[USE_HW_ACCEL] suppressing further DROPCNT warnings" << endl;
                }
                ++dropWarningCount_;
            }
#endif

            const size_t sentinelOffset = static_cast<size_t>(kpCount) * kRecordBytes;
            if(sentinelOffset + kRecordBytes > outLen)
                throw runtime_error("TOP KPCOUNT exceeds udmabuf1 capacity");

            const uint8_t* sentinel = outMap_.bytes() + sentinelOffset;
            if(ReadLe32(sentinel) != 0 || ReadLe32(sentinel + 4) != 0 ||
               ReadLe32(sentinel + 8) != 0 || ReadLe32(sentinel + 12) != 0xFFFFFFFFu)
                throw runtime_error("TOP EOF sentinel mismatch");

            // SelectPerCellHW recomputes the borders from cols/rows (same formula as
            // ComputeKeyPointsOctTree), so the passed border args are now redundant.
            (void)minBorderX; (void)maxBorderX; (void)minBorderY; (void)maxBorderY;

            // Decode every PL keypoint in RAW absolute level coords -- NO shift, NO
            // border filter (SelectPerCellHW bins by W=35 cell and applies the shift).
            // Crucially, read the passed_strict flag (record byte 6, bit 1): the old
            // path discarded it and fed the octree the full permissive flood (F.10).
            vector<PLKeypoint> plKps;
            plKps.reserve(kpCount);

            for(uint32_t i = 0; i < kpCount; ++i)
            {
                const uint8_t* record = outMap_.bytes() + static_cast<size_t>(i) * kRecordBytes;
                const uint16_t x = ReadLe16(record);
                const uint16_t y = ReadLe16(record + 2);
                const uint16_t score = ReadLe16(record + 4);
                const uint8_t flags = record[6];
                const int32_t angleQ24 = static_cast<int32_t>(ReadLe32(record + 8));
                const uint32_t marker = ReadLe32(record + 12);

                if(marker != 0)
                    throw runtime_error("nonzero marker in TOP keypoint record");

                PLKeypoint p;
                p.x = static_cast<float>(x);
                p.y = static_cast<float>(y);
                p.score = score;
                p.angleQ24 = angleQ24;
                p.is_brighter = (flags & 0x01) != 0;   // word1[48]
                p.passed_strict = (flags & 0x02) != 0;  // word1[49] == score>=THRESHOLD_STRICT
                plKps.push_back(p);
            }

            // Per-cell strict/permissive selection (== ORB-SLAM3 per-cell double-FAST).
            rawKeypoints = SelectPerCellHW(image.cols, image.rows, plKps);

#ifdef USE_HW_ACCEL_TRACE
            if(traceCount_ < 20)
                cerr << "[USE_HW_ACCEL] done level=" << level
                     << ", KPCOUNT=" << kpCount
                     << ", kept=" << rawKeypoints.size()
                     << ", DROPCNT=" << dropCount << endl;
            ++traceCount_;
#endif
            return dmaResetUs;
        }

    private:
        Fd dmaFd_;
        Fd topFd_;
        Fd inFd_;
        Fd outFd_;
        uint64_t inPhys_;
        uint64_t outPhys_;
        size_t inSize_;
        size_t outSize_;
        size_t inMapSize_;
        size_t outMapSize_;
        Mapping dmaRegs_;
        Mapping topRegs_;
        Mapping inMap_;
        Mapping outMap_;
        UdmabufSync inSync_{kInSyncDir};
        mutex mutex_;
#ifdef USE_HW_ACCEL_TRACE
        uint32_t dropWarningCount_ = 0;
        uint32_t traceCount_ = 0;
#endif
    };

    OrbHwAccelerator& GetOrbHwAccelerator()
    {
        static OrbHwAccelerator accelerator;
        return accelerator;
    }

}  // namespace
#endif

    static float IC_Angle(const Mat& image, Point2f pt,  const vector<int> & u_max)
    {
        int m_01 = 0, m_10 = 0;

        const uchar* center = &image.at<uchar> (cvRound(pt.y), cvRound(pt.x));

        // Treat the center line differently, v=0
        for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
            m_10 += u * center[u];

        // Go line by line in the circuI853lar patch
        int step = (int)image.step1();
        for (int v = 1; v <= HALF_PATCH_SIZE; ++v)
        {
            // Proceed over the two lines
            int v_sum = 0;
            int d = u_max[v];
            for (int u = -d; u <= d; ++u)
            {
                int val_plus = center[u + v*step], val_minus = center[u - v*step];
                v_sum += (val_plus - val_minus);
                m_10 += u * (val_plus + val_minus);
            }
            m_01 += v * v_sum;
        }

        return fastAtan2((float)m_01, (float)m_10);
    }


    const float factorPI = (float)(CV_PI/180.f);
    static void computeOrbDescriptor(const KeyPoint& kpt,
                                     const Mat& img, const Point* pattern,
                                     uchar* desc)
    {
        float angle = (float)kpt.angle*factorPI;
        float a = (float)cos(angle), b = (float)sin(angle);

        const uchar* center = &img.at<uchar>(cvRound(kpt.pt.y), cvRound(kpt.pt.x));
        const int step = (int)img.step;

#define GET_VALUE(idx) \
        center[cvRound(pattern[idx].x*b + pattern[idx].y*a)*step + \
               cvRound(pattern[idx].x*a - pattern[idx].y*b)]


        for (int i = 0; i < 32; ++i, pattern += 16)
        {
            int t0, t1, val;
            t0 = GET_VALUE(0); t1 = GET_VALUE(1);
            val = t0 < t1;
            t0 = GET_VALUE(2); t1 = GET_VALUE(3);
            val |= (t0 < t1) << 1;
            t0 = GET_VALUE(4); t1 = GET_VALUE(5);
            val |= (t0 < t1) << 2;
            t0 = GET_VALUE(6); t1 = GET_VALUE(7);
            val |= (t0 < t1) << 3;
            t0 = GET_VALUE(8); t1 = GET_VALUE(9);
            val |= (t0 < t1) << 4;
            t0 = GET_VALUE(10); t1 = GET_VALUE(11);
            val |= (t0 < t1) << 5;
            t0 = GET_VALUE(12); t1 = GET_VALUE(13);
            val |= (t0 < t1) << 6;
            t0 = GET_VALUE(14); t1 = GET_VALUE(15);
            val |= (t0 < t1) << 7;

            desc[i] = (uchar)val;
        }

#undef GET_VALUE
    }


    static int bit_pattern_31_[256*4] =
            {
                    8,-3, 9,5/*mean (0), correlation (0)*/,
                    4,2, 7,-12/*mean (1.12461e-05), correlation (0.0437584)*/,
                    -11,9, -8,2/*mean (3.37382e-05), correlation (0.0617409)*/,
                    7,-12, 12,-13/*mean (5.62303e-05), correlation (0.0636977)*/,
                    2,-13, 2,12/*mean (0.000134953), correlation (0.085099)*/,
                    1,-7, 1,6/*mean (0.000528565), correlation (0.0857175)*/,
                    -2,-10, -2,-4/*mean (0.0188821), correlation (0.0985774)*/,
                    -13,-13, -11,-8/*mean (0.0363135), correlation (0.0899616)*/,
                    -13,-3, -12,-9/*mean (0.121806), correlation (0.099849)*/,
                    10,4, 11,9/*mean (0.122065), correlation (0.093285)*/,
                    -13,-8, -8,-9/*mean (0.162787), correlation (0.0942748)*/,
                    -11,7, -9,12/*mean (0.21561), correlation (0.0974438)*/,
                    7,7, 12,6/*mean (0.160583), correlation (0.130064)*/,
                    -4,-5, -3,0/*mean (0.228171), correlation (0.132998)*/,
                    -13,2, -12,-3/*mean (0.00997526), correlation (0.145926)*/,
                    -9,0, -7,5/*mean (0.198234), correlation (0.143636)*/,
                    12,-6, 12,-1/*mean (0.0676226), correlation (0.16689)*/,
                    -3,6, -2,12/*mean (0.166847), correlation (0.171682)*/,
                    -6,-13, -4,-8/*mean (0.101215), correlation (0.179716)*/,
                    11,-13, 12,-8/*mean (0.200641), correlation (0.192279)*/,
                    4,7, 5,1/*mean (0.205106), correlation (0.186848)*/,
                    5,-3, 10,-3/*mean (0.234908), correlation (0.192319)*/,
                    3,-7, 6,12/*mean (0.0709964), correlation (0.210872)*/,
                    -8,-7, -6,-2/*mean (0.0939834), correlation (0.212589)*/,
                    -2,11, -1,-10/*mean (0.127778), correlation (0.20866)*/,
                    -13,12, -8,10/*mean (0.14783), correlation (0.206356)*/,
                    -7,3, -5,-3/*mean (0.182141), correlation (0.198942)*/,
                    -4,2, -3,7/*mean (0.188237), correlation (0.21384)*/,
                    -10,-12, -6,11/*mean (0.14865), correlation (0.23571)*/,
                    5,-12, 6,-7/*mean (0.222312), correlation (0.23324)*/,
                    5,-6, 7,-1/*mean (0.229082), correlation (0.23389)*/,
                    1,0, 4,-5/*mean (0.241577), correlation (0.215286)*/,
                    9,11, 11,-13/*mean (0.00338507), correlation (0.251373)*/,
                    4,7, 4,12/*mean (0.131005), correlation (0.257622)*/,
                    2,-1, 4,4/*mean (0.152755), correlation (0.255205)*/,
                    -4,-12, -2,7/*mean (0.182771), correlation (0.244867)*/,
                    -8,-5, -7,-10/*mean (0.186898), correlation (0.23901)*/,
                    4,11, 9,12/*mean (0.226226), correlation (0.258255)*/,
                    0,-8, 1,-13/*mean (0.0897886), correlation (0.274827)*/,
                    -13,-2, -8,2/*mean (0.148774), correlation (0.28065)*/,
                    -3,-2, -2,3/*mean (0.153048), correlation (0.283063)*/,
                    -6,9, -4,-9/*mean (0.169523), correlation (0.278248)*/,
                    8,12, 10,7/*mean (0.225337), correlation (0.282851)*/,
                    0,9, 1,3/*mean (0.226687), correlation (0.278734)*/,
                    7,-5, 11,-10/*mean (0.00693882), correlation (0.305161)*/,
                    -13,-6, -11,0/*mean (0.0227283), correlation (0.300181)*/,
                    10,7, 12,1/*mean (0.125517), correlation (0.31089)*/,
                    -6,-3, -6,12/*mean (0.131748), correlation (0.312779)*/,
                    10,-9, 12,-4/*mean (0.144827), correlation (0.292797)*/,
                    -13,8, -8,-12/*mean (0.149202), correlation (0.308918)*/,
                    -13,0, -8,-4/*mean (0.160909), correlation (0.310013)*/,
                    3,3, 7,8/*mean (0.177755), correlation (0.309394)*/,
                    5,7, 10,-7/*mean (0.212337), correlation (0.310315)*/,
                    -1,7, 1,-12/*mean (0.214429), correlation (0.311933)*/,
                    3,-10, 5,6/*mean (0.235807), correlation (0.313104)*/,
                    2,-4, 3,-10/*mean (0.00494827), correlation (0.344948)*/,
                    -13,0, -13,5/*mean (0.0549145), correlation (0.344675)*/,
                    -13,-7, -12,12/*mean (0.103385), correlation (0.342715)*/,
                    -13,3, -11,8/*mean (0.134222), correlation (0.322922)*/,
                    -7,12, -4,7/*mean (0.153284), correlation (0.337061)*/,
                    6,-10, 12,8/*mean (0.154881), correlation (0.329257)*/,
                    -9,-1, -7,-6/*mean (0.200967), correlation (0.33312)*/,
                    -2,-5, 0,12/*mean (0.201518), correlation (0.340635)*/,
                    -12,5, -7,5/*mean (0.207805), correlation (0.335631)*/,
                    3,-10, 8,-13/*mean (0.224438), correlation (0.34504)*/,
                    -7,-7, -4,5/*mean (0.239361), correlation (0.338053)*/,
                    -3,-2, -1,-7/*mean (0.240744), correlation (0.344322)*/,
                    2,9, 5,-11/*mean (0.242949), correlation (0.34145)*/,
                    -11,-13, -5,-13/*mean (0.244028), correlation (0.336861)*/,
                    -1,6, 0,-1/*mean (0.247571), correlation (0.343684)*/,
                    5,-3, 5,2/*mean (0.000697256), correlation (0.357265)*/,
                    -4,-13, -4,12/*mean (0.00213675), correlation (0.373827)*/,
                    -9,-6, -9,6/*mean (0.0126856), correlation (0.373938)*/,
                    -12,-10, -8,-4/*mean (0.0152497), correlation (0.364237)*/,
                    10,2, 12,-3/*mean (0.0299933), correlation (0.345292)*/,
                    7,12, 12,12/*mean (0.0307242), correlation (0.366299)*/,
                    -7,-13, -6,5/*mean (0.0534975), correlation (0.368357)*/,
                    -4,9, -3,4/*mean (0.099865), correlation (0.372276)*/,
                    7,-1, 12,2/*mean (0.117083), correlation (0.364529)*/,
                    -7,6, -5,1/*mean (0.126125), correlation (0.369606)*/,
                    -13,11, -12,5/*mean (0.130364), correlation (0.358502)*/,
                    -3,7, -2,-6/*mean (0.131691), correlation (0.375531)*/,
                    7,-8, 12,-7/*mean (0.160166), correlation (0.379508)*/,
                    -13,-7, -11,-12/*mean (0.167848), correlation (0.353343)*/,
                    1,-3, 12,12/*mean (0.183378), correlation (0.371916)*/,
                    2,-6, 3,0/*mean (0.228711), correlation (0.371761)*/,
                    -4,3, -2,-13/*mean (0.247211), correlation (0.364063)*/,
                    -1,-13, 1,9/*mean (0.249325), correlation (0.378139)*/,
                    7,1, 8,-6/*mean (0.000652272), correlation (0.411682)*/,
                    1,-1, 3,12/*mean (0.00248538), correlation (0.392988)*/,
                    9,1, 12,6/*mean (0.0206815), correlation (0.386106)*/,
                    -1,-9, -1,3/*mean (0.0364485), correlation (0.410752)*/,
                    -13,-13, -10,5/*mean (0.0376068), correlation (0.398374)*/,
                    7,7, 10,12/*mean (0.0424202), correlation (0.405663)*/,
                    12,-5, 12,9/*mean (0.0942645), correlation (0.410422)*/,
                    6,3, 7,11/*mean (0.1074), correlation (0.413224)*/,
                    5,-13, 6,10/*mean (0.109256), correlation (0.408646)*/,
                    2,-12, 2,3/*mean (0.131691), correlation (0.416076)*/,
                    3,8, 4,-6/*mean (0.165081), correlation (0.417569)*/,
                    2,6, 12,-13/*mean (0.171874), correlation (0.408471)*/,
                    9,-12, 10,3/*mean (0.175146), correlation (0.41296)*/,
                    -8,4, -7,9/*mean (0.183682), correlation (0.402956)*/,
                    -11,12, -4,-6/*mean (0.184672), correlation (0.416125)*/,
                    1,12, 2,-8/*mean (0.191487), correlation (0.386696)*/,
                    6,-9, 7,-4/*mean (0.192668), correlation (0.394771)*/,
                    2,3, 3,-2/*mean (0.200157), correlation (0.408303)*/,
                    6,3, 11,0/*mean (0.204588), correlation (0.411762)*/,
                    3,-3, 8,-8/*mean (0.205904), correlation (0.416294)*/,
                    7,8, 9,3/*mean (0.213237), correlation (0.409306)*/,
                    -11,-5, -6,-4/*mean (0.243444), correlation (0.395069)*/,
                    -10,11, -5,10/*mean (0.247672), correlation (0.413392)*/,
                    -5,-8, -3,12/*mean (0.24774), correlation (0.411416)*/,
                    -10,5, -9,0/*mean (0.00213675), correlation (0.454003)*/,
                    8,-1, 12,-6/*mean (0.0293635), correlation (0.455368)*/,
                    4,-6, 6,-11/*mean (0.0404971), correlation (0.457393)*/,
                    -10,12, -8,7/*mean (0.0481107), correlation (0.448364)*/,
                    4,-2, 6,7/*mean (0.050641), correlation (0.455019)*/,
                    -2,0, -2,12/*mean (0.0525978), correlation (0.44338)*/,
                    -5,-8, -5,2/*mean (0.0629667), correlation (0.457096)*/,
                    7,-6, 10,12/*mean (0.0653846), correlation (0.445623)*/,
                    -9,-13, -8,-8/*mean (0.0858749), correlation (0.449789)*/,
                    -5,-13, -5,-2/*mean (0.122402), correlation (0.450201)*/,
                    8,-8, 9,-13/*mean (0.125416), correlation (0.453224)*/,
                    -9,-11, -9,0/*mean (0.130128), correlation (0.458724)*/,
                    1,-8, 1,-2/*mean (0.132467), correlation (0.440133)*/,
                    7,-4, 9,1/*mean (0.132692), correlation (0.454)*/,
                    -2,1, -1,-4/*mean (0.135695), correlation (0.455739)*/,
                    11,-6, 12,-11/*mean (0.142904), correlation (0.446114)*/,
                    -12,-9, -6,4/*mean (0.146165), correlation (0.451473)*/,
                    3,7, 7,12/*mean (0.147627), correlation (0.456643)*/,
                    5,5, 10,8/*mean (0.152901), correlation (0.455036)*/,
                    0,-4, 2,8/*mean (0.167083), correlation (0.459315)*/,
                    -9,12, -5,-13/*mean (0.173234), correlation (0.454706)*/,
                    0,7, 2,12/*mean (0.18312), correlation (0.433855)*/,
                    -1,2, 1,7/*mean (0.185504), correlation (0.443838)*/,
                    5,11, 7,-9/*mean (0.185706), correlation (0.451123)*/,
                    3,5, 6,-8/*mean (0.188968), correlation (0.455808)*/,
                    -13,-4, -8,9/*mean (0.191667), correlation (0.459128)*/,
                    -5,9, -3,-3/*mean (0.193196), correlation (0.458364)*/,
                    -4,-7, -3,-12/*mean (0.196536), correlation (0.455782)*/,
                    6,5, 8,0/*mean (0.1972), correlation (0.450481)*/,
                    -7,6, -6,12/*mean (0.199438), correlation (0.458156)*/,
                    -13,6, -5,-2/*mean (0.211224), correlation (0.449548)*/,
                    1,-10, 3,10/*mean (0.211718), correlation (0.440606)*/,
                    4,1, 8,-4/*mean (0.213034), correlation (0.443177)*/,
                    -2,-2, 2,-13/*mean (0.234334), correlation (0.455304)*/,
                    2,-12, 12,12/*mean (0.235684), correlation (0.443436)*/,
                    -2,-13, 0,-6/*mean (0.237674), correlation (0.452525)*/,
                    4,1, 9,3/*mean (0.23962), correlation (0.444824)*/,
                    -6,-10, -3,-5/*mean (0.248459), correlation (0.439621)*/,
                    -3,-13, -1,1/*mean (0.249505), correlation (0.456666)*/,
                    7,5, 12,-11/*mean (0.00119208), correlation (0.495466)*/,
                    4,-2, 5,-7/*mean (0.00372245), correlation (0.484214)*/,
                    -13,9, -9,-5/*mean (0.00741116), correlation (0.499854)*/,
                    7,1, 8,6/*mean (0.0208952), correlation (0.499773)*/,
                    7,-8, 7,6/*mean (0.0220085), correlation (0.501609)*/,
                    -7,-4, -7,1/*mean (0.0233806), correlation (0.496568)*/,
                    -8,11, -7,-8/*mean (0.0236505), correlation (0.489719)*/,
                    -13,6, -12,-8/*mean (0.0268781), correlation (0.503487)*/,
                    2,4, 3,9/*mean (0.0323324), correlation (0.501938)*/,
                    10,-5, 12,3/*mean (0.0399235), correlation (0.494029)*/,
                    -6,-5, -6,7/*mean (0.0420153), correlation (0.486579)*/,
                    8,-3, 9,-8/*mean (0.0548021), correlation (0.484237)*/,
                    2,-12, 2,8/*mean (0.0616622), correlation (0.496642)*/,
                    -11,-2, -10,3/*mean (0.0627755), correlation (0.498563)*/,
                    -12,-13, -7,-9/*mean (0.0829622), correlation (0.495491)*/,
                    -11,0, -10,-5/*mean (0.0843342), correlation (0.487146)*/,
                    5,-3, 11,8/*mean (0.0929937), correlation (0.502315)*/,
                    -2,-13, -1,12/*mean (0.113327), correlation (0.48941)*/,
                    -1,-8, 0,9/*mean (0.132119), correlation (0.467268)*/,
                    -13,-11, -12,-5/*mean (0.136269), correlation (0.498771)*/,
                    -10,-2, -10,11/*mean (0.142173), correlation (0.498714)*/,
                    -3,9, -2,-13/*mean (0.144141), correlation (0.491973)*/,
                    2,-3, 3,2/*mean (0.14892), correlation (0.500782)*/,
                    -9,-13, -4,0/*mean (0.150371), correlation (0.498211)*/,
                    -4,6, -3,-10/*mean (0.152159), correlation (0.495547)*/,
                    -4,12, -2,-7/*mean (0.156152), correlation (0.496925)*/,
                    -6,-11, -4,9/*mean (0.15749), correlation (0.499222)*/,
                    6,-3, 6,11/*mean (0.159211), correlation (0.503821)*/,
                    -13,11, -5,5/*mean (0.162427), correlation (0.501907)*/,
                    11,11, 12,6/*mean (0.16652), correlation (0.497632)*/,
                    7,-5, 12,-2/*mean (0.169141), correlation (0.484474)*/,
                    -1,12, 0,7/*mean (0.169456), correlation (0.495339)*/,
                    -4,-8, -3,-2/*mean (0.171457), correlation (0.487251)*/,
                    -7,1, -6,7/*mean (0.175), correlation (0.500024)*/,
                    -13,-12, -8,-13/*mean (0.175866), correlation (0.497523)*/,
                    -7,-2, -6,-8/*mean (0.178273), correlation (0.501854)*/,
                    -8,5, -6,-9/*mean (0.181107), correlation (0.494888)*/,
                    -5,-1, -4,5/*mean (0.190227), correlation (0.482557)*/,
                    -13,7, -8,10/*mean (0.196739), correlation (0.496503)*/,
                    1,5, 5,-13/*mean (0.19973), correlation (0.499759)*/,
                    1,0, 10,-13/*mean (0.204465), correlation (0.49873)*/,
                    9,12, 10,-1/*mean (0.209334), correlation (0.49063)*/,
                    5,-8, 10,-9/*mean (0.211134), correlation (0.503011)*/,
                    -1,11, 1,-13/*mean (0.212), correlation (0.499414)*/,
                    -9,-3, -6,2/*mean (0.212168), correlation (0.480739)*/,
                    -1,-10, 1,12/*mean (0.212731), correlation (0.502523)*/,
                    -13,1, -8,-10/*mean (0.21327), correlation (0.489786)*/,
                    8,-11, 10,-6/*mean (0.214159), correlation (0.488246)*/,
                    2,-13, 3,-6/*mean (0.216993), correlation (0.50287)*/,
                    7,-13, 12,-9/*mean (0.223639), correlation (0.470502)*/,
                    -10,-10, -5,-7/*mean (0.224089), correlation (0.500852)*/,
                    -10,-8, -8,-13/*mean (0.228666), correlation (0.502629)*/,
                    4,-6, 8,5/*mean (0.22906), correlation (0.498305)*/,
                    3,12, 8,-13/*mean (0.233378), correlation (0.503825)*/,
                    -4,2, -3,-3/*mean (0.234323), correlation (0.476692)*/,
                    5,-13, 10,-12/*mean (0.236392), correlation (0.475462)*/,
                    4,-13, 5,-1/*mean (0.236842), correlation (0.504132)*/,
                    -9,9, -4,3/*mean (0.236977), correlation (0.497739)*/,
                    0,3, 3,-9/*mean (0.24314), correlation (0.499398)*/,
                    -12,1, -6,1/*mean (0.243297), correlation (0.489447)*/,
                    3,2, 4,-8/*mean (0.00155196), correlation (0.553496)*/,
                    -10,-10, -10,9/*mean (0.00239541), correlation (0.54297)*/,
                    8,-13, 12,12/*mean (0.0034413), correlation (0.544361)*/,
                    -8,-12, -6,-5/*mean (0.003565), correlation (0.551225)*/,
                    2,2, 3,7/*mean (0.00835583), correlation (0.55285)*/,
                    10,6, 11,-8/*mean (0.00885065), correlation (0.540913)*/,
                    6,8, 8,-12/*mean (0.0101552), correlation (0.551085)*/,
                    -7,10, -6,5/*mean (0.0102227), correlation (0.533635)*/,
                    -3,-9, -3,9/*mean (0.0110211), correlation (0.543121)*/,
                    -1,-13, -1,5/*mean (0.0113473), correlation (0.550173)*/,
                    -3,-7, -3,4/*mean (0.0140913), correlation (0.554774)*/,
                    -8,-2, -8,3/*mean (0.017049), correlation (0.55461)*/,
                    4,2, 12,12/*mean (0.01778), correlation (0.546921)*/,
                    2,-5, 3,11/*mean (0.0224022), correlation (0.549667)*/,
                    6,-9, 11,-13/*mean (0.029161), correlation (0.546295)*/,
                    3,-1, 7,12/*mean (0.0303081), correlation (0.548599)*/,
                    11,-1, 12,4/*mean (0.0355151), correlation (0.523943)*/,
                    -3,0, -3,6/*mean (0.0417904), correlation (0.543395)*/,
                    4,-11, 4,12/*mean (0.0487292), correlation (0.542818)*/,
                    2,-4, 2,1/*mean (0.0575124), correlation (0.554888)*/,
                    -10,-6, -8,1/*mean (0.0594242), correlation (0.544026)*/,
                    -13,7, -11,1/*mean (0.0597391), correlation (0.550524)*/,
                    -13,12, -11,-13/*mean (0.0608974), correlation (0.55383)*/,
                    6,0, 11,-13/*mean (0.065126), correlation (0.552006)*/,
                    0,-1, 1,4/*mean (0.074224), correlation (0.546372)*/,
                    -13,3, -9,-2/*mean (0.0808592), correlation (0.554875)*/,
                    -9,8, -6,-3/*mean (0.0883378), correlation (0.551178)*/,
                    -13,-6, -8,-2/*mean (0.0901035), correlation (0.548446)*/,
                    5,-9, 8,10/*mean (0.0949843), correlation (0.554694)*/,
                    2,7, 3,-9/*mean (0.0994152), correlation (0.550979)*/,
                    -1,-6, -1,-1/*mean (0.10045), correlation (0.552714)*/,
                    9,5, 11,-2/*mean (0.100686), correlation (0.552594)*/,
                    11,-3, 12,-8/*mean (0.101091), correlation (0.532394)*/,
                    3,0, 3,5/*mean (0.101147), correlation (0.525576)*/,
                    -1,4, 0,10/*mean (0.105263), correlation (0.531498)*/,
                    3,-6, 4,5/*mean (0.110785), correlation (0.540491)*/,
                    -13,0, -10,5/*mean (0.112798), correlation (0.536582)*/,
                    5,8, 12,11/*mean (0.114181), correlation (0.555793)*/,
                    8,9, 9,-6/*mean (0.117431), correlation (0.553763)*/,
                    7,-4, 8,-12/*mean (0.118522), correlation (0.553452)*/,
                    -10,4, -10,9/*mean (0.12094), correlation (0.554785)*/,
                    7,3, 12,4/*mean (0.122582), correlation (0.555825)*/,
                    9,-7, 10,-2/*mean (0.124978), correlation (0.549846)*/,
                    7,0, 12,-2/*mean (0.127002), correlation (0.537452)*/,
                    -1,-6, 0,-11/*mean (0.127148), correlation (0.547401)*/
            };

    ORBextractor::ORBextractor(int _nfeatures, float _scaleFactor, int _nlevels,
                               int _iniThFAST, int _minThFAST):
            nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels),
            iniThFAST(_iniThFAST), minThFAST(_minThFAST)
    {
        mvScaleFactor.resize(nlevels);
        mvLevelSigma2.resize(nlevels);
        mvScaleFactor[0]=1.0f;
        mvLevelSigma2[0]=1.0f;
        for(int i=1; i<nlevels; i++)
        {
            mvScaleFactor[i]=mvScaleFactor[i-1]*scaleFactor;
            mvLevelSigma2[i]=mvScaleFactor[i]*mvScaleFactor[i];
        }

        mvInvScaleFactor.resize(nlevels);
        mvInvLevelSigma2.resize(nlevels);
        for(int i=0; i<nlevels; i++)
        {
            mvInvScaleFactor[i]=1.0f/mvScaleFactor[i];
            mvInvLevelSigma2[i]=1.0f/mvLevelSigma2[i];
        }

        mvImagePyramid.resize(nlevels);

        mnFeaturesPerLevel.resize(nlevels);
        float factor = 1.0f / scaleFactor;
        float nDesiredFeaturesPerScale = nfeatures*(1 - factor)/(1 - (float)pow((double)factor, (double)nlevels));

        int sumFeatures = 0;
        for( int level = 0; level < nlevels-1; level++ )
        {
            mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
            sumFeatures += mnFeaturesPerLevel[level];
            nDesiredFeaturesPerScale *= factor;
        }
        mnFeaturesPerLevel[nlevels-1] = std::max(nfeatures - sumFeatures, 0);

        const int npoints = 512;
        const Point* pattern0 = (const Point*)bit_pattern_31_;
        std::copy(pattern0, pattern0 + npoints, std::back_inserter(pattern));

        //This is for orientation
        // pre-compute the end of a row in a circular patch
        umax.resize(HALF_PATCH_SIZE + 1);

        int v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
        int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
        const double hp2 = HALF_PATCH_SIZE*HALF_PATCH_SIZE;
        for (v = 0; v <= vmax; ++v)
            umax[v] = cvRound(sqrt(hp2 - v * v));

        // Make sure we are symmetric
        for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
        {
            while (umax[v0] == umax[v0 + 1])
                ++v0;
            umax[v] = v0;
            ++v0;
        }
    }

    static void computeOrientation(const Mat& image, vector<KeyPoint>& keypoints, const vector<int>& umax)
    {
        for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
                     keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
        {
            keypoint->angle = IC_Angle(image, keypoint->pt, umax);
        }
    }

    void ExtractorNode::DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4)
    {
        const int halfX = ceil(static_cast<float>(UR.x-UL.x)/2);
        const int halfY = ceil(static_cast<float>(BR.y-UL.y)/2);

        //Define boundaries of childs
        n1.UL = UL;
        n1.UR = cv::Point2i(UL.x+halfX,UL.y);
        n1.BL = cv::Point2i(UL.x,UL.y+halfY);
        n1.BR = cv::Point2i(UL.x+halfX,UL.y+halfY);
        n1.vKeys.reserve(vKeys.size());

        n2.UL = n1.UR;
        n2.UR = UR;
        n2.BL = n1.BR;
        n2.BR = cv::Point2i(UR.x,UL.y+halfY);
        n2.vKeys.reserve(vKeys.size());

        n3.UL = n1.BL;
        n3.UR = n1.BR;
        n3.BL = BL;
        n3.BR = cv::Point2i(n1.BR.x,BL.y);
        n3.vKeys.reserve(vKeys.size());

        n4.UL = n3.UR;
        n4.UR = n2.BR;
        n4.BL = n3.BR;
        n4.BR = BR;
        n4.vKeys.reserve(vKeys.size());

        //Associate points to childs
        for(size_t i=0;i<vKeys.size();i++)
        {
            const cv::KeyPoint &kp = vKeys[i];
            if(kp.pt.x<n1.UR.x)
            {
                if(kp.pt.y<n1.BR.y)
                    n1.vKeys.push_back(kp);
                else
                    n3.vKeys.push_back(kp);
            }
            else if(kp.pt.y<n1.BR.y)
                n2.vKeys.push_back(kp);
            else
                n4.vKeys.push_back(kp);
        }

        if(n1.vKeys.size()==1)
            n1.bNoMore = true;
        if(n2.vKeys.size()==1)
            n2.bNoMore = true;
        if(n3.vKeys.size()==1)
            n3.bNoMore = true;
        if(n4.vKeys.size()==1)
            n4.bNoMore = true;

    }

    static bool compareNodes(pair<int,ExtractorNode*>& e1, pair<int,ExtractorNode*>& e2){
        if(e1.first < e2.first){
            return true;
        }
        else if(e1.first > e2.first){
            return false;
        }
        else{
            if(e1.second->UL.x < e2.second->UL.x){
                return true;
            }
            else{
                return false;
            }
        }
    }

    vector<cv::KeyPoint> ORBextractor::DistributeOctTree(const vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
                                                         const int &maxX, const int &minY, const int &maxY, const int &N, const int &level)
    {
        // Compute how many initial nodes
        const int nIni = round(static_cast<float>(maxX-minX)/(maxY-minY));

        const float hX = static_cast<float>(maxX-minX)/nIni;

        list<ExtractorNode> lNodes;

        vector<ExtractorNode*> vpIniNodes;
        vpIniNodes.resize(nIni);

        for(int i=0; i<nIni; i++)
        {
            ExtractorNode ni;
            ni.UL = cv::Point2i(hX*static_cast<float>(i),0);
            ni.UR = cv::Point2i(hX*static_cast<float>(i+1),0);
            ni.BL = cv::Point2i(ni.UL.x,maxY-minY);
            ni.BR = cv::Point2i(ni.UR.x,maxY-minY);
            ni.vKeys.reserve(vToDistributeKeys.size());

            lNodes.push_back(ni);
            vpIniNodes[i] = &lNodes.back();
        }

        //Associate points to childs
        for(size_t i=0;i<vToDistributeKeys.size();i++)
        {
            const cv::KeyPoint &kp = vToDistributeKeys[i];
            vpIniNodes[kp.pt.x/hX]->vKeys.push_back(kp);
        }

        list<ExtractorNode>::iterator lit = lNodes.begin();

        while(lit!=lNodes.end())
        {
            if(lit->vKeys.size()==1)
            {
                lit->bNoMore=true;
                lit++;
            }
            else if(lit->vKeys.empty())
                lit = lNodes.erase(lit);
            else
                lit++;
        }

        bool bFinish = false;

        int iteration = 0;

        vector<pair<int,ExtractorNode*> > vSizeAndPointerToNode;
        vSizeAndPointerToNode.reserve(lNodes.size()*4);

        while(!bFinish)
        {
            iteration++;

            int prevSize = lNodes.size();

            lit = lNodes.begin();

            int nToExpand = 0;

            vSizeAndPointerToNode.clear();

            while(lit!=lNodes.end())
            {
                if(lit->bNoMore)
                {
                    // If node only contains one point do not subdivide and continue
                    lit++;
                    continue;
                }
                else
                {
                    // If more than one point, subdivide
                    ExtractorNode n1,n2,n3,n4;
                    lit->DivideNode(n1,n2,n3,n4);

                    // Add childs if they contain points
                    if(n1.vKeys.size()>0)
                    {
                        lNodes.push_front(n1);
                        if(n1.vKeys.size()>1)
                        {
                            nToExpand++;
                            vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n2.vKeys.size()>0)
                    {
                        lNodes.push_front(n2);
                        if(n2.vKeys.size()>1)
                        {
                            nToExpand++;
                            vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n3.vKeys.size()>0)
                    {
                        lNodes.push_front(n3);
                        if(n3.vKeys.size()>1)
                        {
                            nToExpand++;
                            vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n4.vKeys.size()>0)
                    {
                        lNodes.push_front(n4);
                        if(n4.vKeys.size()>1)
                        {
                            nToExpand++;
                            vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }

                    lit=lNodes.erase(lit);
                    continue;
                }
            }

            // Finish if there are more nodes than required features
            // or all nodes contain just one point
            if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize)
            {
                bFinish = true;
            }
            else if(((int)lNodes.size()+nToExpand*3)>N)
            {

                while(!bFinish)
                {

                    prevSize = lNodes.size();

                    vector<pair<int,ExtractorNode*> > vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
                    vSizeAndPointerToNode.clear();

                    sort(vPrevSizeAndPointerToNode.begin(),vPrevSizeAndPointerToNode.end(),compareNodes);
                    for(int j=vPrevSizeAndPointerToNode.size()-1;j>=0;j--)
                    {
                        ExtractorNode n1,n2,n3,n4;
                        vPrevSizeAndPointerToNode[j].second->DivideNode(n1,n2,n3,n4);

                        // Add childs if they contain points
                        if(n1.vKeys.size()>0)
                        {
                            lNodes.push_front(n1);
                            if(n1.vKeys.size()>1)
                            {
                                vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));
                                lNodes.front().lit = lNodes.begin();
                            }
                        }
                        if(n2.vKeys.size()>0)
                        {
                            lNodes.push_front(n2);
                            if(n2.vKeys.size()>1)
                            {
                                vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
                                lNodes.front().lit = lNodes.begin();
                            }
                        }
                        if(n3.vKeys.size()>0)
                        {
                            lNodes.push_front(n3);
                            if(n3.vKeys.size()>1)
                            {
                                vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
                                lNodes.front().lit = lNodes.begin();
                            }
                        }
                        if(n4.vKeys.size()>0)
                        {
                            lNodes.push_front(n4);
                            if(n4.vKeys.size()>1)
                            {
                                vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
                                lNodes.front().lit = lNodes.begin();
                            }
                        }

                        lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);

                        if((int)lNodes.size()>=N)
                            break;
                    }

                    if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize)
                        bFinish = true;

                }
            }
        }

        // Retain the best point in each node
        vector<cv::KeyPoint> vResultKeys;
        vResultKeys.reserve(nfeatures);
        for(list<ExtractorNode>::iterator lit=lNodes.begin(); lit!=lNodes.end(); lit++)
        {
            vector<cv::KeyPoint> &vNodeKeys = lit->vKeys;
            cv::KeyPoint* pKP = &vNodeKeys[0];
            float maxResponse = pKP->response;

            for(size_t k=1;k<vNodeKeys.size();k++)
            {
                if(vNodeKeys[k].response>maxResponse)
                {
                    pKP = &vNodeKeys[k];
                    maxResponse = vNodeKeys[k].response;
                }
            }

            vResultKeys.push_back(*pKP);
        }

        return vResultKeys;
    }

    void ORBextractor::ComputeKeyPointsOctTree(vector<vector<KeyPoint> >& allKeypoints)
    {
	allKeypoints.resize(nlevels);

        const float W = 35;

		#ifdef REGISTER_TIMES_SUBSTAGE
			double fast_us_accum = 0.0;
			double octtree_us_accum = 0.0;
		#endif

        for (int level = 0; level < nlevels; ++level)
        {
            const int minBorderX = EDGE_THRESHOLD-3;
            const int minBorderY = minBorderX;
            const int maxBorderX = mvImagePyramid[level].cols-EDGE_THRESHOLD+3;
            const int maxBorderY = mvImagePyramid[level].rows-EDGE_THRESHOLD+3;

            vector<cv::KeyPoint> vToDistributeKeys;
            vToDistributeKeys.reserve(nfeatures*10);

            const float width = (maxBorderX-minBorderX);
            const float height = (maxBorderY-minBorderY);

            const int nCols = width/W;
            const int nRows = height/W;
            const int wCell = ceil(width/nCols);
            const int hCell = ceil(height/nRows);

			#ifdef REGISTER_TIMES_SUBSTAGE
			auto t_fast_start = std::chrono::high_resolution_clock::now();
			#endif

            for(int i=0; i<nRows; i++)
            {
                const float iniY =minBorderY+i*hCell;
                float maxY = iniY+hCell+6;

                if(iniY>=maxBorderY-3)
                    continue;
                if(maxY>maxBorderY)
                    maxY = maxBorderY;

                for(int j=0; j<nCols; j++)
                {
                    const float iniX =minBorderX+j*wCell;
                    float maxX = iniX+wCell+6;
                    if(iniX>=maxBorderX-6)
                        continue;
                    if(maxX>maxBorderX)
                        maxX = maxBorderX;

                    vector<cv::KeyPoint> vKeysCell;

                    FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                         vKeysCell,iniThFAST,true);

                    /*if(bRight && j <= 13){
                        FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                             vKeysCell,10,true);
                    }
                    else if(!bRight && j >= 16){
                        FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                             vKeysCell,10,true);
                    }
                    else{
                        FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                             vKeysCell,iniThFAST,true);
                    }*/


                    if(vKeysCell.empty())
                    {
                        FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                             vKeysCell,minThFAST,true);
                        /*if(bRight && j <= 13){
                            FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                                 vKeysCell,5,true);
                        }
                        else if(!bRight && j >= 16){
                            FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                                 vKeysCell,5,true);
                        }
                        else{
                            FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                                 vKeysCell,minThFAST,true);
                        }*/
                    }

                    if(!vKeysCell.empty())
                    {
                        for(vector<cv::KeyPoint>::iterator vit=vKeysCell.begin(); vit!=vKeysCell.end();vit++)
                        {
                            (*vit).pt.x+=j*wCell;
                            (*vit).pt.y+=i*hCell;
                            vToDistributeKeys.push_back(*vit);
                        }
                    }

                }
            }

            #ifdef REGISTER_TIMES_SUBSTAGE
			auto t_fast_end = std::chrono::high_resolution_clock::now();
			#endif

            vector<KeyPoint> & keypoints = allKeypoints[level];
            keypoints.reserve(nfeatures);

            keypoints = DistributeOctTree(vToDistributeKeys, minBorderX, maxBorderX,
                                          minBorderY, maxBorderY,mnFeaturesPerLevel[level], level);

            #ifdef REGISTER_TIMES_SUBSTAGE
			auto t_octtree_end = std::chrono::high_resolution_clock::now();
			fast_us_accum += std::chrono::duration<double, std::micro>(t_fast_end - t_fast_start).count();
			octtree_us_accum += std::chrono::duration<double, std::micro>(t_octtree_end - t_fast_end).count();
			#endif

            const int scaledPatchSize = PATCH_SIZE*mvScaleFactor[level];

            // Add border to coordinates and scale information
            const int nkps = keypoints.size();
            for(int i=0; i<nkps ; i++)
            {
                keypoints[i].pt.x+=minBorderX;
                keypoints[i].pt.y+=minBorderY;
                keypoints[i].octave=level;
                keypoints[i].size = scaledPatchSize;
            }
        }

	#ifdef REGISTER_TIMES_SUBSTAGE
	auto t_orient_start = std::chrono::high_resolution_clock::now();
	#endif

        // compute orientations
        for (int level = 0; level < nlevels; ++level)
            computeOrientation(mvImagePyramid[level], allKeypoints[level], umax);

	#ifdef REGISTER_TIMES_SUBSTAGE
	auto t_orient_end = std::chrono::high_resolution_clock::now();
	vdFAST_us.push_back(fast_us_accum);
	vdOctTree_us.push_back(octtree_us_accum);
	vdOrient_us.push_back(std::chrono::duration<double, std::micro>(t_orient_end - t_orient_start).count());
	#endif
    }

#ifdef USE_HW_ACCEL
    void ORBextractor::ComputeKeyPointsHardware(vector<vector<KeyPoint> >& allKeypoints)
    {
        allKeypoints.resize(nlevels);

        #ifdef REGISTER_TIMES_SUBSTAGE
            double hw_us_accum = 0.0;
            double hw_dma_reset_us_accum = 0.0;
            double octtree_us_accum = 0.0;
        #endif

        for(int level = 0; level < nlevels; ++level)
        {
            const int minBorderX = EDGE_THRESHOLD-3;
            const int minBorderY = minBorderX;
            const int maxBorderX = mvImagePyramid[level].cols-EDGE_THRESHOLD+3;
            const int maxBorderY = mvImagePyramid[level].rows-EDGE_THRESHOLD+3;

            vector<KeyPoint> vToDistributeKeys;
            vToDistributeKeys.reserve(nfeatures*10);

            #ifdef REGISTER_TIMES_SUBSTAGE
            auto t_hw_start = std::chrono::high_resolution_clock::now();
            #endif

            const double dma_reset_us =
                GetOrbHwAccelerator().RunLevel(mvImagePyramid[level], vToDistributeKeys, level,
                                               minBorderX, maxBorderX, minBorderY, maxBorderY);
#ifndef REGISTER_TIMES_SUBSTAGE
            (void)dma_reset_us;
#endif

            #ifdef REGISTER_TIMES_SUBSTAGE
            auto t_hw_end = std::chrono::high_resolution_clock::now();
            hw_dma_reset_us_accum += dma_reset_us;
            #endif

            vector<KeyPoint>& keypoints = allKeypoints[level];
            keypoints.reserve(nfeatures);

            keypoints = DistributeOctTree(vToDistributeKeys, minBorderX, maxBorderX,
                                          minBorderY, maxBorderY, mnFeaturesPerLevel[level], level);

            #ifdef REGISTER_TIMES_SUBSTAGE
            auto t_octtree_end = std::chrono::high_resolution_clock::now();
            hw_us_accum += std::chrono::duration<double, std::micro>(t_hw_end - t_hw_start).count();
            octtree_us_accum += std::chrono::duration<double, std::micro>(t_octtree_end - t_hw_end).count();
            #endif

            const int scaledPatchSize = PATCH_SIZE*mvScaleFactor[level];

            const int nkps = keypoints.size();
            for(int i = 0; i < nkps; ++i)
            {
                keypoints[i].pt.x += minBorderX;
                keypoints[i].pt.y += minBorderY;
                keypoints[i].octave = level;
                keypoints[i].size = scaledPatchSize;
            }
        }

        #ifdef REGISTER_TIMES_SUBSTAGE
        vdFAST_us.push_back(hw_us_accum);
        vdHwDmaReset_us.push_back(hw_dma_reset_us_accum);
        vdOctTree_us.push_back(octtree_us_accum);
        #endif
    }
#endif

    void ORBextractor::ComputeKeyPointsOld(std::vector<std::vector<KeyPoint> > &allKeypoints)
    {
        allKeypoints.resize(nlevels);

        float imageRatio = (float)mvImagePyramid[0].cols/mvImagePyramid[0].rows;

        for (int level = 0; level < nlevels; ++level)
        {
            const int nDesiredFeatures = mnFeaturesPerLevel[level];

            const int levelCols = sqrt((float)nDesiredFeatures/(5*imageRatio));
            const int levelRows = imageRatio*levelCols;

            const int minBorderX = EDGE_THRESHOLD;
            const int minBorderY = minBorderX;
            const int maxBorderX = mvImagePyramid[level].cols-EDGE_THRESHOLD;
            const int maxBorderY = mvImagePyramid[level].rows-EDGE_THRESHOLD;

            const int W = maxBorderX - minBorderX;
            const int H = maxBorderY - minBorderY;
            const int cellW = ceil((float)W/levelCols);
            const int cellH = ceil((float)H/levelRows);

            const int nCells = levelRows*levelCols;
            const int nfeaturesCell = ceil((float)nDesiredFeatures/nCells);

            vector<vector<vector<KeyPoint> > > cellKeyPoints(levelRows, vector<vector<KeyPoint> >(levelCols));

            vector<vector<int> > nToRetain(levelRows,vector<int>(levelCols,0));
            vector<vector<int> > nTotal(levelRows,vector<int>(levelCols,0));
            vector<vector<bool> > bNoMore(levelRows,vector<bool>(levelCols,false));
            vector<int> iniXCol(levelCols);
            vector<int> iniYRow(levelRows);
            int nNoMore = 0;
            int nToDistribute = 0;


            float hY = cellH + 6;

            for(int i=0; i<levelRows; i++)
            {
                const float iniY = minBorderY + i*cellH - 3;
                iniYRow[i] = iniY;

                if(i == levelRows-1)
                {
                    hY = maxBorderY+3-iniY;
                    if(hY<=0)
                        continue;
                }

                float hX = cellW + 6;

                for(int j=0; j<levelCols; j++)
                {
                    float iniX;

                    if(i==0)
                    {
                        iniX = minBorderX + j*cellW - 3;
                        iniXCol[j] = iniX;
                    }
                    else
                    {
                        iniX = iniXCol[j];
                    }


                    if(j == levelCols-1)
                    {
                        hX = maxBorderX+3-iniX;
                        if(hX<=0)
                            continue;
                    }


                    Mat cellImage = mvImagePyramid[level].rowRange(iniY,iniY+hY).colRange(iniX,iniX+hX);

                    cellKeyPoints[i][j].reserve(nfeaturesCell*5);

                    FAST(cellImage,cellKeyPoints[i][j],iniThFAST,true);

                    if(cellKeyPoints[i][j].size()<=3)
                    {
                        cellKeyPoints[i][j].clear();

                        FAST(cellImage,cellKeyPoints[i][j],minThFAST,true);
                    }


                    const int nKeys = cellKeyPoints[i][j].size();
                    nTotal[i][j] = nKeys;

                    if(nKeys>nfeaturesCell)
                    {
                        nToRetain[i][j] = nfeaturesCell;
                        bNoMore[i][j] = false;
                    }
                    else
                    {
                        nToRetain[i][j] = nKeys;
                        nToDistribute += nfeaturesCell-nKeys;
                        bNoMore[i][j] = true;
                        nNoMore++;
                    }

                }
            }


            // Retain by score

            while(nToDistribute>0 && nNoMore<nCells)
            {
                int nNewFeaturesCell = nfeaturesCell + ceil((float)nToDistribute/(nCells-nNoMore));
                nToDistribute = 0;

                for(int i=0; i<levelRows; i++)
                {
                    for(int j=0; j<levelCols; j++)
                    {
                        if(!bNoMore[i][j])
                        {
                            if(nTotal[i][j]>nNewFeaturesCell)
                            {
                                nToRetain[i][j] = nNewFeaturesCell;
                                bNoMore[i][j] = false;
                            }
                            else
                            {
                                nToRetain[i][j] = nTotal[i][j];
                                nToDistribute += nNewFeaturesCell-nTotal[i][j];
                                bNoMore[i][j] = true;
                                nNoMore++;
                            }
                        }
                    }
                }
            }

            vector<KeyPoint> & keypoints = allKeypoints[level];
            keypoints.reserve(nDesiredFeatures*2);

            const int scaledPatchSize = PATCH_SIZE*mvScaleFactor[level];

            // Retain by score and transform coordinates
            for(int i=0; i<levelRows; i++)
            {
                for(int j=0; j<levelCols; j++)
                {
                    vector<KeyPoint> &keysCell = cellKeyPoints[i][j];
                    KeyPointsFilter::retainBest(keysCell,nToRetain[i][j]);
                    if((int)keysCell.size()>nToRetain[i][j])
                        keysCell.resize(nToRetain[i][j]);


                    for(size_t k=0, kend=keysCell.size(); k<kend; k++)
                    {
                        keysCell[k].pt.x+=iniXCol[j];
                        keysCell[k].pt.y+=iniYRow[i];
                        keysCell[k].octave=level;
                        keysCell[k].size = scaledPatchSize;
                        keypoints.push_back(keysCell[k]);
                    }
                }
            }

            if((int)keypoints.size()>nDesiredFeatures)
            {
                KeyPointsFilter::retainBest(keypoints,nDesiredFeatures);
                keypoints.resize(nDesiredFeatures);
            }
        }

        // and compute orientations
        for (int level = 0; level < nlevels; ++level)
            computeOrientation(mvImagePyramid[level], allKeypoints[level], umax);
    }

    static void computeDescriptors(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors,
                                   const vector<Point>& pattern)
    {
        descriptors = Mat::zeros((int)keypoints.size(), 32, CV_8UC1);

        for (size_t i = 0; i < keypoints.size(); i++)
            computeOrbDescriptor(keypoints[i], image, &pattern[0], descriptors.ptr((int)i));
    }

    int ORBextractor::operator()( InputArray _image, InputArray _mask, vector<KeyPoint>& _keypoints,
                                  OutputArray _descriptors, std::vector<int> &vLappingArea)
    {
        //cout << "[ORBextractor]: Max Features: " << nfeatures << endl;
        if(_image.empty())
            return -1;

        Mat image = _image.getMat();
        assert(image.type() == CV_8UC1 );

        #ifdef REGISTER_TIMES_SUBSTAGE
		auto t_pyr_start = std::chrono::high_resolution_clock::now();
		#endif
		
        // Pre-compute the scale pyramid
        ComputePyramid(image);

        #ifdef REGISTER_TIMES_SUBSTAGE
        auto t_pyr_end = std::chrono::high_resolution_clock::now();
        vdPyramid_us.push_back(std::chrono::duration<double, std::micro>(t_pyr_end - t_pyr_start).count());
		#endif

        vector < vector<KeyPoint> > allKeypoints;
#ifdef USE_HW_ACCEL
        ComputeKeyPointsHardware(allKeypoints);
#endif
#ifndef USE_HW_ACCEL
        ComputeKeyPointsOctTree(allKeypoints);
        //ComputeKeyPointsOld(allKeypoints);
#endif


        #ifdef REGISTER_TIMES_SUBSTAGE
		auto t_desc_start = std::chrono::high_resolution_clock::now();
		#endif

        Mat descriptors;

        int nkeypoints = 0;
        for (int level = 0; level < nlevels; ++level)
            nkeypoints += (int)allKeypoints[level].size();
        if( nkeypoints == 0 )
            _descriptors.release();
        else
        {
            _descriptors.create(nkeypoints, 32, CV_8U);
            descriptors = _descriptors.getMat();
        }

        //_keypoints.clear();
        //_keypoints.reserve(nkeypoints);
        _keypoints = vector<cv::KeyPoint>(nkeypoints);

        int offset = 0;
        //Modified for speeding up stereo fisheye matching
        int monoIndex = 0, stereoIndex = nkeypoints-1;
        for (int level = 0; level < nlevels; ++level)
        {
            vector<KeyPoint>& keypoints = allKeypoints[level];
            int nkeypointsLevel = (int)keypoints.size();

            if(nkeypointsLevel==0)
                continue;

            // preprocess the resized image
            Mat workingMat = mvImagePyramid[level].clone();
            GaussianBlur(workingMat, workingMat, Size(7, 7), 2, 2, BORDER_REFLECT_101);

            // Compute the descriptors
            //Mat desc = descriptors.rowRange(offset, offset + nkeypointsLevel);
            Mat desc = cv::Mat(nkeypointsLevel, 32, CV_8U);
            computeDescriptors(workingMat, keypoints, desc, pattern);

            offset += nkeypointsLevel;


            float scale = mvScaleFactor[level]; //getScale(level, firstLevel, scaleFactor);
            int i = 0;
            for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
                         keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint){

                // Scale keypoint coordinates
                if (level != 0){
                    keypoint->pt *= scale;
                }

                if(keypoint->pt.x >= vLappingArea[0] && keypoint->pt.x <= vLappingArea[1]){
                    _keypoints.at(stereoIndex) = (*keypoint);
                    desc.row(i).copyTo(descriptors.row(stereoIndex));
                    stereoIndex--;
                }
                else{
                    _keypoints.at(monoIndex) = (*keypoint);
                    desc.row(i).copyTo(descriptors.row(monoIndex));
                    monoIndex++;
                }
                i++;
            }
        }

        #ifdef REGISTER_TIMES_SUBSTAGE
		auto t_desc_end = std::chrono::high_resolution_clock::now();
		vdDescriptors_us.push_back(std::chrono::duration<double, std::micro>(t_desc_end - t_desc_start).count());
		#endif



        //cout << "[ORBextractor]: extracted " << _keypoints.size() << " KeyPoints" << endl;
        return monoIndex;
    }

    void ORBextractor::ComputePyramid(cv::Mat image)
    {
        for (int level = 0; level < nlevels; ++level)
        {
            float scale = mvInvScaleFactor[level];
            Size sz(cvRound((float)image.cols*scale), cvRound((float)image.rows*scale));
            Size wholeSize(sz.width + EDGE_THRESHOLD*2, sz.height + EDGE_THRESHOLD*2);
            Mat temp(wholeSize, image.type()), masktemp;
            mvImagePyramid[level] = temp(Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));

            // Compute the resized image
            if( level != 0 )
            {
                resize(mvImagePyramid[level-1], mvImagePyramid[level], sz, 0, 0, INTER_LINEAR);

                copyMakeBorder(mvImagePyramid[level], temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                               BORDER_REFLECT_101+BORDER_ISOLATED);
            }
            else
            {
                copyMakeBorder(image, temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                               BORDER_REFLECT_101);
            }
        }

    }
    
    #ifdef REGISTER_TIMES_SUBSTAGE
    void ORBextractor::WriteTimingsCSV(const std::string& filename)
    {
        std::ofstream f(filename);
#ifdef USE_HW_ACCEL
        f << "#Pyramid_us,HW_Total_us,HW_DmaReset_us,OctTree_us,Descriptors_us\n";
#else
        f << "#Pyramid_us,FAST_us,OctTree_us,Orient_us,Descriptors_us\n";
#endif
        const size_t N = vdFAST_us.size();
        for (size_t i = 0; i < N; ++i) {
            const double pyr  = (i < vdPyramid_us.size())     ? vdPyramid_us[i]     : 0.0;
            const double fast = (i < vdFAST_us.size())        ? vdFAST_us[i]        : 0.0;
            const double oct  = (i < vdOctTree_us.size())     ? vdOctTree_us[i]     : 0.0;
            const double dsc  = (i < vdDescriptors_us.size()) ? vdDescriptors_us[i] : 0.0;
#ifdef USE_HW_ACCEL
            const double dmaReset = (i < vdHwDmaReset_us.size()) ? vdHwDmaReset_us[i] : 0.0;
            f << pyr << "," << fast << "," << dmaReset << "," << oct << "," << dsc << "\n";
#else
            const double ori  = (i < vdOrient_us.size())      ? vdOrient_us[i]      : 0.0;
            f << pyr << "," << fast << "," << oct << "," << ori << "," << dsc << "\n";
#endif
        }
        f.close();
    }
	#endif

} //namespace ORB_SLAM
