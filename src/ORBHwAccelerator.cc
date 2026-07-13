/**
* This file is part of a hardware-accelerated fork of ORB-SLAM3
* (github.com/luett1/ORB_SLAM3) developed for the Master's thesis
* "On-device 3D Mapping" (M. Schieber, SDSU/UniBw, 2026). Licensed under
* GPLv3, like ORB-SLAM3 itself (see the original notice in ORBextractor.cc).
*
* Userspace driver for the custom VHDL ORB feature-extraction accelerator in
* the programmable logic (PL) of the AMD Kria KR260 (Zynq UltraScale+ XCK26).
*
* WHAT THE HARDWARE DOES (per pyramid level): the PS streams the level image
* through an AXI DMA into the accelerator, which performs FAST corner
* detection at the permissive threshold (7) over the whole image, 3x3
* non-maximum suppression, a strict-per-cell gate (see below) and the
* intensity-centroid orientation, and streams back one 16-byte record per
* surviving corner (packing v3: x, y, int32 response, 24-bit angle + flags)
* terminated by an EOF sentinel. Pyramid construction, octree distribution
* and rBRIEF descriptors stay on the PS (ORBextractor.cc) -- the descriptor
* is specified as future work for the smart-camera follow-up, not built.
*
* HOW IT IS DRIVEN: no kernel driver is needed. The AXI DMA and the
* accelerator's TOP register file are mapped from userspace via generic-uio
* (/dev/uioN), and the two DMA data buffers are u-dma-buf devices
* (/dev/udmabufN) whose physical addresses come from sysfs. The input
* (pixel) buffer is mapped cacheable for fast memcpy and explicitly flushed
* before each transfer; the output buffer is mapped uncached (small reads).
* Everything is polled -- the DMA completion interrupts exist in the PL but
* are deliberately masked (registered as generic-uio in the device tree).
*
* MULTI-INSTANCE (F.12): the dual-accelerator bitstream carries two
* independent core+DMA pairs on separate HP ports so the stereo left/right
* extractions can run truly in parallel. kHwDeviceSets[] below maps
* instance -> device files; ORB_HW_NUM_ACCELERATORS (ORBextractor.h) selects
* how many instances the build may use.
*
* ENTRY POINT: ORBextractor::ComputeKeyPointsHardware() calls
* orb_hw::RunLevel() (bottom of this file) once per pyramid level.
*/

#ifdef USE_HW_ACCEL

#include "ORBHwAccelerator.h"
#include "ORBextractor.h"   // for the ORB_HW_NUM_ACCELERATORS default (shared with Frame.cc)

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include <opencv2/core/core.hpp>

using namespace cv;
using namespace std;

namespace ORB_SLAM3
{

namespace {

    // Keypoint records are ranked and binned with the same geometry the PS-side
    // extractor uses; this constant must stay equal to ORBextractor.cc's
    // EDGE_THRESHOLD (and to the border geometry baked into the RTL).
    const int EDGE_THRESHOLD = 19;

    using orb_hw::HwLevelStats;   // defined in ORBHwAccelerator.h

    // ---- Hardware accelerator instance table --------------------------------
    // ORB_HW_NUM_ACCELERATORS (defaulted in ORBextractor.h so Frame.cc sees the
    // same value) = number of physical accelerator instances in the bitstream.
    //   2 -> dual-accelerator build (F.12): the left/right extractors drive
    //        separate cores on separate DMAs/HP ports, and Frame.cc runs the
    //        stereo ExtractORB calls in parallel threads.
    //   1 -> single-instance build: Frame.cc serializes left/right, both stereo
    //        extractors collapse onto instance 0 (see GetOrbHwAccelerator) and
    //        share its RunLevel mutex, exactly reproducing the pre-F.12 path.
    // Switch builds via the header default or -DORB_HW_NUM_ACCELERATORS=1;
    // no other code change is needed (mono/RGBD always use instance 0, and the
    // right extractor's index is folded back to 0 by the modulo in the accessor).

    // Per-instance device set. Instance 1's uio indices follow device-tree probe
    // order and can renumber when the overlay changes -- verify against
    // /sys/class/uio/*/name after loading the dual overlay before trusting them.
    // With the append-only dual overlay (instance 0 nodes kept, instance 1 nodes
    // appended after inst 0's two IRQ uio nodes), instance 1's register files land
    // at /dev/uio8 (DMA) and /dev/uio9 (TOP). u-dma-buf is loaded with
    // udmabuf2/udmabuf3 for instance 1 (modprobe params, not the overlay).
    struct HwDeviceSet
    {
        const char* dmaDev;         // AXI DMA control regs (uio)
        const char* topDev;         // TOP register file (uio)
        const char* inDev;          // input u-dma-buf (pixels, cacheable)
        const char* outDev;         // output u-dma-buf (keypoints, uncached)
        const char* inPhysPath;     // sysfs: input phys addr
        const char* outPhysPath;    // sysfs: output phys addr
        const char* inSizePath;     // sysfs: input size
        const char* outSizePath;    // sysfs: output size
        const char* inSyncDir;      // sysfs dir for the input cache-sync controls
    };

    const HwDeviceSet kHwDeviceSets[] = {
        {   // instance 0 -- left / mono / RGBD (DMA on HPC0, udmabuf0/1)
            "/dev/uio4", "/dev/uio5", "/dev/udmabuf0", "/dev/udmabuf1",
            "/sys/class/u-dma-buf/udmabuf0/phys_addr",
            "/sys/class/u-dma-buf/udmabuf1/phys_addr",
            "/sys/class/u-dma-buf/udmabuf0/size",
            "/sys/class/u-dma-buf/udmabuf1/size",
            "/sys/class/u-dma-buf/udmabuf0",
        },
        {   // instance 1 -- right (second DMA on HPC1, udmabuf2/3)
            "/dev/uio8", "/dev/uio9", "/dev/udmabuf2", "/dev/udmabuf3",
            "/sys/class/u-dma-buf/udmabuf2/phys_addr",
            "/sys/class/u-dma-buf/udmabuf3/phys_addr",
            "/sys/class/u-dma-buf/udmabuf2/size",
            "/sys/class/u-dma-buf/udmabuf3/size",
            "/sys/class/u-dma-buf/udmabuf2",
        },
    };

    constexpr int kMaxHwAccelerators =
        static_cast<int>(sizeof(kHwDeviceSets) / sizeof(kHwDeviceSets[0]));
    constexpr int kNumHwAccelerators = ORB_HW_NUM_ACCELERATORS;
    static_assert(kNumHwAccelerators >= 1 && kNumHwAccelerators <= kMaxHwAccelerators,
                  "ORB_HW_NUM_ACCELERATORS must be between 1 and the device-set table size");

    const size_t kDmaMapSize = 0x10000;
    const size_t kTopMapSize = 0x1000;
    const size_t kHwInputMapBytes = 512 * 1024;
    const size_t kHwOutputMapBytes = 256 * 1024;
    const size_t kRecordBytes = 16;
    // 0xC0DE0003: keypoint packing v3 -- word1[63:32] = int32 response (FAST
    //             score zero-extended or signed Harris response, selected at
    //             synthesis by the G_SCORE_TYPE generic; readback THRESH[16]),
    //             angle 24b + flags moved to word2. Older bitstreams pack a
    //             16-bit score and flags in word1 and would be misparsed.
    // 0xC0DE0002: corner-FIFO backpressure (zero-drop) + STALLCNT register.
    const uint32_t kExpectedTopId = 0xC0DE0003u;

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
    const uint32_t THRESH = 0x10;    // RO: [7:0]=perm [15:8]=strict [16]=score type
    const uint32_t KPCOUNT = 0x14;
    const uint32_t DROPCNT = 0x18;
    const uint32_t ID = 0x1C;
    const uint32_t CELLDIM = 0x20;   // [15:0]=wCell  [31:16]=hCell (strict-cell gate)
    const uint32_t CELLNUM = 0x24;   // [15:0]=nCols  [31:16]=nRows (strict-cell gate)
    const uint32_t SUPPCNT = 0x28;   // RO: gate-suppressed permissive-corner count
    const uint32_t STALLCNT = 0x2C;  // RO: pixel beats stalled by corner-FIFO backpressure
                                     //     this frame (PL cycles; the cost of zero drops)

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
        int32_t  response;      // ranking response: FAST score (0..255) or signed
                                // Harris response, per the bitstream's score type
        int32_t  angleQ24;      // raw 24b sign-extended HW angle
        bool     is_brighter;
        bool     passed_strict; // FAST score >= iniThFAST(20) -- in BOTH score modes
    };

    // Selection-cell grid geometry. Single source of truth shared by the HW gate's
    // config-register writes and SelectPerCellHW, so the PL strict-cell gate and the
    // SW per-cell filter tile the image identically. Integer math is bit-identical to
    // the original float form over the valid range (int / == floor for >=0;
    // (a+b-1)/b == ceil(a/b) for positive ints).
    struct CellGrid { int nCols, nRows, wCell, hCell; };

    static inline CellGrid ComputeCellGrid(int cols, int rows)
    {
        const int W      = 35;                          // ORB-SLAM3 grid cell size
        const int border = 2 * (EDGE_THRESHOLD - 3);    // 2*minBorderX
        const int width  = cols - border;
        const int height = rows - border;
        CellGrid g;
        g.nCols = width  / W;
        g.nRows = height / W;
        g.wCell = (g.nCols > 0) ? (width  + g.nCols - 1) / g.nCols : width;
        g.hCell = (g.nRows > 0) ? (height + g.nRows - 1) / g.nRows : height;
        return g;
    }

    // PS-side per-cell strict/permissive filter == ORB-SLAM3's per-cell double-FAST.
    // The PL detects at THRESHOLD_PERMISSIVE(7) over the whole image and tags every
    // corner with passed_strict (score>=20). Per W=35 cell: if ANY corner is strict,
    // keep only the strict ones (== FAST at iniThFAST); else keep all permissive
    // (== the empty-cell minThFAST fallback). Validated bit-exact vs OpenCV FAST
    // (TYPE_9_16) in select_per_cell.cpp. Without this the octree was fed the full
    // permissive flood, producing weak, non-repeatable keypoints (see F.10).
    // Output is in the (absolute - minBorder) frame DistributeOctTree expects, with
    // the HW angle carried in kp.angle (so the HW path skips computeOrientation).
    //
    // O(corners) implementation: a counting sort buckets each corner into its unique
    // cell, then emits per cell. The previous O(corners x cells) scan with a per-cell
    // std::vector added ~51 ms to HW_Total on the threshold-7 flood (E.18). This is
    // behaviour-identical -- same kept set, same cell-row-major emission order (raster
    // within a cell), so vToDistributeKeys matches ORB-SLAM3 exactly -- at O(corners+cells).
    vector<KeyPoint> SelectPerCellHW(int cols, int rows, const vector<PLKeypoint>& plKps)
    {
        const int   EDGE        = EDGE_THRESHOLD;   // 19
        const int   FAST_BORDER = 3;                // empirically confirmed for TYPE_9_16

        // identical to ComputeKeyPointsOctTree
        const int minBorderX = EDGE - 3;
        const int minBorderY = minBorderX;
        const int maxBorderX = cols - EDGE + 3;
        const int maxBorderY = rows - EDGE + 3;

        // Grid geometry from the shared helper == the values written to the HW gate.
        const CellGrid g = ComputeCellGrid(cols, rows);
        const int nCols = g.nCols, nRows = g.nRows;

        vector<KeyPoint> vToDistributeKeys;
        vToDistributeKeys.reserve(plKps.size());
        if(nCols <= 0 || nRows <= 0)
            return vToDistributeKeys;

        const int wCell  = g.wCell;
        const int hCell  = g.hCell;
        const int nCells = nRows * nCols;

        // Map a corner to its unique cell, or -1 ("no cell"). Cell detection regions
        // are contiguous & non-overlapping (FAST_BORDER trims the 6px sub-image
        // overlap), so integer division gives the candidate cell; validating against
        // the (edge-clamped / skipped) region makes this exactly equivalent to the
        // original region scan.
        auto cell_of = [&](int x, int y) -> int
        {
            const int j = (x - minBorderX - FAST_BORDER) / wCell;
            const int i = (y - minBorderY - FAST_BORDER) / hCell;
            if(i < 0 || i >= nRows || j < 0 || j >= nCols) return -1;
            const int iniY = minBorderY + i * hCell;
            const int iniX = minBorderX + j * wCell;
            if(iniY >= maxBorderY - 3) return -1;   // skipped row (too thin at the edge)
            if(iniX >= maxBorderX - 6) return -1;   // skipped col
            int maxY = iniY + hCell + 6; if(maxY > maxBorderY) maxY = maxBorderY;
            int maxX = iniX + wCell + 6; if(maxX > maxBorderX) maxX = maxBorderX;
            if(x < iniX + FAST_BORDER || x > maxX - FAST_BORDER - 1) return -1;
            if(y < iniY + FAST_BORDER || y > maxY - FAST_BORDER - 1) return -1;
            return i * nCols + j;
        };

        // Pass 1: cell id per corner, per-cell "any strict", per-cell histogram.
        vector<int>  cellId(plKps.size());
        vector<char> anyStrict(nCells, 0);
        vector<int>  start(nCells + 1, 0);
        for(size_t k = 0; k < plKps.size(); ++k)
        {
            const int cid = cell_of(static_cast<int>(plKps[k].x),
                                    static_cast<int>(plKps[k].y));
            cellId[k] = cid;
            if(cid >= 0)
            {
                ++start[cid + 1];
                if(plKps[k].passed_strict) anyStrict[cid] = 1;
            }
        }

        // Prefix sum -> per-cell start offsets; start[nCells] = #corners with a cell.
        for(int c = 0; c < nCells; ++c) start[c + 1] += start[c];
        const int total = start[nCells];

        // Pass 2: stable counting sort of corner indices into cell-row-major order
        // (raster order preserved within a cell) -- matches ORB-SLAM3's scan order.
        vector<int> order(total);
        vector<int> pos(start.begin(), start.begin() + nCells);
        for(size_t k = 0; k < plKps.size(); ++k)
            if(cellId[k] >= 0)
                order[pos[cellId[k]]++] = static_cast<int>(k);

        // Pass 3: emit per cell -- if any strict in the cell, keep only strict; else all.
        for(int t = 0; t < total; ++t)
        {
            const PLKeypoint& p = plKps[order[t]];
            if(anyStrict[cellId[order[t]]] && !p.passed_strict) continue;

            KeyPoint kp;
            kp.pt.x     = p.x - minBorderX;
            kp.pt.y     = p.y - minBorderY;
            kp.angle    = HardwareAngleToDegrees(p.angleQ24);  // HW orientation
            kp.response = static_cast<float>(p.response);
            vToDistributeKeys.push_back(kp);
        }
        return vToDistributeKeys;
    }

    // Emit one console line as a SINGLE stream insertion. cerr is unit-buffered,
    // so the whole line reaches the log in one write. Since F.12 the two
    // accelerator instances log from concurrent left/right threads, and the run
    // scripts grep these lines -- a chained << would let fragments from both
    // instances interleave mid-line. ALL driver console output (probe, DROPCNT
    // canary, dump notice, and since 2026-07-10 the TRACE lines as well) must
    // go through here.
    void LogLine(const string& line)
    {
        cerr << line + '\n';
    }

    class OrbHwAccelerator
    {
    public:
        OrbHwAccelerator(const HwDeviceSet& dev, int instanceIndex)
            : index_(instanceIndex),
              dmaFd_(dev.dmaDev, O_RDWR | O_SYNC),    // MMIO regs: uncached
              topFd_(dev.topDev, O_RDWR | O_SYNC),    // MMIO regs: uncached
              inFd_(dev.inDev, O_RDWR),               // input buffer: CACHEABLE (fast memcpy)
              outFd_(dev.outDev, O_RDWR | O_SYNC),    // output buffer: uncached (small reads, always correct)
              inPhys_(ReadU64Auto(dev.inPhysPath)),
              outPhys_(ReadU64Auto(dev.outPhysPath)),
              inSize_(static_cast<size_t>(ReadU64Auto(dev.inSizePath))),
              outSize_(static_cast<size_t>(ReadU64Auto(dev.outSizePath))),
              inMapSize_(min(inSize_, kHwInputMapBytes)),
              outMapSize_(min(outSize_, kHwOutputMapBytes)),
              inSync_(dev.inSyncDir)
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

            // Score type baked into the bitstream (THRESH[16], OpenCV enum:
            // 0 = HARRIS_SCORE, 1 = FAST_SCORE). The record layout is identical
            // either way (int32 response); log it once so runs are attributable.
            const uint32_t thresh = topRegs_.read32(top::THRESH);
            const bool harris = ((thresh >> 16) & 0x1u) == 0;
            ostringstream probe;
            probe << "[USE_HW_ACCEL] instance " << index_
                  << " (" << dev.dmaDev << ", " << dev.topDev << "): score type "
                  << (harris ? "HARRIS_SCORE" : "FAST_SCORE")
                  << " (THRESH=" << Hex32(thresh) << ")";
            LogLine(probe.str());
        }

        HwLevelStats RunLevel(const Mat& image, vector<KeyPoint>& rawKeypoints, int level)
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
            // Trace lines go through LogLine too: chained << from the parallel
            // left/right threads interleaves mid-line, which spliced unrelated
            // numbers after "DROPCNT=" and tripped the run scripts' drop check
            // as a false positive (2026-07-10 2x_fast batch).
            if(traceCount_ < 20)
            {
                ostringstream trace;
                trace << "[USE_HW_ACCEL] begin level=" << level
                      << ", image=" << image.cols << "x" << image.rows
                      << ", inLen=" << inLen
                      << ", outLen=" << outLen;
                LogLine(trace.str());
            }
#endif

            PulseTopSoftReset(topRegs_);

            dmaRegs_.write32(dma::MM2S_DMACR, dma::DmacrReset);
            dmaRegs_.write32(dma::S2MM_DMACR, dma::DmacrReset);
            const double dmaResetUs =
                WaitForDmaResetClear(dmaRegs_, dma::MM2S_DMACR, "MM2S") +
                WaitForDmaResetClear(dmaRegs_, dma::S2MM_DMACR, "S2MM");

#ifdef USE_HW_ACCEL_TRACE
            if(traceCount_ < 20)
                LogLine("[USE_HW_ACCEL] reset done level=" + to_string(level));
#endif

            CopyImageToMapped(inMap_.bytes(), image);
            inSync_.forDevice(inLen);   // flush CPU cache -> DDR before MM2S reads it

#ifdef USE_HW_ACCEL_TRACE
            if(traceCount_ < 20)
                LogLine("[USE_HW_ACCEL] input copied level=" + to_string(level));
#endif

            topRegs_.write32(top::WIDTH, static_cast<uint32_t>(image.cols));
            topRegs_.write32(top::HEIGHT, static_cast<uint32_t>(image.rows));

            // Strict-cell gate geometry (same grid SelectPerCellHW uses). CELLDIM =
            // hCell:wCell, CELLNUM = nRows:nCols, packed 16b each.
            //
            // A/B toggle: ORB_HW_GATE=0 runs the gate INERT (writes cfg=0 -> nCols=0
            // -> the safe check can never assert -> zero suppression), so the SAME
            // bitstream+binary can be run gate-off vs gate-on. Read once (static).
            static const bool gateOn = []{
                const char* e = getenv("ORB_HW_GATE");
                return !(e != nullptr && e[0] == '0');   // ON unless ORB_HW_GATE=0
            }();
            if(gateOn)
            {
                const CellGrid grid = ComputeCellGrid(image.cols, image.rows);
                topRegs_.write32(top::CELLDIM,
                                 (static_cast<uint32_t>(grid.hCell) << 16) |
                                 static_cast<uint16_t>(grid.wCell));
                topRegs_.write32(top::CELLNUM,
                                 (static_cast<uint32_t>(grid.nRows) << 16) |
                                 static_cast<uint16_t>(grid.nCols));
            }
            else
            {
                topRegs_.write32(top::CELLDIM, 0);
                topRegs_.write32(top::CELLNUM, 0);   // nCols=0 -> gate inert
            }

            // Debug: dump the FIRST level-0 image once (set ORB_HW_DUMP=1) so the RTL
            // pyramid TB can be checked bit-exact against a REAL EuRoC frame, not just
            // synthetic vectors. Off by default; harmless.
            static const bool dumpOn = []{
                const char* e = getenv("ORB_HW_DUMP");
                return e != nullptr && e[0] == '1';
            }();
            // atomic exchange: with parallel L/R threads (F.12) both instances can
            // reach the dump width in the same frame -- exactly one may write.
            static atomic<bool> dumped{false};
            if(dumpOn && image.cols == 600 && !dumped.exchange(true))
            {
                ofstream df("hw_dump_L0.hex");
                for(int r = 0; r < image.rows; ++r)
                    for(int c = 0; c < image.cols; ++c)
                    {
                        const int v = static_cast<int>(image.at<uint8_t>(r, c));
                        df << "0123456789abcdef"[v >> 4] << "0123456789abcdef"[v & 0xf] << '\n';
                    }
                ostringstream msg;
                msg << "[ORB_HW_DUMP] wrote hw_dump_L0.hex ("
                    << image.cols << "x" << image.rows << ")";
                LogLine(msg.str());
            }

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
                LogLine("[USE_HW_ACCEL] transfer started level=" + to_string(level));
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
            // Both counters must be read BEFORE clearing CTRL: dropping enable sends
            // the wrapper FSM to S_IDLE, which holds the core (and DROPCNT) in reset.
            const uint32_t dropCount = topRegs_.read32(top::DROPCNT);
            const uint32_t stallCycles = topRegs_.read32(top::STALLCNT);

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
            // Builds since 0xC0DE0002 stall the pixel stream before the corner FIFO
            // can fill, so tail-drop is structurally impossible. A nonzero DROPCNT means
            // the FIFO's PROG_FULL_GAP no longer covers the pipeline tail -- a real
            // hardware regression, so this canary warns in every build (rate-limited).
            if(dropCount != 0)
            {
                if(dropWarningCount_ < 20)
                {
                    ostringstream warn;
                    warn << "[USE_HW_ACCEL] warning: instance " << index_
                         << " TOP DROPCNT=" << dropCount
                         << " (expected 0 with backpressure)"
                         << ", KPCOUNT=" << kpCount
                         << ", level=" << level
                         << ", image=" << image.cols << "x" << image.rows;
                    LogLine(warn.str());
                    if(dropWarningCount_ == 19)
                        LogLine("[USE_HW_ACCEL] suppressing further DROPCNT warnings");
                }
                ++dropWarningCount_;
            }

            const size_t sentinelOffset = static_cast<size_t>(kpCount) * kRecordBytes;
            if(sentinelOffset + kRecordBytes > outLen)
                throw runtime_error("TOP KPCOUNT exceeds udmabuf1 capacity");

            const uint8_t* sentinel = outMap_.bytes() + sentinelOffset;
            if(ReadLe32(sentinel) != 0 || ReadLe32(sentinel + 4) != 0 ||
               ReadLe32(sentinel + 8) != 0 || ReadLe32(sentinel + 12) != 0xFFFFFFFFu)
                throw runtime_error("TOP EOF sentinel mismatch");

            // Decode every PL keypoint in RAW absolute level coords -- NO shift, NO
            // border filter (SelectPerCellHW bins by W=35 cell and applies the shift).
            // Packing v3 (build 0xC0DE0003): word1 = full int32 response, word2 =
            // 24-bit angle + flags at bits 24/25. passed_strict is FAST-threshold-
            // based in BOTH score modes; feeding it to the octree unfiltered was F.10.
            vector<PLKeypoint> plKps;
            plKps.reserve(kpCount);

            for(uint32_t i = 0; i < kpCount; ++i)
            {
                const uint8_t* record = outMap_.bytes() + static_cast<size_t>(i) * kRecordBytes;
                const uint16_t x = ReadLe16(record);
                const uint16_t y = ReadLe16(record + 2);
                const int32_t response = static_cast<int32_t>(ReadLe32(record + 4));
                const uint32_t word2 = ReadLe32(record + 8);
                const uint32_t marker = ReadLe32(record + 12);

                // word2[23:0] = angle, two's complement: sign-extend 24 -> 32.
                const int32_t angleQ24 = (word2 & 0x00800000u)
                                             ? static_cast<int32_t>(word2 | 0xFF000000u)
                                             : static_cast<int32_t>(word2 & 0x00FFFFFFu);

                if(marker != 0)
                    throw runtime_error("nonzero marker in TOP keypoint record");

                PLKeypoint p;
                p.x = static_cast<float>(x);
                p.y = static_cast<float>(y);
                p.response = response;
                p.angleQ24 = angleQ24;
                p.is_brighter = (word2 & 0x01000000u) != 0;   // word2[88]
                p.passed_strict = (word2 & 0x02000000u) != 0; // word2[89] == FAST score>=strict
                plKps.push_back(p);
            }

            // Per-cell strict/permissive selection (== ORB-SLAM3 per-cell double-FAST).
            rawKeypoints = SelectPerCellHW(image.cols, image.rows, plKps);

#ifdef USE_HW_ACCEL_TRACE
            if(traceCount_ < 20)
            {
                ostringstream trace;
                trace << "[USE_HW_ACCEL] done instance=" << index_
                      << ", level=" << level
                      << ", KPCOUNT=" << kpCount
                      << ", kept=" << rawKeypoints.size()
                      << ", DROPCNT=" << dropCount
                      << ", STALLCNT=" << stallCycles;
                LogLine(trace.str());
            }
            ++traceCount_;
#endif
            return HwLevelStats{dmaResetUs, stallCycles};
        }

    private:
        int index_;                 // which accelerator instance this object drives
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
        UdmabufSync inSync_;
        mutex mutex_;
        uint32_t dropWarningCount_ = 0;
#ifdef USE_HW_ACCEL_TRACE
        uint32_t traceCount_ = 0;
#endif
    };

    // Returns the accelerator instance for the given extractor index. In a
    // single-instance build (kNumHwAccelerators == 1) every request collapses to
    // instance 0, so the left/right extractors share one core and serialize on
    // its RunLevel mutex (the pre-F.12 path). Each instance is built lazily and
    // exactly once: mono/RGBD never open instance 1's devices, and the parallel
    // left/right threads can construct their own instances concurrently.
    OrbHwAccelerator& GetOrbHwAccelerator(int requestedIndex)
    {
        const int index = requestedIndex % kNumHwAccelerators;

        static std::unique_ptr<OrbHwAccelerator> instances[kMaxHwAccelerators];
        static std::once_flag initFlags[kMaxHwAccelerators];

        std::call_once(initFlags[index], [index]{
            instances[index].reset(
                new OrbHwAccelerator(kHwDeviceSets[index], index));
        });
        return *instances[index];
    }

}  // namespace

// Public facade -- the only symbol ORBextractor.cc needs. Keeps the whole
// driver (device tables, register maps, DMA sequencing) internal to this
// translation unit so the upstream files only gain a 4-line call site.
namespace orb_hw
{
    HwLevelStats RunLevel(int instanceIndex, const Mat& image,
                          vector<KeyPoint>& vToDistributeKeys, int level)
    {
        return GetOrbHwAccelerator(instanceIndex).RunLevel(image, vToDistributeKeys, level);
    }
}  // namespace orb_hw

} // namespace ORB_SLAM3

#endif // USE_HW_ACCEL
