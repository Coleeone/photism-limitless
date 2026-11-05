#include "PhotismLimitless.h"
#include "DeterministicState.h"
#include "PrGPUFilterModule.h"
#include "PrSDKVideoSegmentProperties.h"

#if _WIN32
    #include <CL/cl.h>
    #include <cuda_runtime.h>
    #include <vector>

    #define HAS_OPENCL 1
    #define HAS_CUDA   1
#else
    #define HAS_OPENCL 0
    #define HAS_CUDA   0
#endif

// CUDA Kernel Declarations
#if HAS_CUDA
extern "C" void LaunchTrivialKernel_CUDA(
    float4* destImg,
    int destPitch,
    int is16f,
    int width,
    int height,
    float vignetteAmount);

extern "C" void LaunchSmearPipeline_CUDA(
    const float4* srcImg,
    float2* gradBuffer,
    const float4* historyIn,
    float4* historyOut,
    float4* destImg,
    int destPitch,
    int is16f,
    int width,
    int height,
    PhotismParamsGPU params,
    cudaStream_t stream = 0);
#endif

// OpenCL Source Inclusion
#if HAS_OPENCL
static const char* kPhotismKernelOpenCL =
#include "PhotismLimitless.cl"
;

static cl_kernel sTrivialKernelCache[4] = {};
static cl_kernel sSmearKernelCache[4] = {};
#endif

class PhotismLimitlessGPU : public PrGPUFilterBase {
    int mLastMode = -1;
public:
    PhotismLimitlessGPU()
        : mCudaHistoryA(nullptr), mCudaHistoryB(nullptr), mCudaGradBuffer(nullptr),
          mCudaSrcCopy(nullptr), mAllocatedPixels(0), mClHistoryA(nullptr),
          mClHistoryB(nullptr), mClSrcCopy(nullptr) {}

    virtual ~PhotismLimitlessGPU() {
#if HAS_CUDA
        if (mGPUDeviceSuite) {
            if (mCudaHistoryA) mGPUDeviceSuite->FreeDeviceMemory(mDeviceIndex, mCudaHistoryA);
            if (mCudaHistoryB) mGPUDeviceSuite->FreeDeviceMemory(mDeviceIndex, mCudaHistoryB);
            if (mCudaGradBuffer) mGPUDeviceSuite->FreeDeviceMemory(mDeviceIndex, mCudaGradBuffer);
            if (mCudaSrcCopy) mGPUDeviceSuite->FreeDeviceMemory(mDeviceIndex, mCudaSrcCopy);
        }
#endif
#if HAS_OPENCL
        if (mClHistoryA) clReleaseMemObject(mClHistoryA);
        if (mClHistoryB) clReleaseMemObject(mClHistoryB);
        if (mClSrcCopy) clReleaseMemObject(mClSrcCopy);
#endif
    }

    prSuiteError Initialize(PrGPUFilterInstance* ioInstanceData) override {
        prSuiteError res = PrGPUFilterBase::Initialize(ioInstanceData);
        if (res != suiteError_NoError) return res;

        if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_CUDA) {
#if HAS_CUDA
            return suiteError_NoError;
#else
            return suiteError_Fail;
#endif
        } else if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_OpenCL) {
#if HAS_OPENCL
            return InitializeOpenCL();
#else
            return suiteError_Fail;
#endif
        }
        return suiteError_NoError;
    }

    prSuiteError InitializeOpenCL() {
#if HAS_OPENCL
        if (mDeviceIndex >= 4) return suiteError_Fail;

        mCommandQueue = (cl_command_queue)mDeviceInfo.outCommandQueueHandle;
        cl_context context = (cl_context)mDeviceInfo.outContextHandle;
        cl_device_id device = (cl_device_id)mDeviceInfo.outDeviceHandle;

        if (!sTrivialKernelCache[mDeviceIndex] || !sSmearKernelCache[mDeviceIndex]) {
            cl_int result = CL_SUCCESS;
            size_t size = strlen(kPhotismKernelOpenCL);
            cl_program program = clCreateProgramWithSource(context, 1, &kPhotismKernelOpenCL, &size, &result);
            if (result != CL_SUCCESS) return suiteError_Fail;

            result = clBuildProgram(program, 1, &device, "-cl-single-precision-constant -cl-fast-relaxed-math", 0, 0);
            if (result != CL_SUCCESS) return suiteError_Fail;

            sTrivialKernelCache[mDeviceIndex] = clCreateKernel(program, "kTrivialCL", &result);
            if (result != CL_SUCCESS) return suiteError_Fail;

            sSmearKernelCache[mDeviceIndex] = clCreateKernel(program, "kSmearCL", &result);
            if (result != CL_SUCCESS) return suiteError_Fail;
        }
        return suiteError_NoError;
#else
        return suiteError_Fail;
#endif
    }

    prSuiteError Render(
        const PrGPUFilterRenderParams* inRenderParams,
        const PPixHand* inFrames,
        csSDK_size_t inFrameCount,
        PPixHand* outFrame) override
    {
        if (!inRenderParams || !outFrame || !*outFrame) return suiteError_Fail;

        prRect renderRect;
        mPPixSuite->GetBounds(*outFrame, &renderRect);
        int width = renderRect.right - renderRect.left;
        int height = renderRect.bottom - renderRect.top;
        if (width <= 0 || height <= 0) return suiteError_NoError;

        PrPixelFormat pixelFormat = PrPixelFormat_Invalid;
        mPPixSuite->GetPixelFormat(*outFrame, &pixelFormat);
        int is16f = (pixelFormat != PrPixelFormat_GPU_BGRA_4444_32f);
        int bytesPerPx = GetGPUBytesPerPixel(pixelFormat);

        void* destFrameData = nullptr;
        csSDK_int32 destRowBytes = 0;
        mGPUDeviceSuite->GetGPUPPixData(*outFrame, &destFrameData);
        mPPixSuite->GetRowBytes(*outFrame, &destRowBytes);
        int destPitch = destRowBytes / bytesPerPx;

        csSDK_int32 srcRowBytes = destRowBytes;
        void* rawSrcData = destFrameData;
        if (inFrames && inFrameCount > 0 && inFrames[0]) {
            mPPixSuite->GetRowBytes(inFrames[0], &srcRowBytes);
            mGPUDeviceSuite->GetGPUPPixData(inFrames[0], &rawSrcData);
            if (!rawSrcData) {
                rawSrcData = destFrameData;
                srcRowBytes = destRowBytes;
            }
        }
        int srcPitch = width; // We copy into linear mCudaSrcCopy with pitch = width

        // Fetch Effect Parameters
        PrTime time = inRenderParams->inClipTime;
        PhotismParamsGPU params = {};
        params.mode        = GetParam(LIMITLESS_MODE, time).mInt32;
        if (params.mode == 0) params.mode = MODE_SMEAR;

        params.react       = static_cast<float>(GetParam(LIMITLESS_REACT, time).mFloat64);
        params.chaos       = static_cast<float>(GetParam(LIMITLESS_CHAOS, time).mFloat64);
        params.trails      = static_cast<float>(GetParam(LIMITLESS_TRAILS, time).mFloat64);
        params.heat        = static_cast<float>(GetParam(LIMITLESS_HEAT, time).mFloat64);
        params.decay       = static_cast<float>(GetParam(LIMITLESS_DECAY, time).mFloat64);
        params.brightness  = static_cast<float>(GetParam(LIMITLESS_BRIGHTNESS, time).mFloat64);
        params.inputGain   = static_cast<float>(GetParam(LIMITLESS_INPUT_GAIN, time).mFloat64);
        params.symmetry    = static_cast<float>(GetParam(LIMITLESS_SYMMETRY, time).mFloat64);
        params.tiles       = static_cast<float>(GetParam(LIMITLESS_TILES, time).mFloat64);
        params.twist       = static_cast<float>(GetParam(LIMITLESS_TWIST, time).mFloat64);
        params.slice       = static_cast<float>(GetParam(LIMITLESS_SLICE, time).mFloat64);
        params.lens        = static_cast<float>(GetParam(LIMITLESS_LENS, time).mFloat64);
        params.lensStyle   = GetParam(LIMITLESS_LENS_STYLE, time).mInt32;
        params.hueDrift    = static_cast<float>(GetParam(LIMITLESS_HUE_DRIFT, time).mFloat64);
        params.vignette    = static_cast<float>(GetParam(LIMITLESS_VIGNETTE, time).mFloat64);
        params.palette     = GetParam(LIMITLESS_PALETTE, time).mInt32;

        params.speed       = static_cast<float>(GetParam(LIMITLESS_SPEED, time).mFloat64);
        params.shape       = static_cast<float>(GetParam(LIMITLESS_SHAPE, time).mFloat64);
        params.complexity  = static_cast<float>(GetParam(LIMITLESS_COMPLEXITY, time).mFloat64);
        params.fill        = static_cast<float>(GetParam(LIMITLESS_FILL, time).mFloat64);
        params.contrast    = static_cast<float>(GetParam(LIMITLESS_CONTRAST, time).mFloat64);
        params.dryWet      = static_cast<float>(GetParam(LIMITLESS_DRY_WET, time).mFloat64);

        params.driveLow    = static_cast<float>(GetParam(LIMITLESS_DRIVE_LOW, time).mFloat64);
        params.driveMid    = static_cast<float>(GetParam(LIMITLESS_DRIVE_MID, time).mFloat64);
        params.driveHigh   = static_cast<float>(GetParam(LIMITLESS_DRIVE_HIGH, time).mFloat64);
        params.pulse       = static_cast<float>(GetParam(LIMITLESS_PULSE, time).mFloat64);

        params.width       = width;
        params.height      = height;
        params.destPitch   = destPitch;
        params.srcPitch    = srcPitch;
        params.is16f       = is16f;

        // Frame timing & index calculation
        PrTime ticksPerFrame = inRenderParams->inRenderTicksPerFrame > 0 ? inRenderParams->inRenderTicksPerFrame : 4233600000LL;
        params.frameIndex = ticksPerFrame > 0 ? (std::max)(0, static_cast<int>(time / ticksPerFrame)) : 0;
        params.dt = inRenderParams->inRenderTicksPerFrame > 0 ?
                    static_cast<float>(ticksPerFrame) / 254016000000.0f : 0.01667f;

        auto state = StateManager::Instance().GetOrCreateState((void*)(uintptr_t)mNodeID, width, height);
        state->SetActiveMode(params.mode);

        int startFrame = 0;
        float startPhase = 0.0f;
        float speedRate = (params.speed * 0.01f) * (1.0f + (params.react * 0.01f) * (params.driveMid * 0.01f));
        bool seekOccurred = state->CheckSeekAndPrepare(params.frameIndex, params.dt, speedRate, startFrame, startPhase);
        // Deterministic, time-derived phase: identical in preview, scrub and multi-threaded export
        (void)startPhase;
        params.phase = static_cast<float>(params.frameIndex) * params.dt * speedRate;

        // Dispatch to Hardware Framework
        if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_CUDA) {
#if HAS_CUDA
            // Clear any prior sticky/non-sticky error on this thread before doing work
            cudaGetLastError();

            if (!EnsureCUDABuffers(width, height, state)) {
                return suiteError_Fail;
            }

            cudaStream_t stream = reinterpret_cast<cudaStream_t>(mDeviceInfo.outCommandQueueHandle);

            if (seekOccurred || mLastMode != params.mode) {
                // Host history is not downloaded during render (no blocking DMA), so clear instead of uploading stale data
                cudaMemsetAsync(mCudaHistoryA, 0, static_cast<size_t>(width) * height * sizeof(float4), stream);
            }

            // Stage source frame into mCudaSrcCopy on Premiere's active CUDA stream
            cudaMemcpy2DAsync(
                mCudaSrcCopy,
                static_cast<size_t>(width) * bytesPerPx,
                rawSrcData,
                static_cast<size_t>(srcRowBytes),
                static_cast<size_t>(width) * bytesPerPx,
                static_cast<size_t>(height),
                cudaMemcpyDeviceToDevice,
                stream);

            float4* destCUDA = reinterpret_cast<float4*>(destFrameData);
            const float4* srcCUDA = reinterpret_cast<const float4*>(mCudaSrcCopy);

            LaunchSmearPipeline_CUDA(
                srcCUDA,
                reinterpret_cast<float2*>(mCudaGradBuffer),
                reinterpret_cast<const float4*>(mCudaHistoryA),
                reinterpret_cast<float4*>(mCudaHistoryB),
                destCUDA,
                destPitch,
                is16f,
                width,
                height,
                params,
                stream
            );

            // Ping-pong buffer swap
            mLastMode = params.mode;
            void* tmp = mCudaHistoryA;
            mCudaHistoryA = mCudaHistoryB;
            mCudaHistoryB = tmp;

            // Commit lightweight phase & frame index without blocking GPU-to-Host DMA during render/export
            state->CommitStep(params.frameIndex, params.phase, nullptr);

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) return suiteError_Fail;
            return suiteError_NoError;
#else
            return suiteError_NotImplemented;
#endif
        } else if (mDeviceInfo.outDeviceFramework == PrGPUDeviceFramework_OpenCL) {
#if HAS_OPENCL
            cl_mem destBuffer = reinterpret_cast<cl_mem>(destFrameData);
            size_t threadBlock[2] = { 16, 16 };
            size_t grid[2] = { RoundUp(width, threadBlock[0]), RoundUp(height, threadBlock[1]) };

            EnsureOpenCLBuffers(width, height);
            cl_mem rawSrcBuffer = reinterpret_cast<cl_mem>(rawSrcData);
            size_t copySize = static_cast<size_t>(height) * static_cast<size_t>(srcRowBytes);
            size_t maxCopySize = static_cast<size_t>(width) * static_cast<size_t>(height) * sizeof(float) * 4;
            clEnqueueCopyBuffer(mCommandQueue, rawSrcBuffer, mClSrcCopy, 0, 0, (std::min)(copySize, maxCopySize), 0, nullptr, nullptr);

            int srcPitchCL = srcRowBytes / bytesPerPx;
            cl_kernel k = sSmearKernelCache[mDeviceIndex];
            clSetKernelArg(k, 0, sizeof(cl_mem), &mClSrcCopy);
            clSetKernelArg(k, 1, sizeof(cl_mem), &mClHistoryA);
            clSetKernelArg(k, 2, sizeof(cl_mem), &mClHistoryB);
            clSetKernelArg(k, 3, sizeof(cl_mem), &destBuffer);
            clSetKernelArg(k, 4, sizeof(int), &destPitch);
            clSetKernelArg(k, 5, sizeof(int), &is16f);
            clSetKernelArg(k, 6, sizeof(int), &width);
            clSetKernelArg(k, 7, sizeof(int), &height);
            clSetKernelArg(k, 8, sizeof(float), &params.react);
            clSetKernelArg(k, 9, sizeof(float), &params.chaos);
            clSetKernelArg(k, 10, sizeof(float), &params.trails);
            clSetKernelArg(k, 11, sizeof(float), &params.heat);
            clSetKernelArg(k, 12, sizeof(float), &params.decay);
            clSetKernelArg(k, 13, sizeof(float), &params.brightness);
            clSetKernelArg(k, 14, sizeof(float), &params.inputGain);
            clSetKernelArg(k, 15, sizeof(float), &params.speed);
            clSetKernelArg(k, 16, sizeof(float), &params.complexity);
            clSetKernelArg(k, 17, sizeof(float), &params.twist);
            clSetKernelArg(k, 18, sizeof(float), &params.vignette);
            clSetKernelArg(k, 19, sizeof(float), &params.dryWet);
            clSetKernelArg(k, 20, sizeof(float), &params.driveLow);
            clSetKernelArg(k, 21, sizeof(float), &params.driveMid);
            clSetKernelArg(k, 22, sizeof(float), &params.driveHigh);
            clSetKernelArg(k, 23, sizeof(float), &params.pulse);
            clSetKernelArg(k, 24, sizeof(float), &params.phase);
            clSetKernelArg(k, 25, sizeof(float), &params.dt);
            clSetKernelArg(k, 26, sizeof(int), &params.mode);
            clSetKernelArg(k, 27, sizeof(float), &params.symmetry);
            clSetKernelArg(k, 28, sizeof(float), &params.tiles);
            clSetKernelArg(k, 29, sizeof(float), &params.contrast);
            clSetKernelArg(k, 30, sizeof(int), &srcPitchCL);

            cl_int res = clEnqueueNDRangeKernel(mCommandQueue, k, 2, 0, grid, threadBlock, 0, 0, 0);
            if (res != CL_SUCCESS) return suiteError_Fail;

            cl_mem tmp = mClHistoryA;
            mClHistoryA = mClHistoryB;
            mClHistoryB = tmp;
            return suiteError_NoError;
#else
            return suiteError_NotImplemented;
#endif
        }

        return suiteError_NotImplemented;
    }

private:
#if HAS_CUDA
    bool EnsureCUDABuffers(int width, int height, std::shared_ptr<InstanceSimulationState>& state) {
        size_t totalPixels = static_cast<size_t>(width) * height;
        if (totalPixels != mAllocatedPixels) {
            if (mCudaHistoryA) { mGPUDeviceSuite->FreeDeviceMemory(mDeviceIndex, mCudaHistoryA); mCudaHistoryA = nullptr; }
            if (mCudaHistoryB) { mGPUDeviceSuite->FreeDeviceMemory(mDeviceIndex, mCudaHistoryB); mCudaHistoryB = nullptr; }
            if (mCudaGradBuffer) { mGPUDeviceSuite->FreeDeviceMemory(mDeviceIndex, mCudaGradBuffer); mCudaGradBuffer = nullptr; }
            if (mCudaSrcCopy) { mGPUDeviceSuite->FreeDeviceMemory(mDeviceIndex, mCudaSrcCopy); mCudaSrcCopy = nullptr; }

            if (mGPUDeviceSuite->AllocateDeviceMemory(mDeviceIndex, totalPixels * sizeof(float4), &mCudaHistoryA) != suiteError_NoError) return false;
            if (mGPUDeviceSuite->AllocateDeviceMemory(mDeviceIndex, totalPixels * sizeof(float4), &mCudaHistoryB) != suiteError_NoError) return false;
            if (mGPUDeviceSuite->AllocateDeviceMemory(mDeviceIndex, totalPixels * sizeof(float2), &mCudaGradBuffer) != suiteError_NoError) return false;
            if (mGPUDeviceSuite->AllocateDeviceMemory(mDeviceIndex, totalPixels * sizeof(float4), &mCudaSrcCopy) != suiteError_NoError) return false;

            cudaStream_t stream = reinterpret_cast<cudaStream_t>(mDeviceInfo.outCommandQueueHandle);
            cudaMemsetAsync(mCudaHistoryA, 0, totalPixels * sizeof(float4), stream);
            cudaMemsetAsync(mCudaHistoryB, 0, totalPixels * sizeof(float4), stream);
            cudaMemsetAsync(mCudaSrcCopy, 0, totalPixels * sizeof(float4), stream);
            mAllocatedPixels = totalPixels;

            if (state) state->Allocate(width, height);
        }
        return (mCudaHistoryA && mCudaHistoryB && mCudaGradBuffer && mCudaSrcCopy);
    }
#endif

#if HAS_OPENCL
    void EnsureOpenCLBuffers(int width, int height) {
        size_t totalPixels = static_cast<size_t>(width) * height;
        if (totalPixels != mAllocatedPixelsCL) {
            if (mClHistoryA) clReleaseMemObject(mClHistoryA);
            if (mClHistoryB) clReleaseMemObject(mClHistoryB);
            if (mClSrcCopy) clReleaseMemObject(mClSrcCopy);

            cl_context context = (cl_context)mDeviceInfo.outContextHandle;
            mClHistoryA = clCreateBuffer(context, CL_MEM_READ_WRITE, totalPixels * sizeof(float) * 4, nullptr, nullptr);
            mClHistoryB = clCreateBuffer(context, CL_MEM_READ_WRITE, totalPixels * sizeof(float) * 4, nullptr, nullptr);
            mClSrcCopy  = clCreateBuffer(context, CL_MEM_READ_WRITE, totalPixels * sizeof(float) * 4, nullptr, nullptr);
            mAllocatedPixelsCL = totalPixels;
        }
    }
    cl_command_queue mCommandQueue;
    cl_mem mClHistoryA;
    cl_mem mClHistoryB;
    cl_mem mClSrcCopy;
    size_t mAllocatedPixelsCL = 0;
#endif

    void* mCudaHistoryA;
    void* mCudaHistoryB;
    void* mCudaGradBuffer;
    void* mCudaSrcCopy;
    size_t mAllocatedPixels;
};

DECLARE_GPUFILTER_ENTRY(PrGPUFilterModule<PhotismLimitlessGPU>)
