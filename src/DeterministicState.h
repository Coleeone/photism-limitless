#pragma once

#include "PhotismLimitless.h"
#include <vector>
#include <map>
#include <mutex>
#include <memory>
#include <cmath>

struct StateCheckpoint {
    int frameIndex;
    float phase;
    std::vector<float> buffer; // RGBA float linear HDR data
};

class InstanceSimulationState {
public:
    InstanceSimulationState();
    ~InstanceSimulationState();

    void Allocate(int width, int height);
    void Reset();

    // Check if we need re-simulation for frameIndex
    bool CheckSeekAndPrepare(int requestedFrameIndex, float dt, float speedRate,
                             int& outStartFrame, float& outStartPhase);

    // After a simulation step at frameIndex, commit state
    void CommitStep(int frameIndex, float currentPhase, const float* currentBuffer);

    int GetWidth() const { return mWidth; }
    int GetHeight() const { return mHeight; }
    int GetPixelCount() const { return mWidth * mHeight; }

    float* GetCurrentHistory() { return mHistoryA.data(); }
    float* GetNextHistory() { return mHistoryB.data(); }
    void SwapHistory() { mHistoryA.swap(mHistoryB); }

    int GetLastFrameIndex() const { return mLastFrameIndex; }
    float GetPhase() const { return mPhase; }
    void SetPhase(float p) { mPhase = p; }

    int GetActiveMode() const { return mActiveMode; }
    void SetActiveMode(int m) {
        if (m != mActiveMode) {
            mActiveMode = m;
            Reset();
        }
    }

private:
    int mWidth;
    int mHeight;
    int mLastFrameIndex;
    float mPhase;
    int mActiveMode;

    std::vector<float> mHistoryA; // Ping-pong A (RGBA32F)
    std::vector<float> mHistoryB; // Ping-pong B (RGBA32F)

    static const int MAX_CHECKPOINTS = 16;
    static const int CHECKPOINT_INTERVAL = 10;
    static const int MAX_RESIM_FRAMES = 16;
    std::vector<StateCheckpoint> mCheckpoints;

    mutable std::mutex mStateMutex;
};

class StateManager {
public:
    static StateManager& Instance();

    std::shared_ptr<InstanceSimulationState> GetOrCreateState(void* instanceKey, int width, int height);
    void RemoveState(void* instanceKey);

private:
    StateManager() = default;
    std::mutex mMutex;
    std::map<void*, std::shared_ptr<InstanceSimulationState>> mInstances;
};
