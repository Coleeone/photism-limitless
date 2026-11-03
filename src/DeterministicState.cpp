#include "DeterministicState.h"
#include <algorithm>
#include <cstring>

InstanceSimulationState::InstanceSimulationState()
    : mWidth(0), mHeight(0), mLastFrameIndex(-1), mPhase(0.0f), mActiveMode(MODE_SMEAR) {}

InstanceSimulationState::~InstanceSimulationState() {}

void InstanceSimulationState::Allocate(int width, int height) {
    std::lock_guard<std::mutex> lock(mStateMutex);
    if (mWidth != width || mHeight != height) {
        mWidth = width;
        mHeight = height;
        size_t numFloats = static_cast<size_t>(width) * height * 4;
        mHistoryA.assign(numFloats, 0.0f);
        mHistoryB.assign(numFloats, 0.0f);
        mLastFrameIndex = -1;
        mPhase = 0.0f;
        mCheckpoints.clear();
    }
}

void InstanceSimulationState::Reset() {
    std::lock_guard<std::mutex> lock(mStateMutex);
    mLastFrameIndex = -1;
    mPhase = 0.0f;
    std::fill(mHistoryA.begin(), mHistoryA.end(), 0.0f);
    std::fill(mHistoryB.begin(), mHistoryB.end(), 0.0f);
    mCheckpoints.clear();
}

bool InstanceSimulationState::CheckSeekAndPrepare(int requestedFrameIndex, float dt, float speedRate,
                                                 int& outStartFrame, float& outStartPhase) {
    std::lock_guard<std::mutex> lock(mStateMutex);
    // 1. Initial frame or same frame
    if (mLastFrameIndex < 0) {
        mLastFrameIndex = -1;
        mPhase = 0.0f;
        std::fill(mHistoryA.begin(), mHistoryA.end(), 0.0f);
        std::fill(mHistoryB.begin(), mHistoryB.end(), 0.0f);
        mCheckpoints.clear();
        outStartFrame = requestedFrameIndex;
        outStartPhase = 0.0f;
        return false; // clean start
    }

    if (requestedFrameIndex == mLastFrameIndex) {
        outStartFrame = requestedFrameIndex;
        outStartPhase = mPhase;
        return false; // same frame, no sim advance needed
    }

    // 2. Sequential forward playback
    if (requestedFrameIndex == mLastFrameIndex + 1) {
        outStartFrame = requestedFrameIndex;
        mPhase += dt * speedRate;
        outStartPhase = mPhase;
        return false; // normal 1-step advance
    }

    // 3. Seek / discontinuity detected!
    // Find closest checkpoint <= requestedFrameIndex
    int bestIdx = -1;
    int bestDist = 1000000;
    for (size_t i = 0; i < mCheckpoints.size(); ++i) {
        if (mCheckpoints[i].frameIndex <= requestedFrameIndex) {
            int dist = requestedFrameIndex - mCheckpoints[i].frameIndex;
            if (dist < bestDist && dist <= MAX_RESIM_FRAMES) {
                bestDist = dist;
                bestIdx = static_cast<int>(i);
            }
        }
    }

    if (bestIdx >= 0) {
        const auto& cp = mCheckpoints[bestIdx];
        outStartFrame = cp.frameIndex;
        outStartPhase = cp.phase;
        if (cp.buffer.size() == mHistoryA.size()) {
            std::memcpy(mHistoryA.data(), cp.buffer.data(), mHistoryA.size() * sizeof(float));
        }
        return (outStartFrame < requestedFrameIndex); // re-sim needed
    }

    // If no checkpoint within window, start clean black at max(0, requestedFrameIndex - MAX_RESIM_FRAMES)
    outStartFrame = (std::max)(0, requestedFrameIndex - MAX_RESIM_FRAMES);
    outStartPhase = outStartFrame * (dt * speedRate);
    std::fill(mHistoryA.begin(), mHistoryA.end(), 0.0f);
    return (outStartFrame < requestedFrameIndex);
}

void InstanceSimulationState::CommitStep(int frameIndex, float currentPhase, const float* currentBuffer) {
    std::lock_guard<std::mutex> lock(mStateMutex);
    mLastFrameIndex = frameIndex;
    mPhase = currentPhase;

    // Periodically store checkpoint
    if (frameIndex % CHECKPOINT_INTERVAL == 0 && currentBuffer != nullptr) {
        // Prune if full
        if (mCheckpoints.size() >= MAX_CHECKPOINTS) {
            mCheckpoints.erase(mCheckpoints.begin());
        }
        StateCheckpoint cp;
        cp.frameIndex = frameIndex + 1;
        cp.phase = currentPhase;
        cp.buffer.assign(currentBuffer, currentBuffer + mHistoryA.size());
        mCheckpoints.push_back(std::move(cp));
    }
}

StateManager& StateManager::Instance() {
    static StateManager sInstance;
    return sInstance;
}

std::shared_ptr<InstanceSimulationState> StateManager::GetOrCreateState(void* instanceKey, int width, int height) {
    std::lock_guard<std::mutex> lock(mMutex);
    auto it = mInstances.find(instanceKey);
    if (it == mInstances.end()) {
        auto state = std::make_shared<InstanceSimulationState>();
        state->Allocate(width, height);
        mInstances[instanceKey] = state;
        return state;
    }
    it->second->Allocate(width, height);
    return it->second;
}

void StateManager::RemoveState(void* instanceKey) {
    std::lock_guard<std::mutex> lock(mMutex);
    mInstances.erase(instanceKey);
}
