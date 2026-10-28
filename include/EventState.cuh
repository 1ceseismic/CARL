#pragma once

#include "Cuda/DeviceArray.cuh"
#include "State/Events.cuh"

#include <vector>
#include <cstdint>

class EventState
{
    cudaStream_t stream;
    bool captureEvents;
    bool captureTouches;
    int nSim;
    int nCars;
    int capacity = 0;
    std::vector<cudaStream_t> consumers;

    void waitForConsumers();

public:
    DeviceArray<int> carFlags;
    DeviceArray<int> ballFlags;
    DeviceArray<int> carSurfaces;
    DeviceArray<int> ballSurfaces;
    DeviceArray<int> ballSurfaceEntered;
    DeviceArray<int> ballSurfaceLeft;
    DeviceArray<int> goalSide;
    DeviceArray<float> ballSpeedChange;
    DeviceArray<float> ballSpeedBefore;
    DeviceArray<int> touchCounts;
    DeviceArray<int> touchMetadata;
    DeviceArray<float> touchValues;

    int64_t carShape[2] = {};
    int64_t ballShape[1] = {};
    int64_t countShape[1] = {};
    int64_t touchMetadataShape[3] = {};
    int64_t touchValuesShape[3] = {};

    EventState(
        int nSim, int nCars, int frameskip, cudaStream_t stream,
        bool captureEvents, bool captureTouches);

    void resize(int frameskip);
    void clear();
    void reset();
    void acquire(intptr_t consumerStream);

    EventBuffer buffer() const;

    bool enabled() const { return captureEvents; }
    bool touchesEnabled() const { return captureTouches; }
};
