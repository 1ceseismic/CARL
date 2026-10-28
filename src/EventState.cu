#include "EventState.cuh"

#include <limits>
#include <stdexcept>
#include <algorithm>

static void orderStreams(cudaStream_t consumer, cudaStream_t producer)
{
    if (consumer == producer) return;

    cudaEvent_t ready;
    CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(ready, producer));
    CUDA_CHECK(cudaStreamWaitEvent(consumer, ready, 0));
    CUDA_CHECK(cudaEventDestroy(ready));
}

void EventState::acquire(intptr_t consumerStream)
{
    const auto consumer = reinterpret_cast<cudaStream_t>(consumerStream);
    orderStreams(consumer, stream);

    if (std::find(consumers.begin(), consumers.end(), consumer) == consumers.end())
    {
        consumers.push_back(consumer);
    }
}

void EventState::waitForConsumers()
{
    for (cudaStream_t consumer : consumers)
    {
        orderStreams(stream, consumer);
    }

    consumers.clear();
}

EventState::EventState(
    int nSim, int nCars, int frameskip, cudaStream_t stream,
    bool captureEvents, bool captureTouches)
    : stream(stream)
    , captureEvents(captureEvents)
    , captureTouches(captureTouches)
    , nSim(nSim)
    , nCars(nCars)
{
    resize(frameskip);
}

void EventState::resize(int frameskip)
{
    if (frameskip < 1 || nCars < 1 || nSim < 1)
    {
        throw std::invalid_argument("invalid event frameskip capacity");
    }

    const int maxElements = std::numeric_limits<int>::max();
    const int maxFrameskip = maxElements / nCars / nSim / TOUCH_VALUE_SIZE;

    if (frameskip > maxFrameskip)
    {
        throw std::invalid_argument("invalid event frameskip capacity");
    }

    if (frameskip == capacity) return;

    waitForConsumers();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const int totalCars = nSim * nCars;
    const int touchSlots = frameskip * nCars;

    if (captureEvents && !carFlags.ptr)
    {
        carFlags = DeviceArray<int>(totalCars);
        ballFlags = DeviceArray<int>(nSim);
        ballSurfaceEntered = DeviceArray<int>(nSim);
        ballSurfaceLeft = DeviceArray<int>(nSim);
        goalSide = DeviceArray<int>(nSim);
        ballSpeedChange = DeviceArray<float>(nSim);
        ballSpeedBefore = DeviceArray<float>(nSim);

        if (!carSurfaces.ptr)
        {
            carSurfaces = DeviceArray<int>(totalCars);
            ballSurfaces = DeviceArray<int>(nSim);

            const size_t carBytes = totalCars * sizeof(int);
            const size_t ballBytes = nSim * sizeof(int);

            CUDA_CHECK(cudaMemsetAsync(carSurfaces.ptr, 0xFF, carBytes, stream));
            CUDA_CHECK(cudaMemsetAsync(ballSurfaces.ptr, 0xFF, ballBytes, stream));
        }
    }

    if (captureTouches)
    {
        const int carSlots = nSim * touchSlots;
        const int metadataCount = carSlots * TOUCH_METADATA_SIZE;
        const int valueCount = carSlots * TOUCH_VALUE_SIZE;

        touchCounts = DeviceArray<int>(nSim);
        touchMetadata = DeviceArray<int>(metadataCount);
        touchValues = DeviceArray<float>(valueCount);
    }

    capacity = frameskip;

    carShape[0] = nSim;
    carShape[1] = nCars;

    ballShape[0] = nSim;
    countShape[0] = nSim;

    touchMetadataShape[0] = nSim;
    touchMetadataShape[1] = touchSlots;
    touchMetadataShape[2] = TOUCH_METADATA_SIZE;

    touchValuesShape[0] = nSim;
    touchValuesShape[1] = touchSlots;
    touchValuesShape[2] = TOUCH_VALUE_SIZE;

    clear();
}

void EventState::clear()
{
    waitForConsumers();

    if (captureEvents)
    {
        const size_t carBytes = carFlags.count * sizeof(int);
        const size_t ballBytes = ballFlags.count * sizeof(int);
        const size_t speedBytes = ballSpeedChange.count * sizeof(float);

        CUDA_CHECK(cudaMemsetAsync(carFlags.ptr, 0, carBytes, stream));
        CUDA_CHECK(cudaMemsetAsync(ballFlags.ptr, 0, ballBytes, stream));
        CUDA_CHECK(cudaMemsetAsync(ballSurfaceEntered.ptr, 0, ballBytes, stream));
        CUDA_CHECK(cudaMemsetAsync(ballSurfaceLeft.ptr, 0, ballBytes, stream));
        CUDA_CHECK(cudaMemsetAsync(goalSide.ptr, 0, ballBytes, stream));
        CUDA_CHECK(cudaMemsetAsync(ballSpeedChange.ptr, 0, speedBytes, stream));
    }

    if (captureTouches)
    {
        CUDA_CHECK(cudaMemsetAsync(touchCounts.ptr, 0, nSim * sizeof(int), stream));
    }
}

void EventState::reset()
{
    clear();
    if (!captureEvents) return;

    const size_t carBytes = carSurfaces.count * sizeof(int);
    const size_t ballBytes = ballSurfaces.count * sizeof(int);

    CUDA_CHECK(cudaMemsetAsync(carSurfaces.ptr, 0xFF, carBytes, stream));
    CUDA_CHECK(cudaMemsetAsync(ballSurfaces.ptr, 0xFF, ballBytes, stream));
}

EventBuffer EventState::buffer() const
{
    EventBuffer result;
    result.carFlags = carFlags.ptr;
    result.ballFlags = ballFlags.ptr;
    result.carSurfaces = carSurfaces.ptr;
    result.ballSurfaces = ballSurfaces.ptr;
    result.ballSurfaceEntered = ballSurfaceEntered.ptr;
    result.ballSurfaceLeft = ballSurfaceLeft.ptr;
    result.goalSide = goalSide.ptr;
    result.ballSpeedChange = ballSpeedChange.ptr;
    result.ballSpeedBefore = ballSpeedBefore.ptr;

    result.touches = {
        touchCounts.ptr, touchMetadata.ptr, touchValues.ptr, capacity * nCars
    };

    result.nSim = nSim;
    result.nCars = nCars;
    
    return result;
}
