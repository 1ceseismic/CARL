#pragma once

#include "Cuda/Math.cuh"

enum CarlEventType
{
    EVENT_BALL_TOUCH = 1
};

enum BallTouchSurface
{
    TOUCH_FRONT,
    TOUCH_REAR,
    TOUCH_LEFT,
    TOUCH_RIGHT,
    TOUCH_ROOF,
    TOUCH_UNDERSIDE
};

enum BallTouchMetadataField
{
    TOUCH_TICK,
    TOUCH_TYPE,
    TOUCH_CAR,
    TOUCH_SURFACE,
    TOUCH_METADATA_SIZE
};

enum BallTouchValueField
{
    TOUCH_POINT_X,
    TOUCH_POINT_Y,
    TOUCH_POINT_Z,
    TOUCH_NORMAL_X,
    TOUCH_NORMAL_Y,
    TOUCH_NORMAL_Z,
    TOUCH_DEPTH,
    TOUCH_REL_SPEED_BEFORE,
    TOUCH_REL_SPEED_AFTER,
    TOUCH_SPEED_CHANGE,
    TOUCH_ABS_SPEED_CHANGE,
    TOUCH_SOLVER_DELTA_X,
    TOUCH_SOLVER_DELTA_Y,
    TOUCH_SOLVER_DELTA_Z,
    TOUCH_EXTRA_DELTA_X,
    TOUCH_EXTRA_DELTA_Y,
    TOUCH_EXTRA_DELTA_Z,
    TOUCH_VALUE_SIZE
};

struct BallTouchEvent
{
    int tick;
    int car;                    // Per-simulation index, blue first.
    BallTouchSurface surface;
    Vec3 pointLocal;            // Relative to car position, before correction.
    Vec3 normalLocal;           // Points from car toward ball.
    float depth;                // Can be negative within contact tolerance.
    float relativeSpeedBefore;
    float relativeSpeedAfter;   // Before the deferred extra impulse.
    Vec3 solverBallDelta;       // World-space velocity change.
    Vec3 extraBallDelta;        // World-space change queued until tick end.
};

// Non-owning GPU views over [nSim, capacity, fields] arrays.
// The owner clears counts once per API step.
struct BallTouchBuffer
{
    int* counts = nullptr;
    int* metadata = nullptr;
    float* values = nullptr;
    int capacity = 0;

    CARL_HD CARL_FI bool enabled() const { return counts != nullptr; }

    CARL_D CARL_FI void record(int simIdx, const BallTouchEvent& event) const
    {
        if (!enabled()) return;

        const int slot = atomicAdd(counts + simIdx, 1);

        // Counts can exceed capacity to signal overflow to consumers.
        if (slot >= capacity) return;

        const int offset = simIdx * capacity + slot;
        int* meta = metadata + offset * TOUCH_METADATA_SIZE;
        float* data = values + offset * TOUCH_VALUE_SIZE;

        meta[TOUCH_TICK] = event.tick;
        meta[TOUCH_TYPE] = EVENT_BALL_TOUCH;
        meta[TOUCH_CAR] = event.car;
        meta[TOUCH_SURFACE] = event.surface;

        data[TOUCH_POINT_X] = event.pointLocal.x;
        data[TOUCH_POINT_Y] = event.pointLocal.y;
        data[TOUCH_POINT_Z] = event.pointLocal.z;
        data[TOUCH_NORMAL_X] = event.normalLocal.x;
        data[TOUCH_NORMAL_Y] = event.normalLocal.y;
        data[TOUCH_NORMAL_Z] = event.normalLocal.z;
        data[TOUCH_DEPTH] = event.depth;
        data[TOUCH_REL_SPEED_BEFORE] = event.relativeSpeedBefore;
        data[TOUCH_REL_SPEED_AFTER] = event.relativeSpeedAfter;

        const float speedChange =
            event.relativeSpeedAfter - event.relativeSpeedBefore;
            
        data[TOUCH_SPEED_CHANGE] = speedChange;
        data[TOUCH_ABS_SPEED_CHANGE] = fabsf(speedChange);

        data[TOUCH_SOLVER_DELTA_X] = event.solverBallDelta.x;
        data[TOUCH_SOLVER_DELTA_Y] = event.solverBallDelta.y;
        data[TOUCH_SOLVER_DELTA_Z] = event.solverBallDelta.z;
        data[TOUCH_EXTRA_DELTA_X] = event.extraBallDelta.x;
        data[TOUCH_EXTRA_DELTA_Y] = event.extraBallDelta.y;
        data[TOUCH_EXTRA_DELTA_Z] = event.extraBallDelta.z;
    }
};
