#pragma once

#include "Cuda/Math.cuh"

enum CarlEventType
{
    EVENT_BALL_TOUCH = 1 << 0,
    EVENT_JUMP = 1 << 1,
    EVENT_DOUBLE_JUMP = 1 << 2,
    EVENT_DODGE = 1 << 3,
    EVENT_GROUNDED = 1 << 4,
    EVENT_AIRBORNE = 1 << 5,
    EVENT_ENTER_WALL = 1 << 6,
    EVENT_LEAVE_WALL = 1 << 7,
    EVENT_FLIP_RESET = 1 << 8,
    EVENT_DEMOED = 1 << 9,
    EVENT_GOAL = 1 << 10,
    EVENT_BALL_SURFACE_ENTER = 1 << 11,
    EVENT_BALL_SURFACE_LEAVE = 1 << 12,
    EVENT_LEAVE_SURFACE = 1 << 13,
    EVENT_BALL_ACCELERATES = 1 << 14
};

enum ArenaSurface
{
    ARENA_FLOOR = 1 << 0,
    ARENA_SIDE_WALL = 1 << 1,
    ARENA_BACK_WALL = 1 << 2,
    ARENA_CEILING = 1 << 3
};

CARL_HD CARL_FI int arenaSurface(const Vec3& normal)
{
    const float x = fabsf(normal.x);
    const float y = fabsf(normal.y);
    const float z = fabsf(normal.z);

    if (z >= x && z >= y)
    {
        return normal.z >= 0.f ? ARENA_FLOOR : ARENA_CEILING;
    }

    return x >= y ? ARENA_SIDE_WALL : ARENA_BACK_WALL;
}

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

        data[TOUCH_SOLVER_DELTA_X] = event.solverBallDelta.x;
        data[TOUCH_SOLVER_DELTA_Y] = event.solverBallDelta.y;
        data[TOUCH_SOLVER_DELTA_Z] = event.solverBallDelta.z;
        data[TOUCH_EXTRA_DELTA_X] = event.extraBallDelta.x;
        data[TOUCH_EXTRA_DELTA_Y] = event.extraBallDelta.y;
        data[TOUCH_EXTRA_DELTA_Z] = event.extraBallDelta.z;
    }
};

struct EventBuffer
{
    int* carFlags = nullptr;
    int* ballFlags = nullptr;
    int* carSurfaces = nullptr;
    int* ballSurfaces = nullptr;
    int* ballSurfaceEntered = nullptr;
    int* ballSurfaceLeft = nullptr;
    int* goalSide = nullptr;
    float* ballSpeedChange = nullptr;
    float* ballSpeedBefore = nullptr;

    BallTouchBuffer touches;

    int nSim = 0;
    int nCars = 0;

    CARL_HD CARL_FI bool enabled() const { return carFlags != nullptr; }

    CARL_D CARL_FI void car(int carIdx, int flags) const
    {
        if (!enabled()) return;

        carFlags[carIdx] |= flags;
    }

    CARL_D CARL_FI void ball(int simIdx, int flags) const
    {
        if (!enabled()) return;

        ballFlags[simIdx] |= flags;
    }

    CARL_D CARL_FI void invalidate(int simIdx) const
    {
        if (!enabled()) return;

        ballSurfaces[simIdx] = -1;

        for (int local = 0; local < nCars; local++)
        {
            carSurfaces[simIdx * nCars + local] = -1;
        }
    }

    CARL_D CARL_FI void carSurface(int carIdx, int surfaces) const
    {
        if (!enabled()) return;

        const int previous = carSurfaces[carIdx];
        carSurfaces[carIdx] = surfaces;

        if (previous < 0) return;

        const int walls = ARENA_SIDE_WALL | ARENA_BACK_WALL;

        const bool wasGrounded = previous != 0;
        const bool isGrounded = surfaces != 0;
        const bool wasOnWall = (previous & walls) != 0;
        const bool isOnWall = (surfaces & walls) != 0;

        if (!wasGrounded && isGrounded)
        {
            car(carIdx, EVENT_GROUNDED);
        }

        if (wasGrounded && !isGrounded)
        {
            car(carIdx, EVENT_AIRBORNE | EVENT_LEAVE_SURFACE);
        }

        if (!wasOnWall && isOnWall)
        {
            car(carIdx, EVENT_ENTER_WALL);
        }

        if (wasOnWall && !isOnWall)
        {
            car(carIdx, EVENT_LEAVE_WALL);
        }
    }

    CARL_D CARL_FI void ballSurface(int simIdx, int surfaces) const
    {
        if (!enabled()) return;

        const int previous = ballSurfaces[simIdx];
        ballSurfaces[simIdx] = surfaces;

        if (previous < 0) return;

        const int entered = surfaces & ~previous;
        const int left = previous & ~surfaces;

        ballSurfaceEntered[simIdx] |= entered;
        ballSurfaceLeft[simIdx] |= left;

        if (entered)
        {
            ball(simIdx, EVENT_BALL_SURFACE_ENTER);
        }

        if (left)
        {
            ball(simIdx, EVENT_BALL_SURFACE_LEAVE);
        }
    }
};
