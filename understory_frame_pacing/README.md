# Understory Frame Pacing

Platform-independent frame pacing models and scheduling primitives.

This crate is the math and policy layer for a render loop. It is `no_std`, has no
platform bindings, and currently does not require `alloc`.

The scheduler answers three questions:

- When should the app wake up?
- Should it start pre-surface work, acquire a surface, or present?
- Should presentation be immediate, at a target time, or after a minimum visible duration?

The API deliberately separates work that can happen before acquiring a drawable
from work that requires the surface. This lets an application build scene state,
animate, cull, or prepare commands early, then acquire the scarce surface later
for final encoding, blitting, submission, and presentation.

`surface_work` is CPU-side work that requires an acquired drawable/swapchain
image. `gpu_work` is the time after submission where the GPU executes the work
needed before the frame is ready to present. The scheduler keeps them separate
because surface work affects when to acquire and how long to hold a scarce
surface, while GPU work affects the submit deadline and presentation readiness.

The timed wake points are `FramePlan::pre_surface_work_start` and
`FramePlan::acquire_surface_at`. After surface work completes, ask the scheduler
again immediately to get the present instruction.

## Example

```rust
use understory_frame_pacing::{
    Action, DisplayTiming, Duration, FrameDemand, FramePacer, FramePhaseReport,
    FrameTimingEstimate, Time,
};

let mut pacer = FramePacer::new(DisplayTiming::fixed(Duration::from_hz(60)));
pacer.set_estimate(FrameTimingEstimate {
    pre_surface_work: Duration::from_millis(4),
    surface_work: Duration::from_millis(2),
    gpu_work: Duration::from_millis(7),
    safety_margin: Duration::from_millis(1),
});

let now = Time::from_nanos(1_000_000_000);
pacer.request_frame(FrameDemand::Animation, now);

let mut action = pacer.next_action(now);
if let Action::SleepUntil(wake_at) = action {
    action = pacer.next_action(wake_at);
}
assert!(action.is_start_pre_surface_work());
pacer.report_phase(FramePhaseReport::pre_surface_work(
    action.frame_id().unwrap(),
    now,
    now + Duration::from_millis(4),
));
```

## Prior Art

- Apple Metal frame pacing guidance: use minimum-duration or target-time
  presentation to avoid micro-stutter when work takes longer than one fixed
  refresh interval.
- Apple Adaptive-Sync guidance: on VRR displays, present evenly at the highest
  sustainable rate inside the display range.
- CADisplayLink guidance: use `targetTimestamp` for animation/simulation timing
  and `targetTimestamp - timestamp` for actual frame interval.
- Chromium compositor scheduling: `BeginFrame` carries frame timing and deadline
  information; frame completion acknowledgements provide back pressure.
