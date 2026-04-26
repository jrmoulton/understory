// Copyright 2026 the Understory Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Understory Frame Pacing: platform-independent frame timing decisions.
//!
//! This crate is the math and policy layer for a render loop. It does not know
//! about Metal, Vulkan, WGPU, Core Animation, browser compositors, windows,
//! threads, or timers. A host platform feeds it display timing, rough phase
//! estimates, frame demand, and observed timings. The scheduler returns the
//! next thing the app should do and when it should wake again.
//!
//! The model is intentionally split around surface acquisition:
//!
//! - [`Action::StartPreSurfaceWork`] starts work that can happen before acquiring a
//!   drawable or swapchain image, such as animation sampling, layout, scene
//!   preparation, culling, and render graph building.
//! - [`Action::AcquireSurface`] is emitted only when the scheduler wants the
//!   host to acquire the scarce surface resource and perform surface-bound work,
//!   such as final encoding, blitting, or submitting.
//! - [`Presentation`] tells the host whether to present as soon as work is done
//!   or request a later presentation time/minimum duration when the platform can
//!   express one.
//!
//! The scheduler keeps CPU surface work separate from GPU work because they
//! constrain different resources. Surface work is CPU-side work that needs an
//! acquired drawable or swapchain image, so doing it too early can hold scarce
//! presentation resources. GPU work happens after submission and determines when
//! the frame can actually be displayed.
//!
//! ## Prior Art
//!
//! The API follows a few durable ideas from platform and browser schedulers:
//!
//! - Apple's Metal guidance recommends presenting at a minimum duration that is
//!   longer than the time required to render when fixed-rate displays would
//!   otherwise micro-stutter.
//! - Apple's variable-refresh guidance recommends presenting evenly at the
//!   highest sustainable rate inside the display's supported range.
//! - `CADisplayLink` exposes a `targetTimestamp`; animation and simulation should
//!   target the time the next frame is expected to appear, not the callback time.
//! - Chromium's compositor uses `BeginFrame` as the signal that input for a frame
//!   is normally complete, carries frame time/interval/deadline through the
//!   pipeline, and uses acknowledgements as back pressure.
//!
//! This crate turns those ideas into small no-std data structures that higher
//! layers can map onto their platform of choice.
//!
//! ## Example
//!
//! ```rust
//! use understory_frame_pacing::{
//!     Action, DisplayTiming, Duration, FrameDemand, FramePacer, FramePhaseReport,
//!     FrameTimingEstimate, Time,
//! };
//!
//! let mut pacer = FramePacer::new(DisplayTiming::fixed(Duration::from_hz(60)));
//! pacer.set_estimate(FrameTimingEstimate {
//!     pre_surface_work: Duration::from_millis(4),
//!     surface_work: Duration::from_millis(2),
//!     gpu_work: Duration::from_millis(7),
//!     safety_margin: Duration::from_millis(1),
//! });
//!
//! let now = Time::from_nanos(1_000_000_000);
//! pacer.request_frame(FrameDemand::Animation, now);
//!
//! let mut action = pacer.next_action(now);
//! if let Action::SleepUntil(wake_at) = action {
//!     action = pacer.next_action(wake_at);
//! }
//! assert!(action.is_start_pre_surface_work());
//! pacer.report_phase(FramePhaseReport::pre_surface_work(action.frame_id().unwrap(), now, now + Duration::from_millis(4)));
//! ```
//!
//! ## `no_std`
//!
//! This crate is `no_std` and currently does not require `alloc`.

#![no_std]

#[cfg(test)]
extern crate std;

use core::cmp::{max, min};
use core::ops::{Add, AddAssign, Sub, SubAssign};

const TIMING_QUANTIZATION_TOLERANCE_NS: u64 = 1_000;
const MAX_FRAME_RATE_DIVISOR: u64 = 16;

/// A monotonic timestamp in nanoseconds.
///
/// The scheduler treats this as an opaque host-time domain. It can represent
/// Mach absolute time converted to nanoseconds, `Instant` deltas from process
/// start, or any other monotonic clock chosen by the embedding application.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Time(i64);

impl Time {
    /// The zero timestamp.
    pub const ZERO: Self = Self(0);

    /// Creates a timestamp from nanoseconds.
    #[must_use]
    pub const fn from_nanos(nanos: i64) -> Self {
        Self(nanos)
    }

    /// Returns this timestamp as nanoseconds.
    #[must_use]
    pub const fn as_nanos(self) -> i64 {
        self.0
    }

    /// Returns the later of two timestamps.
    #[must_use]
    pub fn max(self, other: Self) -> Self {
        if self >= other { self } else { other }
    }
}

impl Add<Duration> for Time {
    type Output = Self;

    fn add(self, rhs: Duration) -> Self::Output {
        Self(self.0.saturating_add_unsigned(rhs.0))
    }
}

impl AddAssign<Duration> for Time {
    fn add_assign(&mut self, rhs: Duration) {
        *self = *self + rhs;
    }
}

impl Sub<Duration> for Time {
    type Output = Self;

    fn sub(self, rhs: Duration) -> Self::Output {
        Self(self.0.saturating_sub_unsigned(rhs.0))
    }
}

impl SubAssign<Duration> for Time {
    fn sub_assign(&mut self, rhs: Duration) {
        *self = *self - rhs;
    }
}

impl Sub<Self> for Time {
    type Output = Duration;

    fn sub(self, rhs: Self) -> Self::Output {
        Duration(self.0.saturating_sub(rhs.0).max(0) as u64)
    }
}

/// A non-negative duration in nanoseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Duration(u64);

impl Duration {
    /// A zero duration.
    pub const ZERO: Self = Self(0);

    /// Creates a duration from nanoseconds.
    #[must_use]
    pub const fn from_nanos(nanos: u64) -> Self {
        Self(nanos)
    }

    /// Creates a duration from microseconds.
    #[must_use]
    pub const fn from_micros(micros: u64) -> Self {
        Self(micros.saturating_mul(1_000))
    }

    /// Creates a duration from milliseconds.
    #[must_use]
    pub const fn from_millis(millis: u64) -> Self {
        Self(millis.saturating_mul(1_000_000))
    }

    /// Creates the interval for an integer hertz value.
    #[must_use]
    pub const fn from_hz(hz: u64) -> Self {
        if hz == 0 {
            Self::ZERO
        } else {
            Self(1_000_000_000 / hz)
        }
    }

    /// Returns this duration as nanoseconds.
    #[must_use]
    pub const fn as_nanos(self) -> u64 {
        self.0
    }

    /// Returns whether the duration is zero.
    #[must_use]
    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }

    /// Saturating integer multiplication.
    #[must_use]
    pub const fn saturating_mul(self, rhs: u64) -> Self {
        Self(self.0.saturating_mul(rhs))
    }

    /// Saturating integer division.
    #[must_use]
    pub const fn div_u64(self, rhs: u64) -> Self {
        if rhs == 0 {
            Self::ZERO
        } else {
            Self(self.0 / rhs)
        }
    }

    /// Clamps this duration to an inclusive range.
    #[must_use]
    pub fn clamp(self, min_value: Self, max_value: Self) -> Self {
        Self(min(max(self.0, min_value.0), max_value.0))
    }
}

impl Add for Duration {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0.saturating_add(rhs.0))
    }
}

impl AddAssign for Duration {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Duration {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0.saturating_sub(rhs.0))
    }
}

impl SubAssign for Duration {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

/// A requested presentation cadence expressed as a target frame interval.
///
/// This is the platform-independent policy for throttling frame opportunities
/// to a layer, animation callback, or producer. Fixed-rate displays quantize the
/// request to an even multiple of the display interval. Variable-refresh
/// displays with an explicit update granularity quantize to that granularity.
/// Displays that only report a broad VRR range are treated like fixed-rate
/// sources at their fastest reported cadence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetFrameCadence {
    target_interval: Duration,
}

impl TargetFrameCadence {
    /// Creates a cadence from a requested fps value.
    ///
    /// Returns `None` for non-finite, zero, or negative values.
    #[must_use]
    pub fn from_fps(fps: f64) -> Option<Self> {
        if !fps.is_finite() || fps <= 0.0 {
            return None;
        }

        let nanos = (1_000_000_000.0 / fps).round();
        if !nanos.is_finite() || nanos <= 0.0 || nanos > u64::MAX as f64 {
            return None;
        }

        Some(Self {
            target_interval: Duration::from_nanos(nanos as u64),
        })
    }

    /// Creates a cadence from a requested target interval.
    #[must_use]
    pub const fn from_interval(target_interval: Duration) -> Option<Self> {
        if target_interval.is_zero() {
            None
        } else {
            Some(Self { target_interval })
        }
    }

    /// Returns the requested target interval.
    #[must_use]
    pub const fn target_interval(self) -> Duration {
        self.target_interval
    }

    /// Returns a cadence for a frame-rate preference on `display`.
    #[must_use]
    pub fn from_preference(
        preference: FrameRatePreference,
        display: DisplayTiming,
    ) -> Option<Self> {
        preference
            .plan(display)
            .map(FrameRatePlan::delivery_interval)
            .and_then(Self::from_interval)
    }

    /// Returns the effective target interval to report to diagnostics and use
    /// for opportunity delivery.
    ///
    /// Fixed-rate displays round down to a stable divisor of the hardware
    /// refresh.
    /// Variable-refresh displays with an explicit update granularity quantize
    /// to that granularity. Displays that only report a broad VRR range round
    /// down from the fastest reported source cadence.
    #[must_use]
    pub fn effective_interval(self, display: DisplayTiming) -> Duration {
        display.choose_interval(self.target_interval)
    }

    /// Returns whether a frame opportunity should be delivered for `frame_index`.
    ///
    /// `frame_index` is expected to be a monotonically increasing display tick
    /// index in the host's frame source. This function is intentionally stateless
    /// so the same policy can be used by UI callbacks, compositor surfaces, and
    /// diagnostics without carrying duplicated scheduler state.
    #[must_use]
    pub fn should_deliver(
        self,
        frame_index: u64,
        display: DisplayTiming,
        tick_interval: Duration,
    ) -> bool {
        should_deliver_interval(frame_index, tick_interval, self.effective_interval(display))
    }
}

/// Source and delivery cadence chosen for a frame-rate preference.
///
/// `source_interval` is the cadence the host should prefer for the shared frame
/// source when this preference is part of the active work set.
/// `delivery_interval` is the cadence for the individual layer, callback, or
/// producer. These can differ on variable-refresh displays: for example, a
/// 48-75 Hz display can drive a 60 Hz source and deliver an `at_most(30)`
/// consumer every other source tick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameRatePlan {
    source_interval: Duration,
    delivery_interval: Duration,
}

impl FrameRatePlan {
    /// Creates a plan from explicit source and delivery intervals.
    #[must_use]
    pub const fn new(source_interval: Duration, delivery_interval: Duration) -> Option<Self> {
        if source_interval.is_zero() || delivery_interval.is_zero() {
            None
        } else {
            Some(Self {
                source_interval,
                delivery_interval,
            })
        }
    }

    /// The desired cadence for the shared frame source.
    #[must_use]
    pub const fn source_interval(self) -> Duration {
        self.source_interval
    }

    /// The desired cadence for this consumer.
    #[must_use]
    pub const fn delivery_interval(self) -> Duration {
        self.delivery_interval
    }

    /// Returns whether a source opportunity should be delivered to this
    /// consumer.
    #[must_use]
    pub fn should_deliver(self, frame_index: u64, source_interval: Duration) -> bool {
        should_deliver_interval(frame_index, source_interval, self.delivery_interval)
    }
}

/// Chooses the fastest source cadence needed by an active group of frame-rate
/// preferences.
///
/// The returned interval is for the shared display/frame source. Hosts should
/// still evaluate each consumer's own [`FrameRatePlan::delivery_interval`] to
/// decide whether that consumer receives a given source opportunity.
#[must_use]
pub fn choose_frame_rate_source_interval(
    preferences: &[FrameRatePreference],
    display: DisplayTiming,
) -> Duration {
    if preferences.is_empty() || preferences.contains(&FrameRatePreference::Full) {
        return display.min_interval();
    }

    let mut best_source = display.min_interval();
    let mut best_score = score_frame_rate_source_interval(best_source, preferences, display);
    for preference in preferences {
        let Some(plan) = preference.plan(display) else {
            return display.min_interval();
        };
        let source = plan.source_interval();
        let score = score_frame_rate_source_interval(source, preferences, display);
        if score < best_score || (score == best_score && source > best_source) {
            best_source = source;
            best_score = score;
        }
    }
    best_source
}

fn score_frame_rate_source_interval(
    source_interval: Duration,
    preferences: &[FrameRatePreference],
    display: DisplayTiming,
) -> u128 {
    preferences
        .iter()
        .copied()
        .filter_map(|preference| preference.plan(display))
        .map(|plan| {
            let delivery =
                quantize_delivery_to_source_interval(plan.delivery_interval(), source_interval);
            tolerant_interval_error(delivery, plan.delivery_interval()) as u128
        })
        .sum()
}

fn quantize_delivery_to_source_interval(
    delivery_interval: Duration,
    source_interval: Duration,
) -> Duration {
    if source_interval.is_zero() || delivery_interval <= source_interval {
        return source_interval;
    }
    source_interval.saturating_mul(rounded_up_multiple_count(
        delivery_interval,
        source_interval,
    ))
}

/// Frame-rate hint and acceptable fallback range.
///
/// This is modeled after platform APIs such as `CAFrameRateRange`, but keeps
/// invalid combinations unrepresentable:
///
/// - [`FrameRatePreference::full`] means no throttling by this policy.
/// - [`FrameRatePreference::at_most`] is a power-saving cap. Fixed-rate displays
///   choose the nearest clean cadence that does not exceed the cap.
/// - [`FrameRatePreference::preferred`] requests a best rate and can be refined
///   with [`FrameRatePreferenceBuilder::minimum`] or
///   [`FrameRatePreferenceBuilder::range`]. If fixed-rate quantization would
///   fall below the minimum acceptable FPS, the policy chooses the next clean
///   higher cadence when possible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameRatePreference {
    /// Deliver every eligible display opportunity.
    Full,
    /// Prefer `preferred_interval`, constrained to an optional acceptable range.
    Preferred {
        /// Fastest acceptable interval, equivalent to the maximum acceptable FPS.
        min_interval: Option<Duration>,
        /// Requested best interval.
        preferred_interval: Duration,
        /// Slowest acceptable interval, equivalent to the minimum acceptable FPS.
        max_interval: Option<Duration>,
    },
}

impl FrameRatePreference {
    /// No throttling by this policy.
    #[must_use]
    pub const fn full() -> Self {
        Self::Full
    }

    /// Runs no faster than `fps`, choosing a clean lower cadence on fixed-rate
    /// displays when `fps` is not directly supported.
    #[must_use]
    pub fn at_most(fps: f64) -> Option<Self> {
        let preferred_interval = interval_from_fps(fps)?;
        Some(Self::Preferred {
            min_interval: Some(preferred_interval),
            preferred_interval,
            max_interval: None,
        })
    }

    /// Starts a preference builder with a preferred FPS.
    #[must_use]
    pub fn preferred(fps: f64) -> Option<FrameRatePreferenceBuilder> {
        Some(FrameRatePreferenceBuilder {
            min_interval: None,
            preferred_interval: interval_from_fps(fps)?,
            max_interval: None,
        })
    }

    /// Starts a preference builder with an acceptable FPS range.
    #[must_use]
    pub fn range(min_fps: f64, max_fps: f64) -> Option<FrameRatePreferenceBuilder> {
        let min_interval = interval_from_fps(max_fps)?;
        let max_interval = interval_from_fps(min_fps)?;
        if min_interval > max_interval {
            return None;
        }
        Some(FrameRatePreferenceBuilder {
            min_interval: Some(min_interval),
            preferred_interval: min_interval,
            max_interval: Some(max_interval),
        })
    }

    /// The requested preferred interval when this preference throttles.
    #[must_use]
    pub const fn preferred_interval(self) -> Option<Duration> {
        match self {
            Self::Full => None,
            Self::Preferred {
                preferred_interval, ..
            } => Some(preferred_interval),
        }
    }

    /// Returns the effective interval to report and use for opportunity delivery.
    #[must_use]
    pub fn effective_interval(self, display: DisplayTiming) -> Option<Duration> {
        self.plan(display).map(FrameRatePlan::delivery_interval)
    }

    /// Returns the source/delivery cadence plan for this preference.
    #[must_use]
    pub fn plan(self, display: DisplayTiming) -> Option<FrameRatePlan> {
        match self {
            Self::Full => None,
            Self::Preferred {
                min_interval,
                preferred_interval,
                max_interval,
            } => {
                let delivery_interval = display.choose_preferred_interval(
                    preferred_interval,
                    min_interval,
                    max_interval,
                );
                FrameRatePlan::new(
                    display.choose_source_interval_for_delivery(delivery_interval),
                    delivery_interval,
                )
            }
        }
    }

    /// Returns whether a frame opportunity should be delivered for `frame_index`.
    #[must_use]
    pub fn should_deliver(
        self,
        frame_index: u64,
        display: DisplayTiming,
        tick_interval: Duration,
    ) -> bool {
        let Some(plan) = self.plan(display) else {
            return true;
        };
        plan.should_deliver(frame_index, tick_interval)
    }
}

/// Builder for [`FrameRatePreference`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameRatePreferenceBuilder {
    min_interval: Option<Duration>,
    preferred_interval: Duration,
    max_interval: Option<Duration>,
}

impl FrameRatePreferenceBuilder {
    /// Sets the preferred FPS.
    #[must_use]
    pub fn preferred(mut self, fps: f64) -> Option<Self> {
        let preferred_interval = interval_from_fps(fps)?;
        if let Some(min_interval) = self.min_interval
            && preferred_interval < min_interval
        {
            return None;
        }
        if let Some(max_interval) = self.max_interval
            && preferred_interval > max_interval
        {
            return None;
        }
        self.preferred_interval = preferred_interval;
        Some(self)
    }

    /// Sets the minimum acceptable FPS.
    #[must_use]
    pub fn minimum(mut self, fps: f64) -> Option<Self> {
        let max_interval = interval_from_fps(fps)?;
        if let Some(min_interval) = self.min_interval
            && min_interval > max_interval
        {
            return None;
        }
        if self.preferred_interval > max_interval {
            return None;
        }
        self.max_interval = Some(max_interval);
        Some(self)
    }

    /// Sets the maximum acceptable FPS.
    #[must_use]
    pub fn maximum(mut self, fps: f64) -> Option<Self> {
        let min_interval = interval_from_fps(fps)?;
        if let Some(max_interval) = self.max_interval
            && min_interval > max_interval
        {
            return None;
        }
        if self.preferred_interval < min_interval {
            return None;
        }
        self.min_interval = Some(min_interval);
        Some(self)
    }

    /// Sets an acceptable FPS range.
    #[must_use]
    pub fn range(mut self, min_fps: f64, max_fps: f64) -> Option<Self> {
        let min_interval = interval_from_fps(max_fps)?;
        let max_interval = interval_from_fps(min_fps)?;
        if min_interval > max_interval {
            return None;
        }
        if self.preferred_interval < min_interval || self.preferred_interval > max_interval {
            return None;
        }
        self.min_interval = Some(min_interval);
        self.max_interval = Some(max_interval);
        Some(self)
    }

    /// Builds the preference.
    #[must_use]
    pub const fn build(self) -> FrameRatePreference {
        FrameRatePreference::Preferred {
            min_interval: self.min_interval,
            preferred_interval: self.preferred_interval,
            max_interval: self.max_interval,
        }
    }
}

fn interval_from_fps(fps: f64) -> Option<Duration> {
    if !fps.is_finite() || fps <= 0.0 {
        return None;
    }

    let nanos = (1_000_000_000.0 / fps).round();
    if !nanos.is_finite() || nanos <= 0.0 || nanos > u64::MAX as f64 {
        return None;
    }

    Some(Duration::from_nanos(nanos as u64))
}

fn should_deliver_interval(
    frame_index: u64,
    tick_interval: Duration,
    target_interval: Duration,
) -> bool {
    if tick_interval.is_zero() || target_interval <= tick_interval {
        return true;
    }

    let display_ns = tick_interval.as_nanos() as u128;
    let target_ns = target_interval.as_nanos() as u128;
    if display_ns == 0 || target_ns == 0 {
        return true;
    }

    let next_tick = (frame_index as u128).saturating_add(1);
    let current = ceil_div(next_tick.saturating_mul(display_ns), target_ns);
    let previous = ceil_div((frame_index as u128).saturating_mul(display_ns), target_ns);
    current > previous
}

const fn ceil_div(value: u128, divisor: u128) -> u128 {
    if divisor == 0 || value == 0 {
        0
    } else {
        1 + ((value - 1) / divisor)
    }
}

/// Display timing constraints observed by the platform layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DisplayTiming {
    min_interval: Duration,
    max_interval: Duration,
    granularity: Option<Duration>,
}

impl DisplayTiming {
    /// Creates a fixed-rate display timing model.
    #[must_use]
    pub const fn fixed(interval: Duration) -> Self {
        Self {
            min_interval: interval,
            max_interval: interval,
            granularity: Some(interval),
        }
    }

    /// Creates a variable-refresh timing model.
    ///
    /// `min_interval` is the fastest interval, `max_interval` is the slowest
    /// interval, and `granularity` can describe fixed supported steps when a
    /// platform reports them. Pass `None` for continuous VRR.
    #[must_use]
    pub fn variable(
        min_interval: Duration,
        max_interval: Duration,
        granularity: Option<Duration>,
    ) -> Self {
        debug_assert!(
            min_interval <= max_interval,
            "minimum interval must not exceed maximum interval"
        );
        Self {
            min_interval,
            max_interval: max(max_interval, min_interval),
            granularity,
        }
    }

    /// The fastest supported frame interval.
    #[must_use]
    pub const fn min_interval(self) -> Duration {
        self.min_interval
    }

    /// The slowest supported frame interval.
    #[must_use]
    pub const fn max_interval(self) -> Duration {
        self.max_interval
    }

    /// The fixed interval step for displays that expose discrete cadences.
    #[must_use]
    pub const fn granularity(self) -> Option<Duration> {
        self.granularity
    }

    /// Returns true for a display with a range of supported intervals.
    #[must_use]
    pub fn is_variable(self) -> bool {
        self.min_interval != self.max_interval
    }

    /// Chooses the smallest supported interval that can contain `needed`.
    #[must_use]
    pub fn choose_interval(self, needed: Duration) -> Duration {
        let clamped = needed.clamp(self.min_interval, self.max_interval);
        match self.granularity {
            Some(step) if !step.is_zero() && step != self.min_interval => {
                let steps = rounded_up_multiple_count(clamped, step);
                Duration(steps.saturating_mul(step.0)).clamp(self.min_interval, self.max_interval)
            }
            _ if self.min_interval == self.max_interval => {
                let interval = self.min_interval;
                if interval.is_zero() {
                    Duration::ZERO
                } else {
                    let multiples = rounded_up_multiple_count(needed, interval);
                    interval.saturating_mul(multiples.max(1))
                }
            }
            None => choose_fixed_multiple_interval(
                needed,
                needed,
                Duration(u64::MAX),
                self.min_interval,
            ),
            _ => clamped,
        }
    }

    /// Chooses a supported interval for a preferred cadence and acceptable
    /// interval range.
    #[must_use]
    pub fn choose_preferred_interval(
        self,
        preferred: Duration,
        min_interval: Option<Duration>,
        max_interval: Option<Duration>,
    ) -> Duration {
        let acceptable_min = max(min_interval.unwrap_or(self.min_interval), self.min_interval);
        let acceptable_max = max_interval.unwrap_or(Duration(u64::MAX));
        if acceptable_min > acceptable_max {
            return self.choose_interval(preferred);
        }
        let preferred = preferred.clamp(acceptable_min, acceptable_max);

        if self.min_interval == self.max_interval {
            return choose_fixed_multiple_interval(
                preferred,
                acceptable_min,
                acceptable_max,
                self.min_interval,
            );
        }

        let direct_min = max(acceptable_min, self.min_interval);
        let direct_max = min(acceptable_max, self.max_interval);
        if direct_min <= direct_max {
            return match self.granularity {
                Some(step) if !step.is_zero() && step != self.min_interval => {
                    choose_nearest_stepped_interval(preferred, direct_min, direct_max, step)
                }
                Some(_) => preferred.clamp(direct_min, direct_max),
                None => choose_fixed_multiple_interval(
                    preferred,
                    acceptable_min,
                    acceptable_max,
                    self.min_interval,
                ),
            };
        }

        if acceptable_min > self.max_interval {
            if self.min_interval != self.max_interval {
                return choose_variable_multiple_delivery_interval(
                    preferred,
                    acceptable_min,
                    acceptable_max,
                    self.min_interval,
                    self.max_interval,
                    self.granularity,
                );
            }
            return choose_fixed_multiple_interval(
                preferred,
                acceptable_min,
                acceptable_max,
                self.max_interval,
            );
        }

        self.min_interval
    }

    /// Chooses the frame-source interval that best supports `delivery_interval`.
    ///
    /// Fixed-rate displays always use the hardware interval. Variable-refresh
    /// displays can choose a faster direct cadence and let a consumer divide it
    /// down. This is how a 48-75 Hz VRR display can support a 30 Hz consumer by
    /// asking the frame source for 60 Hz and delivering every other tick.
    #[must_use]
    pub fn choose_source_interval_for_delivery(self, delivery_interval: Duration) -> Duration {
        if self.min_interval == self.max_interval {
            return self.min_interval;
        }
        if delivery_interval >= self.min_interval && delivery_interval <= self.max_interval {
            return match self.granularity {
                Some(step) if !step.is_zero() && step != self.min_interval => {
                    choose_nearest_stepped_interval(
                        delivery_interval,
                        self.min_interval,
                        self.max_interval,
                        step,
                    )
                }
                Some(_) => delivery_interval,
                None => self.min_interval,
            };
        }

        choose_variable_source_interval_for_delivery(
            delivery_interval,
            self.min_interval,
            self.max_interval,
            self.granularity,
        )
    }
}

fn choose_fixed_multiple_interval(
    preferred: Duration,
    min_interval: Duration,
    max_interval: Duration,
    base_interval: Duration,
) -> Duration {
    if base_interval.is_zero() {
        return Duration::ZERO;
    }

    let floor_multiple = max(1, preferred.0 / base_interval.0);
    let mut candidates = [
        base_interval.saturating_mul(floor_multiple),
        base_interval.saturating_mul(floor_multiple.saturating_add(1)),
    ];
    candidates.sort_by_key(|candidate| interval_error(*candidate, preferred));
    candidates
        .into_iter()
        .find(|candidate| *candidate >= min_interval && *candidate <= max_interval)
        .unwrap_or_else(|| preferred.clamp(min_interval, max_interval))
}

fn choose_variable_multiple_delivery_interval(
    preferred: Duration,
    min_interval: Duration,
    max_interval: Duration,
    source_min_interval: Duration,
    source_max_interval: Duration,
    granularity: Option<Duration>,
) -> Duration {
    let mut best = None;
    for divisor in 1..=MAX_FRAME_RATE_DIVISOR {
        let source = divide_interval(preferred, divisor);
        let source = quantize_variable_source_interval(
            source,
            source_min_interval,
            source_max_interval,
            granularity,
        );
        let delivery = source.saturating_mul(divisor);
        if delivery < min_interval || delivery > max_interval {
            continue;
        }
        let score = FrameRateCandidateScore {
            interval_error: tolerant_interval_error(delivery, preferred),
            divisor,
        };
        if best.is_none_or(|(_, best_score)| score < best_score) {
            best = Some((delivery, score));
        }
    }
    best.map(|(delivery, _)| delivery).unwrap_or_else(|| {
        choose_fixed_multiple_interval(preferred, min_interval, max_interval, source_max_interval)
    })
}

fn choose_variable_source_interval_for_delivery(
    delivery_interval: Duration,
    min_interval: Duration,
    max_interval: Duration,
    granularity: Option<Duration>,
) -> Duration {
    let mut best = None;
    for divisor in 1..=MAX_FRAME_RATE_DIVISOR {
        let source = divide_interval(delivery_interval, divisor);
        let source =
            quantize_variable_source_interval(source, min_interval, max_interval, granularity);
        if source < min_interval || source > max_interval {
            continue;
        }
        let reconstructed_delivery = source.saturating_mul(divisor);
        let score = FrameRateCandidateScore {
            interval_error: tolerant_interval_error(reconstructed_delivery, delivery_interval),
            divisor,
        };
        if best.is_none_or(|(_, best_score)| score < best_score) {
            best = Some((source, score));
        }
    }
    best.map(|(source, _)| source)
        .unwrap_or_else(|| delivery_interval.clamp(min_interval, max_interval))
}

fn quantize_variable_source_interval(
    source: Duration,
    min_interval: Duration,
    max_interval: Duration,
    granularity: Option<Duration>,
) -> Duration {
    let source = source.clamp(min_interval, max_interval);
    match granularity {
        Some(step) if !step.is_zero() && step != min_interval => {
            choose_nearest_stepped_interval(source, min_interval, max_interval, step)
        }
        Some(_) => source,
        None => min_interval,
    }
}

fn divide_interval(interval: Duration, divisor: u64) -> Duration {
    if divisor <= 1 {
        interval
    } else {
        Duration(interval.0.saturating_add(divisor / 2) / divisor)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct FrameRateCandidateScore {
    interval_error: u64,
    divisor: u64,
}

fn choose_nearest_stepped_interval(
    preferred: Duration,
    min_interval: Duration,
    max_interval: Duration,
    step: Duration,
) -> Duration {
    let floor_steps = preferred.0 / step.0;
    let mut candidates = [
        Duration(step.0.saturating_mul(floor_steps)),
        Duration(step.0.saturating_mul(floor_steps.saturating_add(1))),
    ];
    candidates.sort_by_key(|candidate| interval_error(*candidate, preferred));
    candidates
        .into_iter()
        .find(|candidate| *candidate >= min_interval && *candidate <= max_interval)
        .unwrap_or_else(|| preferred.clamp(min_interval, max_interval))
}

fn interval_error(candidate: Duration, preferred: Duration) -> u64 {
    if candidate >= preferred {
        candidate.0.saturating_sub(preferred.0)
    } else {
        preferred.0.saturating_sub(candidate.0)
    }
}

fn tolerant_interval_error(candidate: Duration, preferred: Duration) -> u64 {
    let error = interval_error(candidate, preferred);
    if error <= TIMING_QUANTIZATION_TOLERANCE_NS {
        0
    } else {
        error
    }
}

fn rounded_up_multiple_count(needed: Duration, interval: Duration) -> u64 {
    if interval.is_zero() {
        return 0;
    }

    let floor = needed.0 / interval.0;
    if floor > 0 {
        let floor_duration = interval.0.saturating_mul(floor);
        if needed.0.saturating_sub(floor_duration) <= TIMING_QUANTIZATION_TOLERANCE_NS {
            return floor;
        }
    }

    needed.0.saturating_add(interval.0 - 1) / interval.0
}

bitflags::bitflags! {
    /// Why a frame is requested.
    ///
    /// Multiple causes can be pending at once. The planner derives an effective
    /// policy from the full set instead of requiring callers to collapse demand
    /// before pacing.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct FrameDemand: u8 {
        /// Visual work that should be smooth and evenly paced.
        const Animation = 1 << 0;
        /// Latency-sensitive one-shot user input such as clicks, keyboard, or IME.
        const Input = 1 << 1;
        /// Continuous user input such as scroll, resize, pointer move, gesture,
        /// or stylus drag.
        ///
        /// This is latency-sensitive, but should step down to a sustainable
        /// display cadence when the work is too slow for every hardware tick.
        const ContinuousInput = 1 << 2;
        /// Work that can be delayed to reduce resource usage.
        const Background = 1 << 3;
    }
}

impl FrameDemand {
    /// No frame is currently needed.
    pub const NONE: Self = Self::empty();

    fn policy(self) -> DemandPolicy {
        if self.contains(Self::Input) {
            DemandPolicy::Input
        } else if self.contains(Self::ContinuousInput) {
            DemandPolicy::ContinuousInput
        } else if self.contains(Self::Animation) {
            DemandPolicy::Animation
        } else if self.contains(Self::Background) {
            DemandPolicy::Background
        } else {
            DemandPolicy::None
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DemandPolicy {
    None,
    Animation,
    Input,
    ContinuousInput,
    Background,
}

/// Estimated work durations used to schedule a frame.
///
/// The app should wake at `FramePlan::pre_surface_work_start` to begin
/// [`Action::StartPreSurfaceWork`]. When that work is reported complete, the
/// scheduler may ask the app to sleep again until `FramePlan::acquire_surface_at`
/// and then return [`Action::AcquireSurface`]. After the app reports
/// [`Self::surface_work`] complete, it should ask the scheduler again immediately
/// and handle [`Action::Present`]. [`Self::gpu_work`] is not a wake point; it is
/// the expected/observed GPU execution time after submission.
///
/// ```text
/// pre_surface_work        surface_work             gpu_work
/// scene/layout/rendergraph acquire+encode/finalize  GPU executes
/// |----------------------|------------------------|------------|
/// wake                   wake/acquire             submit       ready/present
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameTimingEstimate {
    /// CPU work that can happen before the surface is acquired.
    ///
    /// Examples include animation sampling, layout, scene preparation, culling,
    /// render graph building, and offscreen command preparation that does not
    /// reference the final drawable.
    pub pre_surface_work: Duration,
    /// CPU work that needs an acquired drawable/swapchain image.
    ///
    /// Examples include acquiring the surface, encoding passes that target the
    /// drawable, final blits into the drawable, and finalizing command buffers
    /// for submission. This is separate from [`Self::gpu_work`] because it
    /// affects how long the app holds a scarce presentation surface.
    pub surface_work: Duration,
    /// GPU execution time for submitted work before the frame is ready.
    ///
    /// Examples include vertex, fragment, compute, resolve, store, and blit
    /// execution. This is separate from [`Self::surface_work`] because the CPU
    /// may be free while the GPU is still determining whether the frame can meet
    /// its presentation target.
    pub gpu_work: Duration,
    /// Extra time reserved for timer, IPC, and scheduler variance.
    pub safety_margin: Duration,
}

impl FrameTimingEstimate {
    /// Returns the estimated duration from wake-up to GPU completion.
    #[must_use]
    pub fn total_work(self) -> Duration {
        self.pre_surface_work + self.surface_work + self.gpu_work + self.safety_margin
    }

    /// Returns the estimated duration from surface acquisition to GPU completion.
    #[must_use]
    pub fn surface_to_ready(self) -> Duration {
        self.surface_work + self.gpu_work + self.safety_margin
    }
}

impl Default for FrameTimingEstimate {
    fn default() -> Self {
        Self {
            pre_surface_work: Duration::from_millis(2),
            surface_work: Duration::from_millis(2),
            gpu_work: Duration::from_millis(4),
            safety_margin: Duration::from_millis(1),
        }
    }
}

/// A platform frame opportunity used for stateless pacing decisions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameOpportunity {
    /// Current host time for the decision.
    pub now: Time,
    /// The platform's next predicted present time, when known.
    pub predicted_present_time: Option<Time>,
    /// The platform's current display-link interval, when known.
    pub frame_interval: Option<Duration>,
    /// The last time this content was presented, if known.
    ///
    /// This is retained for stateful hosts that need diagnostics or future
    /// policy extensions. Stateless planning treats the platform frame
    /// prediction as authoritative and does not synthesize cadence from this
    /// value.
    pub last_present_time: Option<Time>,
    /// A target chosen by an earlier opportunity that has not presented yet.
    ///
    /// Hosts with an external tick source can pass this to avoid re-planning a
    /// slow fixed-rate animation past its already selected cadence slot.
    pub pending_target_present_time: Option<Time>,
}

impl FrameOpportunity {
    /// Creates an opportunity at `now` without platform prediction.
    #[must_use]
    pub const fn new(now: Time) -> Self {
        Self {
            now,
            predicted_present_time: None,
            frame_interval: None,
            last_present_time: None,
            pending_target_present_time: None,
        }
    }
}

/// A stateless frame pacing decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FramePacingDecision {
    /// The ideal visible time for produced content.
    pub target_present_time: Time,
    /// The selected interval for this frame.
    pub frame_interval: Duration,
    /// Time to start independent pre-surface work.
    pub pre_surface_work_start: Time,
    /// Time to acquire the surface and start surface-bound work.
    pub acquire_surface_at: Time,
    /// Deadline for CPU submission so GPU work can complete on time.
    pub submit_deadline: Time,
    /// Backend present instruction.
    pub presentation: Presentation,
}

/// Timing attached to one compositor begin-frame opportunity.
///
/// This mirrors the browser-compositor model where a begin-frame carries the
/// semantic frame time, the deadline for producing output, and the current
/// target interval. The scheduler state machine uses this as a lifecycle token;
/// policy math still lives in [`plan_frame`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BeginFrameTiming {
    /// Current host time when the opportunity was received.
    pub now: Time,
    /// Semantic frame time for animation sampling.
    pub frame_time: Time,
    /// Deadline for producing compositor output for this opportunity.
    pub deadline: Time,
    /// Current target frame interval.
    pub interval: Duration,
}

/// A compositor begin-frame tracked by [`BeginFrameScheduler`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BeginFrame {
    /// Monotonic scheduler sequence number.
    pub sequence: u64,
    /// Demand that caused this frame.
    pub demand: FrameDemand,
    /// Timing for this frame opportunity.
    pub timing: BeginFrameTiming,
}

/// Result of feeding a begin-frame opportunity into [`BeginFrameScheduler`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BeginFrameResult {
    /// No work is pending, so the opportunity was ignored.
    Idle,
    /// Start frame work for this opportunity.
    Start(BeginFrame),
    /// A frame is already active. The opportunity was coalesced into one
    /// pending frame and must not start new render work yet.
    Coalesced {
        /// Currently active frame.
        active: BeginFrame,
        /// Latest pending frame that should run after the active frame ends.
        pending: BeginFrame,
    },
}

/// Result of entering the deadline phase for the active begin-frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BeginFrameDeadlineResult {
    /// No frame is active.
    Idle,
    /// The active frame reached its deadline. The host should commit whatever
    /// output is currently ready, then call [`BeginFrameScheduler::finish_frame`].
    Commit(BeginFrame),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BeginFrameSchedulerPhase {
    Idle,
    InsideBeginFrame(BeginFrame),
    InsideDeadline(BeginFrame),
}

/// A small BeginFrame/deadline state machine.
///
/// This is intentionally narrower than Chromium's full scheduler state machine:
/// it only owns lifecycle and coalescing. Hosts still decide what "render",
/// "commit", or "present" mean. The important invariant is that only one frame
/// can be active at a time; while it is active, later opportunities are
/// coalesced rather than starting overlapping work.
#[derive(Clone, Debug)]
pub struct BeginFrameScheduler {
    phase: BeginFrameSchedulerPhase,
    pending: Option<BeginFrame>,
    next_sequence: u64,
}

impl Default for BeginFrameScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl BeginFrameScheduler {
    /// Creates an idle scheduler.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            phase: BeginFrameSchedulerPhase::Idle,
            pending: None,
            next_sequence: 1,
        }
    }

    /// Returns whether a frame is currently inside begin-frame or deadline
    /// processing.
    #[must_use]
    pub const fn has_active_frame(&self) -> bool {
        !matches!(self.phase, BeginFrameSchedulerPhase::Idle)
    }

    /// Returns the active frame, if any.
    #[must_use]
    pub const fn active_frame(&self) -> Option<BeginFrame> {
        match self.phase {
            BeginFrameSchedulerPhase::Idle => None,
            BeginFrameSchedulerPhase::InsideBeginFrame(frame)
            | BeginFrameSchedulerPhase::InsideDeadline(frame) => Some(frame),
        }
    }

    /// Returns whether a coalesced begin-frame is waiting behind the active
    /// frame.
    #[must_use]
    pub const fn has_pending_frame(&self) -> bool {
        self.pending.is_some()
    }

    /// Feeds a begin-frame opportunity into the scheduler.
    ///
    /// If another frame is active, this stores only the latest opportunity and
    /// returns [`BeginFrameResult::Coalesced`]. This is the back-pressure point
    /// that prevents fixed-rate displays from starting overlapping render work.
    pub fn begin_frame(
        &mut self,
        demand: FrameDemand,
        timing: BeginFrameTiming,
    ) -> BeginFrameResult {
        if demand.is_empty() {
            return BeginFrameResult::Idle;
        }

        let frame = BeginFrame {
            sequence: self.next_sequence,
            demand,
            timing,
        };
        self.next_sequence = self.next_sequence.saturating_add(1).max(1);

        match self.phase {
            BeginFrameSchedulerPhase::Idle => {
                self.phase = BeginFrameSchedulerPhase::InsideBeginFrame(frame);
                BeginFrameResult::Start(frame)
            }
            BeginFrameSchedulerPhase::InsideBeginFrame(active)
            | BeginFrameSchedulerPhase::InsideDeadline(active) => {
                self.pending = Some(frame);
                BeginFrameResult::Coalesced {
                    active,
                    pending: frame,
                }
            }
        }
    }

    /// Enters the active frame's deadline phase.
    pub fn begin_deadline(&mut self) -> BeginFrameDeadlineResult {
        match self.phase {
            BeginFrameSchedulerPhase::Idle => BeginFrameDeadlineResult::Idle,
            BeginFrameSchedulerPhase::InsideBeginFrame(frame) => {
                self.phase = BeginFrameSchedulerPhase::InsideDeadline(frame);
                BeginFrameDeadlineResult::Commit(frame)
            }
            BeginFrameSchedulerPhase::InsideDeadline(frame) => {
                BeginFrameDeadlineResult::Commit(frame)
            }
        }
    }

    /// Marks the active frame complete and returns the latest coalesced frame,
    /// if one exists.
    ///
    /// The returned frame is advisory. Hosts with an external display-link
    /// source usually wait for the next real tick before starting it, while
    /// synthetic schedulers may choose to post a task to process it promptly.
    pub fn finish_frame(&mut self) -> Option<BeginFrame> {
        if matches!(self.phase, BeginFrameSchedulerPhase::Idle) {
            return self.pending.take();
        }
        self.phase = BeginFrameSchedulerPhase::Idle;
        self.pending.take()
    }

    /// Drops the coalesced pending frame, if any.
    pub fn clear_pending_frame(&mut self) {
        self.pending = None;
    }

    /// Clears active and pending state.
    pub fn reset(&mut self) {
        self.phase = BeginFrameSchedulerPhase::Idle;
        self.pending = None;
    }
}

/// Work visible to the compositor-frame scheduler.
///
/// This is intentionally a snapshot of host state, not scheduler state. The
/// embedding UI/compositor owns facts like dirty style, pending scene jobs, and
/// commit-ready layer changes; the scheduler owns only frame lifecycle.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompositorWorkStatus {
    /// The host can produce or commit frames.
    pub can_draw: bool,
    /// There is work that should run at begin-frame time: animation callbacks,
    /// style, layout, paint, scene construction, or compositor-surface pulls.
    pub needs_frame_work: bool,
    /// Scene/render jobs from the active frame are still in flight.
    pub scene_jobs_pending: bool,
    /// There is compositor output ready to commit.
    pub commit_ready: bool,
}

impl CompositorWorkStatus {
    /// Returns true when the scheduler should request frame opportunities.
    #[must_use]
    pub const fn needs_frame_source(self) -> bool {
        self.can_draw && (self.needs_frame_work || self.scene_jobs_pending || self.commit_ready)
    }
}

/// Why a compositor commit is requested.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompositorCommitReason {
    /// All required render jobs completed before the deadline.
    SceneReady,
    /// The frame deadline fired; commit ready output or old visible state.
    Deadline,
    /// A commit-ready frame was carried into the next scheduler turn.
    ReadyCarry,
}

/// Result of attempting a compositor commit requested by the scheduler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompositorCommitResult {
    /// Output was committed for the active frame.
    Committed,
    /// The host had no ready output to commit.
    NoWork,
    /// The host reached the deadline, but required scene output is still
    /// unavailable.
    ///
    /// This keeps the active frame alive. When scene output becomes ready, the
    /// scheduler will request the deadline commit instead of dropping the
    /// frame or starting overlapping work.
    ScenePending,
    /// The host could not commit the ready output in the current opportunity.
    ///
    /// This is explicit carry state. Ready output must remain live and the
    /// scheduler must keep requesting frame opportunities instead of letting the
    /// host silently drop the publication.
    Rejected,
}

/// A deadline for the active compositor frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompositorDeadline {
    /// Begin-frame sequence this deadline belongs to.
    pub frame_sequence: u64,
    /// Absolute host-time deadline.
    pub deadline: Time,
}

/// Result of a compositor scheduler event.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompositorFrameAction {
    /// No scheduler action is currently needed.
    Idle,
    /// The host should start begin-frame work for this frame.
    StartFrame(BeginFrame),
    /// A begin-frame arrived while another frame was active. The latest one was
    /// retained and should run after the active frame finishes.
    Coalesced {
        /// Active frame still owning the pipeline.
        active: BeginFrame,
        /// Latest pending frame retained by the scheduler.
        pending: BeginFrame,
    },
    /// Arm or replace the active frame deadline.
    ArmDeadline(CompositorDeadline),
    /// Commit compositor output now.
    Commit {
        /// Active frame for the commit.
        frame: BeginFrame,
        /// Reason the commit was requested.
        reason: CompositorCommitReason,
    },
    /// The active frame completed. If `pending` is present, the host should post
    /// a task to process it promptly, matching Chromium's pending/retro
    /// BeginFrame behavior.
    FinishFrame {
        /// Latest pending begin-frame retained while the active frame was busy.
        pending: Option<BeginFrame>,
    },
}

/// Chromium-style compositor-frame lifecycle scheduler.
///
/// The scheduler owns requested demand, active/pending BeginFrame lifecycle,
/// active deadline state, and pending-frame handoff. It does not own rendering,
/// layer trees, timers, or platform frame sources; hosts feed it snapshots and
/// execute returned actions.
#[derive(Clone, Debug)]
pub struct CompositorFrameScheduler {
    begin_frames: BeginFrameScheduler,
    pending_demand: FrameDemand,
    active_deadline: Option<CompositorDeadline>,
    deadline_waiting_for_scene: bool,
    ready_carry: bool,
}

impl Default for CompositorFrameScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl CompositorFrameScheduler {
    /// Creates an idle scheduler.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            begin_frames: BeginFrameScheduler::new(),
            pending_demand: FrameDemand::NONE,
            active_deadline: None,
            deadline_waiting_for_scene: false,
            ready_carry: false,
        }
    }

    /// Adds frame demand. Multiple calls coalesce until a frame starts.
    pub fn request_frame(&mut self, demand: FrameDemand) {
        self.pending_demand.insert(demand);
    }

    /// Returns true if a frame is active.
    #[must_use]
    pub const fn has_active_frame(&self) -> bool {
        self.begin_frames.has_active_frame()
    }

    /// Returns true if a coalesced frame is waiting behind the active frame.
    #[must_use]
    pub const fn has_pending_frame(&self) -> bool {
        self.begin_frames.has_pending_frame()
    }

    /// Returns the active frame if one exists.
    #[must_use]
    pub const fn active_frame(&self) -> Option<BeginFrame> {
        self.begin_frames.active_frame()
    }

    /// Returns the active deadline if one has been armed.
    #[must_use]
    pub const fn active_deadline(&self) -> Option<CompositorDeadline> {
        self.active_deadline
    }

    /// Returns true if the host should keep its frame source active.
    ///
    /// Chromium only stops observing BeginFrames once the scheduler is idle. Do
    /// the same: active frames, pending frames, or external ready work keep the
    /// source alive even if no new app damage is currently queued.
    #[must_use]
    pub fn needs_frame_source(&self, status: CompositorWorkStatus) -> bool {
        status.can_draw
            && (!self.pending_demand.is_empty()
                || self.begin_frames.has_active_frame()
                || self.begin_frames.has_pending_frame()
                || self.ready_carry
                || status.needs_frame_source())
    }

    /// Handles a platform BeginFrame opportunity.
    ///
    /// If the scheduler is busy, only the latest opportunity is retained and the
    /// returned action is [`CompositorFrameAction::Coalesced`]. Hosts should not
    /// start new frame work until the active frame finishes.
    pub fn on_begin_frame(
        &mut self,
        timing: BeginFrameTiming,
        status: CompositorWorkStatus,
    ) -> CompositorFrameAction {
        if !status.can_draw {
            return CompositorFrameAction::Idle;
        }

        if (self.ready_carry || status.commit_ready)
            && !status.scene_jobs_pending
            && let Some(frame) = self.begin_frames.active_frame()
        {
            return CompositorFrameAction::Commit {
                frame,
                reason: CompositorCommitReason::ReadyCarry,
            };
        }

        let demand = self.effective_demand(status);
        if demand.is_empty() {
            return CompositorFrameAction::Idle;
        }
        self.pending_demand = FrameDemand::NONE;

        match self.begin_frames.begin_frame(demand, timing) {
            BeginFrameResult::Idle => CompositorFrameAction::Idle,
            BeginFrameResult::Start(frame) => CompositorFrameAction::StartFrame(frame),
            BeginFrameResult::Coalesced { active, pending } => {
                CompositorFrameAction::Coalesced { active, pending }
            }
        }
    }

    fn effective_demand(&self, status: CompositorWorkStatus) -> FrameDemand {
        let mut demand = self.pending_demand;
        if status.needs_frame_work || status.scene_jobs_pending || status.commit_ready {
            demand.insert(FrameDemand::Animation);
        }
        demand
    }

    /// Reports that the host submitted frame work and should wait for scene
    /// readiness or deadline.
    ///
    /// If there are no pending scene jobs and output is already ready, this
    /// returns a commit action immediately instead of arming a deadline.
    pub fn on_frame_submitted(
        &mut self,
        submit_deadline: Time,
        status: CompositorWorkStatus,
    ) -> CompositorFrameAction {
        let Some(frame) = self.begin_frames.active_frame() else {
            return CompositorFrameAction::Idle;
        };
        if status.commit_ready && !status.scene_jobs_pending {
            return CompositorFrameAction::Commit {
                frame,
                reason: CompositorCommitReason::SceneReady,
            };
        }
        let deadline = CompositorDeadline {
            frame_sequence: frame.sequence,
            deadline: submit_deadline,
        };
        self.active_deadline = Some(deadline);
        CompositorFrameAction::ArmDeadline(deadline)
    }

    /// Reports scene/render job readiness.
    ///
    /// Ready output commits immediately if it arrives before the active
    /// deadline. Late output waits for the deadline/next scheduler turn, which
    /// prevents surprise second commits inside one frame opportunity.
    pub fn on_scene_ready(
        &mut self,
        now: Time,
        status: CompositorWorkStatus,
    ) -> CompositorFrameAction {
        let Some(frame) = self.begin_frames.active_frame() else {
            return CompositorFrameAction::Idle;
        };
        if status.scene_jobs_pending || !status.commit_ready {
            return CompositorFrameAction::Idle;
        }
        if self.deadline_waiting_for_scene {
            return CompositorFrameAction::Commit {
                frame,
                reason: CompositorCommitReason::Deadline,
            };
        }
        if self
            .active_deadline
            .is_none_or(|deadline| now <= deadline.deadline)
        {
            return CompositorFrameAction::Commit {
                frame,
                reason: CompositorCommitReason::SceneReady,
            };
        }
        CompositorFrameAction::Idle
    }

    /// Handles the active frame deadline.
    ///
    /// The returned commit action does not imply the host has new output ready.
    /// Hosts that cannot commit because required scene output is still pending
    /// must report [`CompositorCommitResult::ScenePending`] so the scheduler
    /// keeps the active frame alive until that scene output completes.
    pub fn on_deadline(&mut self) -> CompositorFrameAction {
        let BeginFrameDeadlineResult::Commit(frame) = self.begin_frames.begin_deadline() else {
            return CompositorFrameAction::Idle;
        };
        CompositorFrameAction::Commit {
            frame,
            reason: CompositorCommitReason::Deadline,
        }
    }

    /// Reports the result of a commit action returned by the scheduler.
    pub fn on_commit_attempt(&mut self, result: CompositorCommitResult) -> CompositorFrameAction {
        match result {
            CompositorCommitResult::Committed => {
                self.ready_carry = false;
                self.on_commit_complete()
            }
            CompositorCommitResult::NoWork => self.on_commit_complete(),
            CompositorCommitResult::ScenePending => {
                self.deadline_waiting_for_scene = true;
                CompositorFrameAction::Idle
            }
            CompositorCommitResult::Rejected => {
                self.ready_carry = true;
                self.pending_demand.insert(FrameDemand::Animation);
                CompositorFrameAction::Idle
            }
        }
    }

    /// Reports that the active frame is complete.
    pub fn on_commit_complete(&mut self) -> CompositorFrameAction {
        self.active_deadline = None;
        self.deadline_waiting_for_scene = false;
        self.ready_carry = false;
        let pending = self.begin_frames.finish_frame();
        if let Some(pending) = pending {
            self.pending_demand.insert(pending.demand);
        }
        CompositorFrameAction::FinishFrame { pending }
    }

    /// Clears active and pending scheduler state.
    pub fn reset(&mut self) {
        self.begin_frames.reset();
        self.pending_demand = FrameDemand::NONE;
        self.active_deadline = None;
        self.deadline_waiting_for_scene = false;
        self.ready_carry = false;
    }
}

/// Computes a platform-independent pacing decision for one frame.
///
/// This is the pure policy function behind [`FramePacer`]. Hosts that already
/// own a frame-opportunity source, such as a display-link callback or browser
/// `requestAnimationFrame`, can use this directly without adopting the stateful
/// pacer.
#[must_use]
pub fn plan_frame(
    display: DisplayTiming,
    estimate: FrameTimingEstimate,
    demand: FrameDemand,
    opportunity: FrameOpportunity,
) -> FramePacingDecision {
    let policy = demand.policy();
    let mut needed = estimate.total_work();
    if matches!(policy, DemandPolicy::Animation | DemandPolicy::Background) {
        // Animation wants an even cadence, not "barely fits". Reserve part of
        // the fastest interval so fixed displays naturally choose stable
        // divisors and VRR displays avoid hovering at an unsustainable edge.
        needed += display.min_interval.div_u64(4);
    }
    let platform_interval = opportunity
        .frame_interval
        .unwrap_or(display.min_interval)
        .clamp(display.min_interval, display.max_interval);
    let selected_interval = if policy == DemandPolicy::Input
        || (!display.is_variable()
            && matches!(
                policy,
                DemandPolicy::ContinuousInput | DemandPolicy::Animation | DemandPolicy::Background
            )) {
        platform_interval
    } else if display.is_variable() {
        display.choose_interval(needed).max(platform_interval)
    } else {
        display.choose_interval(needed)
    };
    let platform_present = opportunity
        .predicted_present_time
        .unwrap_or(opportunity.now + platform_interval)
        .max(opportunity.now);
    if !matches!(policy, DemandPolicy::Input | DemandPolicy::ContinuousInput)
        && selected_interval > platform_interval
        && let Some(pending_target) = opportunity.pending_target_present_time
        && pending_target >= opportunity.now + estimate.total_work()
        && opportunity
            .last_present_time
            .is_none_or(|last_present| pending_target >= last_present + selected_interval)
        && pending_target >= platform_present
        && pending_target > opportunity.now
    {
        let pre_surface_work_start = pending_target - estimate.total_work();
        let acquire_surface_at = pending_target - estimate.surface_to_ready();
        let submit_deadline = pending_target - estimate.gpu_work;
        let presentation = if display.is_variable() {
            Presentation::At(pending_target)
        } else {
            Presentation::AfterMinimumDuration(selected_interval)
        };
        return FramePacingDecision {
            target_present_time: pending_target,
            frame_interval: selected_interval,
            pre_surface_work_start,
            acquire_surface_at,
            submit_deadline,
            presentation,
        };
    }
    let cadence_present = if selected_interval > platform_interval {
        platform_present + (selected_interval - platform_interval)
    } else {
        platform_present
    };
    let mut target_present_time = match policy {
        DemandPolicy::Input => platform_present.max(opportunity.now + estimate.total_work()),
        DemandPolicy::ContinuousInput
        | DemandPolicy::Animation
        | DemandPolicy::Background
        | DemandPolicy::None => cadence_present,
    };
    if matches!(
        policy,
        DemandPolicy::ContinuousInput
            | DemandPolicy::Animation
            | DemandPolicy::Background
            | DemandPolicy::None
    ) {
        let minimum_target = opportunity.now + estimate.total_work();
        while target_present_time < minimum_target {
            target_present_time = target_present_time + selected_interval;
        }
    }
    let pre_surface_work_start = target_present_time - estimate.total_work();
    let acquire_surface_at = target_present_time - estimate.surface_to_ready();
    let submit_deadline = target_present_time - estimate.gpu_work;
    let presentation = match policy {
        DemandPolicy::Input => Presentation::AsSoonAsReady,
        DemandPolicy::ContinuousInput if selected_interval == platform_interval => {
            Presentation::AsSoonAsReady
        }
        DemandPolicy::ContinuousInput
        | DemandPolicy::Animation
        | DemandPolicy::Background
        | DemandPolicy::None => Presentation::AfterMinimumDuration(selected_interval),
    };

    FramePacingDecision {
        target_present_time,
        frame_interval: selected_interval,
        pre_surface_work_start,
        acquire_surface_at,
        submit_deadline,
        presentation,
    }
}

/// A scheduler-selected frame identifier.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FrameId(u64);

impl FrameId {
    /// Returns the raw numeric identifier.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// A planned frame target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FramePlan {
    /// Stable frame identifier for phase reports.
    pub id: FrameId,
    /// The demand that selected this frame.
    pub demand: FrameDemand,
    /// The ideal visible time for the produced content.
    pub target_present_time: Time,
    /// The selected display interval for this frame.
    pub frame_interval: Duration,
    /// Time at which the app should wake to begin independent pre-surface work.
    pub pre_surface_work_start: Time,
    /// Time at which the app should wake again and acquire the surface.
    pub acquire_surface_at: Time,
    /// Deadline for finishing surface CPU work and submitting GPU work.
    pub submit_deadline: Time,
}

/// Presentation instruction for platforms that can control present timing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Presentation {
    /// Present as soon as rendering is complete.
    AsSoonAsReady,
    /// Present no earlier than a host timestamp.
    At(Time),
    /// Present after the previous frame has remained visible for this duration.
    AfterMinimumDuration(Duration),
}

/// The next scheduler action for the host app.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// No work is pending.
    Idle,
    /// Sleep until this timestamp before asking the scheduler again.
    SleepUntil(Time),
    /// Start work that does not require an acquired surface.
    ///
    /// The app receives this action at or after [`FramePlan::pre_surface_work_start`].
    StartPreSurfaceWork(FramePlan),
    /// Acquire a drawable/swapchain image and perform surface-bound work.
    ///
    /// The app receives this action at or after [`FramePlan::acquire_surface_at`].
    AcquireSurface(FramePlan),
    /// Submit/present the finished frame using the provided presentation mode.
    ///
    /// This is returned immediately after surface work is reported complete. It
    /// is not a separate timed wake point.
    Present {
        /// Planned frame.
        plan: FramePlan,
        /// Presentation instruction.
        presentation: Presentation,
    },
}

impl Action {
    /// Returns true if this action starts independent pre-surface work.
    #[must_use]
    pub const fn is_start_pre_surface_work(self) -> bool {
        matches!(self, Self::StartPreSurfaceWork(_))
    }

    /// Returns the frame id carried by frame-specific actions.
    #[must_use]
    pub const fn frame_id(self) -> Option<FrameId> {
        match self {
            Self::StartPreSurfaceWork(plan)
            | Self::AcquireSurface(plan)
            | Self::Present { plan, .. } => Some(plan.id),
            Self::Idle | Self::SleepUntil(_) => None,
        }
    }
}

/// A measured phase that updates the scheduler's timing model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FramePhaseReport {
    frame_id: FrameId,
    phase: Phase,
    start: Time,
    end: Time,
}

impl FramePhaseReport {
    /// Reports independent pre-surface-work timing.
    #[must_use]
    pub const fn pre_surface_work(frame_id: FrameId, start: Time, end: Time) -> Self {
        Self {
            frame_id,
            phase: Phase::Frame,
            start,
            end,
        }
    }

    /// Reports surface-bound CPU work timing.
    #[must_use]
    pub const fn surface_work(frame_id: FrameId, start: Time, end: Time) -> Self {
        Self {
            frame_id,
            phase: Phase::Surface,
            start,
            end,
        }
    }

    /// Reports GPU work timing.
    #[must_use]
    pub const fn gpu_work(frame_id: FrameId, start: Time, end: Time) -> Self {
        Self {
            frame_id,
            phase: Phase::Gpu,
            start,
            end,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Frame,
    Surface,
    Gpu,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FrameState {
    Idle,
    Planned(FramePlan),
    PreSurfaceWorkStarted(FramePlan),
    PreSurfaceWorkDone(FramePlan),
    SurfaceAcquired(FramePlan),
    SurfaceWorkDone(FramePlan),
}

/// A platform-independent frame scheduler.
#[derive(Clone, Debug)]
pub struct FramePacer {
    display: DisplayTiming,
    estimate: FrameTimingEstimate,
    next_frame_id: u64,
    last_present_time: Option<Time>,
    pending_demand: FrameDemand,
    state: FrameState,
}

impl FramePacer {
    /// Creates a frame pacer for a display.
    #[must_use]
    pub fn new(display: DisplayTiming) -> Self {
        Self {
            display,
            estimate: FrameTimingEstimate::default(),
            next_frame_id: 1,
            last_present_time: None,
            pending_demand: FrameDemand::NONE,
            state: FrameState::Idle,
        }
    }

    /// Updates the display timing model.
    pub fn set_display_timing(&mut self, display: DisplayTiming) {
        self.display = display;
    }

    /// Updates the current work estimate.
    pub fn set_estimate(&mut self, estimate: FrameTimingEstimate) {
        self.estimate = estimate;
    }

    /// Returns the current work estimate.
    #[must_use]
    pub const fn estimate(&self) -> FrameTimingEstimate {
        self.estimate
    }

    /// Requests a frame.
    ///
    /// Multiple requests coalesce, with input taking priority over animation and
    /// animation taking priority over background work.
    pub fn request_frame(&mut self, demand: FrameDemand, _now: Time) {
        self.pending_demand.insert(demand);
    }

    /// Returns the next scheduler action for `now`.
    #[must_use]
    pub fn next_action(&mut self, now: Time) -> Action {
        match self.state {
            FrameState::Idle => {
                if self.pending_demand.is_empty() {
                    return Action::Idle;
                }
                let plan = self.plan_next_frame(now);
                self.pending_demand = FrameDemand::NONE;
                self.state = FrameState::Planned(plan);
                self.action_for_planned(plan, now)
            }
            FrameState::Planned(plan) => self.action_for_planned(plan, now),
            FrameState::PreSurfaceWorkStarted(_) => Action::Idle,
            FrameState::PreSurfaceWorkDone(plan) => {
                if now < plan.acquire_surface_at {
                    Action::SleepUntil(plan.acquire_surface_at)
                } else {
                    self.state = FrameState::SurfaceAcquired(plan);
                    Action::AcquireSurface(plan)
                }
            }
            FrameState::SurfaceAcquired(_) => Action::Idle,
            FrameState::SurfaceWorkDone(plan) => Action::Present {
                plan,
                presentation: self.presentation_for(plan),
            },
        }
    }

    fn action_for_planned(&mut self, plan: FramePlan, now: Time) -> Action {
        if now < plan.pre_surface_work_start {
            Action::SleepUntil(plan.pre_surface_work_start)
        } else {
            self.state = FrameState::PreSurfaceWorkStarted(plan);
            Action::StartPreSurfaceWork(plan)
        }
    }

    fn plan_next_frame(&mut self, now: Time) -> FramePlan {
        let demand = self.pending_demand;
        let decision = plan_frame(
            self.display,
            self.estimate,
            demand,
            FrameOpportunity {
                now,
                predicted_present_time: None,
                frame_interval: None,
                last_present_time: self.last_present_time,
                pending_target_present_time: None,
            },
        );
        let id = FrameId(self.next_frame_id);
        self.next_frame_id = self.next_frame_id.saturating_add(1);
        FramePlan {
            id,
            demand,
            target_present_time: decision.target_present_time,
            frame_interval: decision.frame_interval,
            pre_surface_work_start: decision.pre_surface_work_start,
            acquire_surface_at: decision.acquire_surface_at,
            submit_deadline: decision.submit_deadline,
        }
    }

    fn presentation_for(&self, plan: FramePlan) -> Presentation {
        match plan.demand.policy() {
            DemandPolicy::Input => Presentation::AsSoonAsReady,
            DemandPolicy::ContinuousInput if plan.frame_interval == self.display.min_interval => {
                Presentation::AsSoonAsReady
            }
            DemandPolicy::ContinuousInput
            | DemandPolicy::Animation
            | DemandPolicy::Background
            | DemandPolicy::None => Presentation::AfterMinimumDuration(plan.frame_interval),
        }
    }

    /// Reports that a scheduled phase completed.
    ///
    /// Reports for stale frame ids are ignored. Duration estimates react
    /// immediately to slower observations and decay slowly after faster ones.
    /// Underestimating frame work causes missed deadlines; overestimating by a
    /// small amount only starts work slightly earlier.
    pub fn report_phase(&mut self, report: FramePhaseReport) {
        if Some(report.frame_id) != self.active_frame_id() {
            return;
        }
        let observed = report.end - report.start;
        match report.phase {
            Phase::Frame => {
                self.estimate.pre_surface_work =
                    smooth_duration(self.estimate.pre_surface_work, observed);
                if let FrameState::PreSurfaceWorkStarted(plan) = self.state {
                    self.state = FrameState::PreSurfaceWorkDone(plan);
                }
            }
            Phase::Surface => {
                self.estimate.surface_work = smooth_duration(self.estimate.surface_work, observed);
                if let FrameState::SurfaceAcquired(plan) = self.state {
                    self.state = FrameState::SurfaceWorkDone(plan);
                }
            }
            Phase::Gpu => {
                self.estimate.gpu_work = smooth_duration(self.estimate.gpu_work, observed);
            }
        }
    }

    /// Reports that a frame was presented.
    pub fn report_presented(&mut self, frame_id: FrameId, presented_at: Time) {
        if Some(frame_id) != self.active_frame_id() {
            return;
        }
        self.last_present_time = Some(presented_at);
        self.state = FrameState::Idle;
    }

    /// Drops the active frame and returns to idle.
    pub fn drop_active_frame(&mut self) {
        self.state = FrameState::Idle;
    }

    /// Returns the active frame id, if any.
    #[must_use]
    pub const fn active_frame_id(&self) -> Option<FrameId> {
        match self.state {
            FrameState::Idle => None,
            FrameState::Planned(plan)
            | FrameState::PreSurfaceWorkStarted(plan)
            | FrameState::PreSurfaceWorkDone(plan)
            | FrameState::SurfaceAcquired(plan)
            | FrameState::SurfaceWorkDone(plan) => Some(plan.id),
        }
    }
}

fn smooth_duration(previous: Duration, observed: Duration) -> Duration {
    if observed >= previous {
        return observed;
    }
    Duration((previous.0.saturating_mul(7).saturating_add(observed.0)) / 8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec::Vec;

    const TEST_75HZ_INTERVAL_NS: u64 = 13_333_333;

    fn action_plan(pacer: &mut FramePacer, action: Action) -> FramePlan {
        match action {
            Action::StartPreSurfaceWork(plan)
            | Action::AcquireSurface(plan)
            | Action::Present { plan, .. } => plan,
            Action::SleepUntil(time) => {
                let action = pacer.next_action(time);
                action_plan(pacer, action)
            }
            Action::Idle => panic!("expected frame action"),
        }
    }

    fn begin_timing(now: i64) -> BeginFrameTiming {
        BeginFrameTiming {
            now: Time::from_nanos(now),
            frame_time: Time::from_nanos(now),
            deadline: Time::from_nanos(now + TEST_75HZ_INTERVAL_NS as i64),
            interval: Duration::from_nanos(TEST_75HZ_INTERVAL_NS),
        }
    }

    fn estimate_ms(pre: u64, surface: u64, gpu: u64, safety: u64) -> FrameTimingEstimate {
        FrameTimingEstimate {
            pre_surface_work: Duration::from_millis(pre),
            surface_work: Duration::from_millis(surface),
            gpu_work: Duration::from_millis(gpu),
            safety_margin: Duration::from_millis(safety),
        }
    }

    fn delivered_frames(
        cadence: TargetFrameCadence,
        display: DisplayTiming,
        tick_interval: Duration,
        ticks: u64,
    ) -> Vec<u64> {
        (0..ticks)
            .filter(|frame_index| cadence.should_deliver(*frame_index, display, tick_interval))
            .collect()
    }

    #[test]
    fn target_frame_cadence_rounds_down_60fps_on_fixed_75hz() {
        let cadence = TargetFrameCadence::from_fps(60.0).unwrap();
        let display = DisplayTiming::fixed(Duration::from_hz(75));
        let delivered = delivered_frames(cadence, display, Duration::from_hz(75), 150);

        assert_eq!(delivered.len(), 75);
        assert_eq!(
            cadence.effective_interval(display),
            Duration::from_hz(75).saturating_mul(2)
        );
        assert!(delivered.windows(2).all(|pair| pair[1] - pair[0] == 2));
    }

    #[test]
    fn target_frame_cadence_rounds_down_60fps_on_variable_48_to_75hz_with_unknown_granularity() {
        let cadence = TargetFrameCadence::from_fps(60.0).unwrap();
        let display = DisplayTiming::variable(Duration::from_hz(75), Duration::from_hz(48), None);
        let delivered = delivered_frames(cadence, display, Duration::from_hz(75), 150);

        assert_eq!(delivered.len(), 75);
        assert_eq!(
            cadence.effective_interval(display),
            Duration::from_hz(75).saturating_mul(2)
        );
    }

    #[test]
    fn target_frame_cadence_rounds_down_when_target_is_outside_variable_range() {
        let display = DisplayTiming::variable(Duration::from_hz(75), Duration::from_hz(48), None);

        let too_fast = TargetFrameCadence::from_fps(90.0).unwrap();
        assert_eq!(too_fast.effective_interval(display), Duration::from_hz(75));

        let too_slow = TargetFrameCadence::from_fps(30.0).unwrap();
        assert_eq!(
            too_slow.effective_interval(display),
            Duration::from_hz(75).saturating_mul(3)
        );
    }

    #[test]
    fn target_frame_cadence_delivers_even_60fps_on_120hz() {
        let cadence = TargetFrameCadence::from_fps(60.0).unwrap();
        let display = DisplayTiming::fixed(Duration::from_hz(120));
        let delivered = delivered_frames(cadence, display, Duration::from_hz(120), 120);

        assert_eq!(delivered.len(), 60);
        assert!(delivered.windows(2).all(|pair| pair[1] - pair[0] == 2));
    }

    #[test]
    fn target_frame_cadence_rounds_down_30fps_on_fixed_75hz() {
        let cadence = TargetFrameCadence::from_fps(30.0).unwrap();
        let display = DisplayTiming::fixed(Duration::from_hz(75));
        let delivered = delivered_frames(cadence, display, Duration::from_hz(75), 75);

        assert_eq!(delivered.len(), 25);
        assert_eq!(
            cadence.effective_interval(display),
            Duration::from_hz(75).saturating_mul(3)
        );
    }

    #[test]
    fn target_frame_cadence_does_not_throttle_faster_than_display_target() {
        let cadence = TargetFrameCadence::from_fps(120.0).unwrap();
        let display = DisplayTiming::fixed(Duration::from_hz(75));
        let delivered = delivered_frames(cadence, display, Duration::from_hz(75), 75);

        assert_eq!(delivered.len(), 75);
        assert_eq!(cadence.effective_interval(display), Duration::from_hz(75));
    }

    #[test]
    fn target_frame_cadence_rejects_invalid_fps() {
        assert!(TargetFrameCadence::from_fps(0.0).is_none());
        assert!(TargetFrameCadence::from_fps(-60.0).is_none());
        assert!(TargetFrameCadence::from_fps(f64::NAN).is_none());
        assert!(TargetFrameCadence::from_fps(f64::INFINITY).is_none());
    }

    #[test]
    fn frame_rate_at_most_rounds_down_on_fixed_75hz() {
        let preference = FrameRatePreference::at_most(60.0).unwrap();
        let display = DisplayTiming::fixed(Duration::from_hz(75));

        assert_eq!(
            preference.effective_interval(display),
            Some(Duration::from_hz(75).saturating_mul(2))
        );
    }

    #[test]
    fn frame_rate_at_most_below_vrr_minimum_rounds_down_from_full_source() {
        let preference = FrameRatePreference::at_most(30.0).unwrap();
        let display = DisplayTiming::variable(Duration::from_hz(75), Duration::from_hz(48), None);

        assert_eq!(
            preference.effective_interval(display),
            Some(Duration::from_hz(75).saturating_mul(3))
        );
        assert_eq!(
            preference.plan(display).map(FrameRatePlan::source_interval),
            Some(Duration::from_hz(75))
        );
    }

    #[test]
    fn frame_rate_group_chooses_source_that_best_serves_members() {
        let display = DisplayTiming::variable(Duration::from_hz(75), Duration::from_hz(48), None);
        let cube = FrameRatePreference::at_most(30.0).unwrap();
        let animation = FrameRatePreference::at_most(60.0).unwrap();

        assert_eq!(
            choose_frame_rate_source_interval(&[cube, animation], display),
            Duration::from_hz(75)
        );
    }

    #[test]
    fn frame_rate_at_most_60_on_vrr_48_to_75_with_unknown_granularity_rounds_down_from_full_source()
    {
        let preference = FrameRatePreference::at_most(60.0).unwrap();
        let display = DisplayTiming::variable(Duration::from_hz(75), Duration::from_hz(48), None);
        let plan = preference.plan(display).unwrap();

        assert_eq!(plan.source_interval(), Duration::from_hz(75));
        assert_eq!(
            plan.delivery_interval(),
            Duration::from_hz(75).saturating_mul(2)
        );
        assert!(plan.should_deliver(0, Duration::from_hz(75)));
    }

    #[test]
    fn frame_rate_at_most_30_on_vrr_48_to_75_with_unknown_granularity_rounds_down_from_full_source()
    {
        let preference = FrameRatePreference::at_most(30.0).unwrap();
        let display = DisplayTiming::variable(Duration::from_hz(75), Duration::from_hz(48), None);
        let plan = preference.plan(display).unwrap();

        assert_eq!(plan.source_interval(), Duration::from_hz(75));
        assert_eq!(
            plan.delivery_interval(),
            Duration::from_hz(75).saturating_mul(3)
        );
    }

    #[test]
    fn frame_rate_group_at_most_60_and_30_on_vrr_48_to_75_with_unknown_granularity_uses_full_source()
     {
        let display = DisplayTiming::variable(Duration::from_hz(75), Duration::from_hz(48), None);
        let at_most_60 = FrameRatePreference::at_most(60.0).unwrap();
        let at_most_30 = FrameRatePreference::at_most(30.0).unwrap();

        assert_eq!(
            choose_frame_rate_source_interval(&[at_most_60, at_most_30], display),
            Duration::from_hz(75)
        );
    }

    #[test]
    fn frame_rate_at_most_60_and_30_on_vrr_48_to_75_throttle_from_full_75hz_source() {
        let display = DisplayTiming::variable(Duration::from_hz(75), Duration::from_hz(48), None);
        let full_source = Duration::from_hz(75);
        let at_most_60 = FrameRatePreference::at_most(60.0).unwrap();
        let at_most_30 = FrameRatePreference::at_most(30.0).unwrap();

        let delivered_60 = (0..150)
            .filter(|frame_index| at_most_60.should_deliver(*frame_index, display, full_source))
            .count();
        let delivered_30 = (0..150)
            .filter(|frame_index| at_most_30.should_deliver(*frame_index, display, full_source))
            .count();

        assert_eq!(delivered_60, 75);
        assert_eq!(delivered_30, 50);
    }

    #[test]
    fn frame_rate_at_most_24_on_vrr_with_unknown_granularity_uses_full_source_and_18_75hz_delivery()
    {
        let preference = FrameRatePreference::at_most(24.0).unwrap();
        let display = DisplayTiming::variable(Duration::from_hz(75), Duration::from_hz(48), None);

        assert_eq!(
            preference.plan(display).map(FrameRatePlan::source_interval),
            Some(Duration::from_hz(75))
        );
        assert_eq!(
            preference.effective_interval(display),
            Some(Duration::from_hz(75).saturating_mul(4))
        );
    }

    #[test]
    fn variable_rate_with_granularity_uses_caller_source_interval() {
        let preference = FrameRatePreference::at_most(60.0).unwrap();
        let display = DisplayTiming::variable(
            Duration::from_hz(120),
            Duration::from_hz(48),
            Some(Duration::from_nanos(4_166_667)),
        );
        let plan = preference.plan(display).unwrap();
        let selected_source = plan.source_interval();

        assert_eq!(selected_source, Duration::from_nanos(16_666_668));
        assert_eq!(plan.delivery_interval(), Duration::from_nanos(16_666_668));
        assert!(plan.should_deliver(0, selected_source));
        assert!(plan.should_deliver(1, selected_source));
        assert!(preference.should_deliver(0, display, selected_source));
        assert!(preference.should_deliver(1, display, selected_source));
    }

    #[test]
    fn variable_rate_with_granularity_30hz_divides_selected_60hz_source_once() {
        let preference = FrameRatePreference::at_most(30.0).unwrap();
        let display = DisplayTiming::variable(
            Duration::from_hz(120),
            Duration::from_hz(48),
            Some(Duration::from_nanos(4_166_667)),
        );
        let plan = preference.plan(display).unwrap();
        let selected_source = plan.source_interval();

        assert_eq!(selected_source, Duration::from_nanos(16_666_668));
        assert_eq!(plan.delivery_interval(), Duration::from_nanos(33_333_336));
        assert!(plan.should_deliver(0, selected_source));
        assert!(!plan.should_deliver(1, selected_source));
        assert!(plan.should_deliver(2, selected_source));
        assert!(preference.should_deliver(0, display, selected_source));
        assert!(!preference.should_deliver(1, display, selected_source));
        assert!(preference.should_deliver(2, display, selected_source));
    }

    #[test]
    fn frame_rate_group_full_member_requests_full_source_rate() {
        let display = DisplayTiming::variable(Duration::from_hz(75), Duration::from_hz(48), None);
        let cube = FrameRatePreference::at_most(30.0).unwrap();

        assert_eq!(
            choose_frame_rate_source_interval(&[cube, FrameRatePreference::full()], display),
            Duration::from_hz(75)
        );
    }

    #[test]
    fn frame_rate_minimum_can_choose_higher_fixed_cadence() {
        let preference = FrameRatePreference::preferred(60.0)
            .unwrap()
            .minimum(50.0)
            .unwrap()
            .build();
        let display = DisplayTiming::fixed(Duration::from_hz(75));

        assert_eq!(
            preference.effective_interval(display),
            Some(Duration::from_hz(75))
        );
    }

    #[test]
    fn frame_rate_range_with_unknown_granularity_chooses_clean_cadence_within_range() {
        let preference = FrameRatePreference::range(50.0, 75.0)
            .unwrap()
            .preferred(60.0)
            .unwrap()
            .build();
        let display = DisplayTiming::variable(Duration::from_hz(75), Duration::from_hz(48), None);

        assert_eq!(
            preference.effective_interval(display),
            Some(Duration::from_hz(75))
        );
    }

    #[test]
    fn frame_rate_range_accounts_for_display_granularity() {
        let preference = FrameRatePreference::range(50.0, 120.0)
            .unwrap()
            .preferred(60.0)
            .unwrap()
            .build();
        let display = DisplayTiming::variable(
            Duration::from_hz(120),
            Duration::from_hz(24),
            Some(Duration::from_nanos(4_166_667)),
        );

        assert_eq!(
            preference.effective_interval(display),
            Some(Duration::from_nanos(16_666_668))
        );
    }

    #[test]
    fn frame_rate_preference_rejects_invalid_combinations() {
        assert!(FrameRatePreference::at_most(0.0).is_none());
        assert!(FrameRatePreference::preferred(f64::NAN).is_none());
        assert!(FrameRatePreference::range(75.0, 50.0).is_none());
        assert!(
            FrameRatePreference::preferred(60.0)
                .unwrap()
                .range(75.0, 50.0)
                .is_none()
        );
    }

    #[test]
    fn begin_frame_scheduler_coalesces_while_active() {
        let mut scheduler = BeginFrameScheduler::new();
        let first = match scheduler.begin_frame(FrameDemand::Animation, begin_timing(0)) {
            BeginFrameResult::Start(frame) => frame,
            other => panic!("expected start, got {other:?}"),
        };

        let second = match scheduler.begin_frame(FrameDemand::ContinuousInput, begin_timing(1)) {
            BeginFrameResult::Coalesced { active, pending } => {
                assert_eq!(active, first);
                pending
            }
            other => panic!("expected coalesced frame, got {other:?}"),
        };

        assert_eq!(scheduler.active_frame(), Some(first));
        assert_eq!(scheduler.finish_frame(), Some(second));
        assert!(!scheduler.has_active_frame());
        assert!(!scheduler.has_pending_frame());
    }

    #[test]
    fn begin_frame_scheduler_enters_deadline_once_active() {
        let mut scheduler = BeginFrameScheduler::new();
        let frame = match scheduler.begin_frame(FrameDemand::Animation, begin_timing(0)) {
            BeginFrameResult::Start(frame) => frame,
            other => panic!("expected start, got {other:?}"),
        };

        assert_eq!(
            scheduler.begin_deadline(),
            BeginFrameDeadlineResult::Commit(frame)
        );
        assert_eq!(
            scheduler.begin_deadline(),
            BeginFrameDeadlineResult::Commit(frame)
        );
        assert_eq!(scheduler.finish_frame(), None);
        assert_eq!(scheduler.begin_deadline(), BeginFrameDeadlineResult::Idle);
    }

    fn drawable_work() -> CompositorWorkStatus {
        CompositorWorkStatus {
            can_draw: true,
            needs_frame_work: true,
            scene_jobs_pending: false,
            commit_ready: false,
        }
    }

    fn commit_ready_work() -> CompositorWorkStatus {
        CompositorWorkStatus {
            can_draw: true,
            needs_frame_work: false,
            scene_jobs_pending: false,
            commit_ready: true,
        }
    }

    fn pending_scene_work() -> CompositorWorkStatus {
        CompositorWorkStatus {
            can_draw: true,
            needs_frame_work: false,
            scene_jobs_pending: true,
            commit_ready: false,
        }
    }

    #[test]
    fn compositor_scheduler_starts_frame_for_damage() {
        let mut scheduler = CompositorFrameScheduler::new();
        scheduler.request_frame(FrameDemand::ContinuousInput);

        let frame = match scheduler.on_begin_frame(begin_timing(0), drawable_work()) {
            CompositorFrameAction::StartFrame(frame) => frame,
            other => panic!("expected start frame, got {other:?}"),
        };

        assert_eq!(frame.sequence, 1);
        assert_eq!(
            frame.demand,
            FrameDemand::ContinuousInput | FrameDemand::Animation
        );
        assert!(scheduler.has_active_frame());
        assert!(!scheduler.has_pending_frame());
    }

    #[test]
    fn compositor_scheduler_drops_opportunity_without_work() {
        let mut scheduler = CompositorFrameScheduler::new();

        let action = scheduler.on_begin_frame(
            begin_timing(0),
            CompositorWorkStatus {
                can_draw: true,
                needs_frame_work: false,
                scene_jobs_pending: false,
                commit_ready: false,
            },
        );

        assert_eq!(action, CompositorFrameAction::Idle);
        assert!(!scheduler.has_active_frame());
    }

    #[test]
    fn chromium_scheduler_begin_frame_ack_for_dropped_begin_frame_keeps_source_idle() {
        // Chromium sends a no-damage ack when a BeginFrame arrives while the
        // scheduler has no work, and it must not confirm/hold the frame as
        // active. The equivalent invariant here is that an idle opportunity
        // leaves no active frame and does not keep the frame source alive.
        let mut scheduler = CompositorFrameScheduler::new();
        let idle = CompositorWorkStatus {
            can_draw: true,
            needs_frame_work: false,
            scene_jobs_pending: false,
            commit_ready: false,
        };

        assert_eq!(
            scheduler.on_begin_frame(begin_timing(0), idle),
            CompositorFrameAction::Idle
        );
        assert!(!scheduler.has_active_frame());
        assert!(!scheduler.has_pending_frame());
        assert!(!scheduler.needs_frame_source(idle));
    }

    #[test]
    fn compositor_scheduler_does_not_start_when_cannot_draw() {
        let mut scheduler = CompositorFrameScheduler::new();
        scheduler.request_frame(FrameDemand::Animation);

        let action = scheduler.on_begin_frame(
            begin_timing(0),
            CompositorWorkStatus {
                can_draw: false,
                needs_frame_work: true,
                scene_jobs_pending: false,
                commit_ready: false,
            },
        );

        assert_eq!(action, CompositorFrameAction::Idle);
        assert!(!scheduler.has_active_frame());
        assert!(scheduler.needs_frame_source(CompositorWorkStatus {
            can_draw: true,
            needs_frame_work: false,
            scene_jobs_pending: false,
            commit_ready: false,
        }));
    }

    #[test]
    fn compositor_scheduler_coalesces_latest_begin_frame_while_active() {
        let mut scheduler = CompositorFrameScheduler::new();
        scheduler.request_frame(FrameDemand::Animation);
        let first = match scheduler.on_begin_frame(begin_timing(0), drawable_work()) {
            CompositorFrameAction::StartFrame(frame) => frame,
            other => panic!("expected start, got {other:?}"),
        };

        scheduler.request_frame(FrameDemand::Input);
        let second = match scheduler.on_begin_frame(begin_timing(1), drawable_work()) {
            CompositorFrameAction::Coalesced { active, pending } => {
                assert_eq!(active, first);
                pending
            }
            other => panic!("expected coalesced, got {other:?}"),
        };

        scheduler.request_frame(FrameDemand::ContinuousInput);
        let third = match scheduler.on_begin_frame(begin_timing(2), drawable_work()) {
            CompositorFrameAction::Coalesced { active, pending } => {
                assert_eq!(active, first);
                pending
            }
            other => panic!("expected coalesced, got {other:?}"),
        };

        assert_ne!(second, third);
        assert_eq!(third.sequence, 3);
        assert_eq!(
            scheduler.on_commit_complete(),
            CompositorFrameAction::FinishFrame {
                pending: Some(third)
            }
        );
    }

    #[test]
    fn compositor_scheduler_arms_deadline_when_scene_jobs_are_pending() {
        let mut scheduler = CompositorFrameScheduler::new();
        scheduler.request_frame(FrameDemand::Animation);
        let frame = match scheduler.on_begin_frame(begin_timing(0), drawable_work()) {
            CompositorFrameAction::StartFrame(frame) => frame,
            other => panic!("expected start, got {other:?}"),
        };

        let deadline = Time::from_nanos(10);
        let action = scheduler.on_frame_submitted(deadline, pending_scene_work());

        assert_eq!(
            action,
            CompositorFrameAction::ArmDeadline(CompositorDeadline {
                frame_sequence: frame.sequence,
                deadline,
            })
        );
        assert_eq!(
            scheduler.active_deadline(),
            Some(CompositorDeadline {
                frame_sequence: frame.sequence,
                deadline,
            })
        );
    }

    #[test]
    fn compositor_scheduler_commits_immediately_when_submission_is_ready() {
        let mut scheduler = CompositorFrameScheduler::new();
        scheduler.request_frame(FrameDemand::Animation);
        let frame = match scheduler.on_begin_frame(begin_timing(0), drawable_work()) {
            CompositorFrameAction::StartFrame(frame) => frame,
            other => panic!("expected start, got {other:?}"),
        };

        assert_eq!(
            scheduler.on_frame_submitted(Time::from_nanos(10), commit_ready_work()),
            CompositorFrameAction::Commit {
                frame,
                reason: CompositorCommitReason::SceneReady,
            }
        );
    }

    #[test]
    fn compositor_scheduler_scene_ready_before_deadline_commits() {
        let mut scheduler = CompositorFrameScheduler::new();
        scheduler.request_frame(FrameDemand::Animation);
        let frame = match scheduler.on_begin_frame(begin_timing(0), drawable_work()) {
            CompositorFrameAction::StartFrame(frame) => frame,
            other => panic!("expected start, got {other:?}"),
        };
        assert!(matches!(
            scheduler.on_frame_submitted(Time::from_nanos(10), pending_scene_work()),
            CompositorFrameAction::ArmDeadline(_)
        ));

        assert_eq!(
            scheduler.on_scene_ready(Time::from_nanos(9), commit_ready_work()),
            CompositorFrameAction::Commit {
                frame,
                reason: CompositorCommitReason::SceneReady,
            }
        );
    }

    #[test]
    fn compositor_scheduler_commit_attempt_committed_finishes_frame() {
        let mut scheduler = CompositorFrameScheduler::new();
        scheduler.request_frame(FrameDemand::Animation);
        let frame = match scheduler.on_begin_frame(begin_timing(0), drawable_work()) {
            CompositorFrameAction::StartFrame(frame) => frame,
            other => panic!("expected start, got {other:?}"),
        };
        assert_eq!(
            scheduler.on_frame_submitted(Time::from_nanos(10), commit_ready_work()),
            CompositorFrameAction::Commit {
                frame,
                reason: CompositorCommitReason::SceneReady,
            }
        );

        assert_eq!(
            scheduler.on_commit_attempt(CompositorCommitResult::Committed),
            CompositorFrameAction::FinishFrame { pending: None }
        );
        assert!(!scheduler.has_active_frame());
        assert!(!scheduler.needs_frame_source(CompositorWorkStatus {
            can_draw: true,
            needs_frame_work: false,
            scene_jobs_pending: false,
            commit_ready: false,
        }));
    }

    #[test]
    fn compositor_scheduler_commit_rejection_carries_ready_work() {
        let mut scheduler = CompositorFrameScheduler::new();
        scheduler.request_frame(FrameDemand::Animation);
        let frame = match scheduler.on_begin_frame(begin_timing(0), drawable_work()) {
            CompositorFrameAction::StartFrame(frame) => frame,
            other => panic!("expected start, got {other:?}"),
        };
        assert!(matches!(
            scheduler.on_frame_submitted(Time::from_nanos(10), pending_scene_work()),
            CompositorFrameAction::ArmDeadline(_)
        ));
        assert_eq!(
            scheduler.on_scene_ready(Time::from_nanos(9), commit_ready_work()),
            CompositorFrameAction::Commit {
                frame,
                reason: CompositorCommitReason::SceneReady,
            }
        );
        assert_eq!(
            scheduler.on_commit_attempt(CompositorCommitResult::Rejected),
            CompositorFrameAction::Idle
        );
        assert!(scheduler.needs_frame_source(CompositorWorkStatus {
            can_draw: true,
            needs_frame_work: false,
            scene_jobs_pending: false,
            commit_ready: false,
        }));

        assert_eq!(
            scheduler.on_begin_frame(
                begin_timing(1),
                CompositorWorkStatus {
                    can_draw: true,
                    needs_frame_work: false,
                    scene_jobs_pending: false,
                    commit_ready: false,
                },
            ),
            CompositorFrameAction::Commit {
                frame,
                reason: CompositorCommitReason::ReadyCarry,
            }
        );
    }

    #[test]
    fn compositor_scheduler_no_work_commit_attempt_finishes_frame() {
        let mut scheduler = CompositorFrameScheduler::new();
        scheduler.request_frame(FrameDemand::Animation);
        assert!(matches!(
            scheduler.on_begin_frame(begin_timing(0), drawable_work()),
            CompositorFrameAction::StartFrame(_)
        ));
        assert!(matches!(
            scheduler.on_frame_submitted(Time::from_nanos(10), pending_scene_work()),
            CompositorFrameAction::ArmDeadline(_)
        ));
        assert!(matches!(
            scheduler.on_deadline(),
            CompositorFrameAction::Commit {
                reason: CompositorCommitReason::Deadline,
                ..
            }
        ));

        assert_eq!(
            scheduler.on_commit_attempt(CompositorCommitResult::NoWork),
            CompositorFrameAction::FinishFrame { pending: None }
        );
        assert!(!scheduler.has_active_frame());
    }

    #[test]
    fn compositor_scheduler_scene_ready_while_more_jobs_pending_does_not_commit() {
        let mut scheduler = CompositorFrameScheduler::new();
        scheduler.request_frame(FrameDemand::Animation);
        assert!(matches!(
            scheduler.on_begin_frame(begin_timing(0), drawable_work()),
            CompositorFrameAction::StartFrame(_)
        ));
        assert!(matches!(
            scheduler.on_frame_submitted(Time::from_nanos(10), pending_scene_work()),
            CompositorFrameAction::ArmDeadline(_)
        ));

        assert_eq!(
            scheduler.on_scene_ready(
                Time::from_nanos(9),
                CompositorWorkStatus {
                    can_draw: true,
                    needs_frame_work: false,
                    scene_jobs_pending: true,
                    commit_ready: true,
                },
            ),
            CompositorFrameAction::Idle
        );
    }

    #[test]
    fn compositor_scheduler_scene_ready_after_deadline_waits_for_deadline() {
        let mut scheduler = CompositorFrameScheduler::new();
        scheduler.request_frame(FrameDemand::Animation);
        let frame = match scheduler.on_begin_frame(begin_timing(0), drawable_work()) {
            CompositorFrameAction::StartFrame(frame) => frame,
            other => panic!("expected start, got {other:?}"),
        };
        assert!(matches!(
            scheduler.on_frame_submitted(Time::from_nanos(10), pending_scene_work()),
            CompositorFrameAction::ArmDeadline(_)
        ));

        assert_eq!(
            scheduler.on_scene_ready(Time::from_nanos(11), commit_ready_work()),
            CompositorFrameAction::Idle
        );
        assert_eq!(
            scheduler.on_deadline(),
            CompositorFrameAction::Commit {
                frame,
                reason: CompositorCommitReason::Deadline,
            }
        );
    }

    #[test]
    fn compositor_scheduler_carries_ready_commit_before_starting_new_frame() {
        let mut scheduler = CompositorFrameScheduler::new();
        scheduler.request_frame(FrameDemand::Animation);
        let frame = match scheduler.on_begin_frame(begin_timing(0), drawable_work()) {
            CompositorFrameAction::StartFrame(frame) => frame,
            other => panic!("expected start, got {other:?}"),
        };
        assert!(matches!(
            scheduler.on_frame_submitted(Time::from_nanos(10), pending_scene_work()),
            CompositorFrameAction::ArmDeadline(_)
        ));

        assert_eq!(
            scheduler.on_begin_frame(begin_timing(1), commit_ready_work()),
            CompositorFrameAction::Commit {
                frame,
                reason: CompositorCommitReason::ReadyCarry,
            }
        );
    }

    #[test]
    fn chromium_style_begin_frame_keeps_authoritative_interval_when_tick_arrives_late() {
        // Chromium's scheduler reports the interval carried by BeginFrameArgs,
        // not the wall-clock gap between delivery callbacks. A late tick must
        // not make fixed-rate pacing infer a slower display.
        let interval_120hz = Duration::from_nanos(8_333_333);
        let late_arrival = Duration::from_millis(4);
        let decision = plan_frame(
            DisplayTiming::fixed(interval_120hz),
            estimate_ms(1, 1, 1, 1),
            FrameDemand::Animation,
            FrameOpportunity {
                now: Time::ZERO + interval_120hz + late_arrival,
                predicted_present_time: Some(Time::ZERO + interval_120hz.saturating_mul(2)),
                frame_interval: Some(interval_120hz),
                last_present_time: Some(Time::ZERO),
                pending_target_present_time: None,
            },
        );

        assert_eq!(decision.frame_interval, interval_120hz);
        assert_eq!(
            decision.target_present_time,
            Time::ZERO + interval_120hz.saturating_mul(2)
        );
    }

    #[test]
    fn chromium_style_begin_frame_interval_change_is_used_on_next_opportunity() {
        let interval_120hz = Duration::from_nanos(8_333_333);
        let interval_90hz = Duration::from_nanos(11_111_111);
        let estimate = estimate_ms(1, 1, 1, 1);

        let first = plan_frame(
            DisplayTiming::fixed(interval_120hz),
            estimate,
            FrameDemand::Animation,
            FrameOpportunity {
                now: Time::ZERO,
                predicted_present_time: Some(Time::ZERO + interval_120hz),
                frame_interval: Some(interval_120hz),
                last_present_time: None,
                pending_target_present_time: None,
            },
        );
        let second = plan_frame(
            DisplayTiming::fixed(interval_90hz),
            estimate,
            FrameDemand::Animation,
            FrameOpportunity {
                now: first.target_present_time,
                predicted_present_time: Some(first.target_present_time + interval_90hz),
                frame_interval: Some(interval_90hz),
                last_present_time: Some(first.target_present_time),
                pending_target_present_time: None,
            },
        );

        assert_eq!(first.frame_interval, interval_120hz);
        assert_eq!(second.frame_interval, interval_90hz);
    }

    #[test]
    fn chromium_style_zero_begin_frame_interval_falls_back_to_display_timing() {
        // Chromium ignores a zero BeginFrame interval for client interval
        // updates. Understory should likewise use the display timing supplied by
        // the host instead of accepting zero as a real cadence.
        let interval_90hz = Duration::from_nanos(11_111_111);
        let decision = plan_frame(
            DisplayTiming::fixed(interval_90hz),
            estimate_ms(1, 1, 1, 1),
            FrameDemand::Animation,
            FrameOpportunity {
                now: Time::ZERO,
                predicted_present_time: Some(Time::ZERO + interval_90hz),
                frame_interval: Some(Duration::ZERO),
                last_present_time: None,
                pending_target_present_time: None,
            },
        );

        assert_eq!(decision.frame_interval, interval_90hz);
        assert_eq!(decision.target_present_time, Time::ZERO + interval_90hz);
    }

    #[test]
    fn chromium_style_deadline_regular_late_and_immediate_commit_behavior() {
        let mut scheduler = CompositorFrameScheduler::new();
        scheduler.request_frame(FrameDemand::Animation);
        let frame = match scheduler.on_begin_frame(begin_timing(0), drawable_work()) {
            CompositorFrameAction::StartFrame(frame) => frame,
            other => panic!("expected start, got {other:?}"),
        };

        let regular_deadline = Time::from_nanos(10);
        assert_eq!(
            scheduler.on_frame_submitted(regular_deadline, pending_scene_work()),
            CompositorFrameAction::ArmDeadline(CompositorDeadline {
                frame_sequence: frame.sequence,
                deadline: regular_deadline,
            })
        );
        assert_eq!(
            scheduler.on_scene_ready(Time::from_nanos(9), commit_ready_work()),
            CompositorFrameAction::Commit {
                frame,
                reason: CompositorCommitReason::SceneReady,
            }
        );

        assert_eq!(
            scheduler.on_commit_complete(),
            CompositorFrameAction::FinishFrame { pending: None }
        );
        scheduler.request_frame(FrameDemand::Animation);
        let frame = match scheduler.on_begin_frame(begin_timing(20), drawable_work()) {
            CompositorFrameAction::StartFrame(frame) => frame,
            other => panic!("expected start, got {other:?}"),
        };
        let late_deadline = frame.timing.frame_time + frame.timing.interval;
        assert_eq!(
            scheduler.on_frame_submitted(late_deadline, pending_scene_work()),
            CompositorFrameAction::ArmDeadline(CompositorDeadline {
                frame_sequence: frame.sequence,
                deadline: late_deadline,
            })
        );
        assert_eq!(
            scheduler.on_deadline(),
            CompositorFrameAction::Commit {
                frame,
                reason: CompositorCommitReason::Deadline,
            }
        );

        assert_eq!(
            scheduler.on_commit_complete(),
            CompositorFrameAction::FinishFrame { pending: None }
        );
        scheduler.request_frame(FrameDemand::Animation);
        let frame = match scheduler.on_begin_frame(begin_timing(40), drawable_work()) {
            CompositorFrameAction::StartFrame(frame) => frame,
            other => panic!("expected start, got {other:?}"),
        };
        assert_eq!(
            scheduler.on_frame_submitted(Time::from_nanos(40), commit_ready_work()),
            CompositorFrameAction::Commit {
                frame,
                reason: CompositorCommitReason::SceneReady,
            }
        );
    }

    #[test]
    fn chromium_style_pending_scene_jobs_commit_old_state_at_deadline() {
        let mut scheduler = CompositorFrameScheduler::new();
        scheduler.request_frame(FrameDemand::Animation);
        let frame = match scheduler.on_begin_frame(begin_timing(0), drawable_work()) {
            CompositorFrameAction::StartFrame(frame) => frame,
            other => panic!("expected start, got {other:?}"),
        };
        assert!(matches!(
            scheduler.on_frame_submitted(Time::from_nanos(10), pending_scene_work()),
            CompositorFrameAction::ArmDeadline(_)
        ));

        // The new scene is still pending. Chromium's display scheduler still
        // hits the deadline and draws/commits whatever is currently ready
        // rather than starting overlapping work for the next BeginFrame.
        assert_eq!(
            scheduler.on_deadline(),
            CompositorFrameAction::Commit {
                frame,
                reason: CompositorCommitReason::Deadline,
            }
        );
        assert_eq!(
            scheduler.on_commit_complete(),
            CompositorFrameAction::FinishFrame { pending: None }
        );
    }

    #[test]
    fn deadline_scene_pending_keeps_active_frame_until_scene_ready() {
        let mut scheduler = CompositorFrameScheduler::new();
        scheduler.request_frame(FrameDemand::Animation);
        let frame = match scheduler.on_begin_frame(begin_timing(0), drawable_work()) {
            CompositorFrameAction::StartFrame(frame) => frame,
            other => panic!("expected start, got {other:?}"),
        };
        assert!(matches!(
            scheduler.on_frame_submitted(Time::from_nanos(10), pending_scene_work()),
            CompositorFrameAction::ArmDeadline(_)
        ));

        assert_eq!(
            scheduler.on_deadline(),
            CompositorFrameAction::Commit {
                frame,
                reason: CompositorCommitReason::Deadline,
            }
        );
        assert_eq!(
            scheduler.on_commit_attempt(CompositorCommitResult::ScenePending),
            CompositorFrameAction::Idle
        );
        assert!(scheduler.has_active_frame());

        assert_eq!(
            scheduler.on_scene_ready(Time::from_nanos(11), commit_ready_work()),
            CompositorFrameAction::Commit {
                frame,
                reason: CompositorCommitReason::Deadline,
            }
        );
        assert_eq!(
            scheduler.on_commit_attempt(CompositorCommitResult::Committed),
            CompositorFrameAction::FinishFrame { pending: None }
        );
        assert!(!scheduler.has_active_frame());
    }

    #[test]
    fn deadline_scene_pending_preserves_latest_coalesced_frame() {
        let mut scheduler = CompositorFrameScheduler::new();
        scheduler.request_frame(FrameDemand::Animation);
        let active = match scheduler.on_begin_frame(begin_timing(0), drawable_work()) {
            CompositorFrameAction::StartFrame(frame) => frame,
            other => panic!("expected start, got {other:?}"),
        };
        assert!(matches!(
            scheduler.on_frame_submitted(Time::from_nanos(10), pending_scene_work()),
            CompositorFrameAction::ArmDeadline(_)
        ));
        assert!(matches!(
            scheduler.on_deadline(),
            CompositorFrameAction::Commit {
                reason: CompositorCommitReason::Deadline,
                ..
            }
        ));
        assert_eq!(
            scheduler.on_commit_attempt(CompositorCommitResult::ScenePending),
            CompositorFrameAction::Idle
        );

        scheduler.request_frame(FrameDemand::ContinuousInput);
        let pending = match scheduler.on_begin_frame(begin_timing(20), drawable_work()) {
            CompositorFrameAction::Coalesced {
                active: seen,
                pending,
            } => {
                assert_eq!(seen, active);
                pending
            }
            other => panic!("expected coalesced, got {other:?}"),
        };

        assert_eq!(
            scheduler.on_scene_ready(Time::from_nanos(21), commit_ready_work()),
            CompositorFrameAction::Commit {
                frame: active,
                reason: CompositorCommitReason::Deadline,
            }
        );
        assert_eq!(
            scheduler.on_commit_attempt(CompositorCommitResult::Committed),
            CompositorFrameAction::FinishFrame {
                pending: Some(pending)
            }
        );
    }

    #[test]
    fn chromium_style_resize_like_pending_surface_uses_late_deadline() {
        let mut scheduler = CompositorFrameScheduler::new();
        scheduler.request_frame(FrameDemand::ContinuousInput);
        let frame = match scheduler.on_begin_frame(begin_timing(100), drawable_work()) {
            CompositorFrameAction::StartFrame(frame) => frame,
            other => panic!("expected start, got {other:?}"),
        };

        let late_deadline = frame.timing.frame_time + frame.timing.interval;
        assert_eq!(
            scheduler.on_frame_submitted(late_deadline, pending_scene_work()),
            CompositorFrameAction::ArmDeadline(CompositorDeadline {
                frame_sequence: frame.sequence,
                deadline: late_deadline,
            })
        );
        assert_eq!(scheduler.active_deadline().unwrap().deadline, late_deadline);
    }

    #[test]
    fn chromium_style_frame_source_stays_live_for_pending_scene_without_new_damage() {
        let mut scheduler = CompositorFrameScheduler::new();
        scheduler.request_frame(FrameDemand::Animation);
        assert!(matches!(
            scheduler.on_begin_frame(begin_timing(0), drawable_work()),
            CompositorFrameAction::StartFrame(_)
        ));
        assert!(matches!(
            scheduler.on_frame_submitted(Time::from_nanos(10), pending_scene_work()),
            CompositorFrameAction::ArmDeadline(_)
        ));

        assert!(scheduler.needs_frame_source(CompositorWorkStatus {
            can_draw: true,
            needs_frame_work: false,
            scene_jobs_pending: true,
            commit_ready: false,
        }));
    }

    #[test]
    fn chromium_style_coalesced_tick_uses_latest_timing_after_active_frame_finishes() {
        let mut scheduler = CompositorFrameScheduler::new();
        scheduler.request_frame(FrameDemand::Animation);
        let active = match scheduler.on_begin_frame(begin_timing(0), drawable_work()) {
            CompositorFrameAction::StartFrame(frame) => frame,
            other => panic!("expected start, got {other:?}"),
        };

        scheduler.request_frame(FrameDemand::ContinuousInput);
        let stale_pending = match scheduler.on_begin_frame(begin_timing(10), drawable_work()) {
            CompositorFrameAction::Coalesced { active: a, pending } => {
                assert_eq!(a, active);
                pending
            }
            other => panic!("expected coalesced, got {other:?}"),
        };
        scheduler.request_frame(FrameDemand::ContinuousInput);
        let latest_pending = match scheduler.on_begin_frame(begin_timing(20), drawable_work()) {
            CompositorFrameAction::Coalesced { active: a, pending } => {
                assert_eq!(a, active);
                pending
            }
            other => panic!("expected coalesced, got {other:?}"),
        };

        assert_ne!(stale_pending.timing, latest_pending.timing);
        assert_eq!(
            scheduler.on_commit_complete(),
            CompositorFrameAction::FinishFrame {
                pending: Some(latest_pending)
            }
        );
    }

    #[test]
    fn compositor_scheduler_commit_complete_finishes_and_requeues_pending_demand() {
        let mut scheduler = CompositorFrameScheduler::new();
        scheduler.request_frame(FrameDemand::Animation);
        let first = match scheduler.on_begin_frame(begin_timing(0), drawable_work()) {
            CompositorFrameAction::StartFrame(frame) => frame,
            other => panic!("expected start, got {other:?}"),
        };
        scheduler.request_frame(FrameDemand::ContinuousInput);
        let pending = match scheduler.on_begin_frame(begin_timing(1), drawable_work()) {
            CompositorFrameAction::Coalesced { active, pending } => {
                assert_eq!(active, first);
                pending
            }
            other => panic!("expected coalesced, got {other:?}"),
        };

        assert_eq!(
            scheduler.on_commit_complete(),
            CompositorFrameAction::FinishFrame {
                pending: Some(pending)
            }
        );

        let next = match scheduler.on_begin_frame(begin_timing(2), drawable_work()) {
            CompositorFrameAction::StartFrame(frame) => frame,
            other => panic!("expected pending demand to start, got {other:?}"),
        };
        assert!(next.demand.contains(FrameDemand::ContinuousInput));
    }

    #[test]
    fn chromium_scheduler_main_frame_not_skipped_after_late_commit() {
        // Chromium has explicit tests that a late commit must not cause the
        // next requested main frame to be skipped. Model that as an active
        // frame with a pending coalesced frame: once the late commit completes,
        // the latest pending demand is requeued and starts on the next
        // opportunity.
        let mut scheduler = CompositorFrameScheduler::new();
        scheduler.request_frame(FrameDemand::Animation);
        let active = match scheduler.on_begin_frame(begin_timing(0), drawable_work()) {
            CompositorFrameAction::StartFrame(frame) => frame,
            other => panic!("expected start, got {other:?}"),
        };

        scheduler.request_frame(FrameDemand::ContinuousInput);
        let pending = match scheduler.on_begin_frame(begin_timing(10), drawable_work()) {
            CompositorFrameAction::Coalesced {
                active: seen,
                pending,
            } => {
                assert_eq!(seen, active);
                pending
            }
            other => panic!("expected coalesced, got {other:?}"),
        };

        assert_eq!(
            scheduler.on_commit_attempt(CompositorCommitResult::Committed),
            CompositorFrameAction::FinishFrame {
                pending: Some(pending)
            }
        );
        let next = match scheduler.on_begin_frame(begin_timing(20), drawable_work()) {
            CompositorFrameAction::StartFrame(frame) => frame,
            other => panic!("expected pending demand to start, got {other:?}"),
        };
        assert!(next.demand.contains(FrameDemand::ContinuousInput));
        assert_ne!(next.sequence, active.sequence);
    }

    #[test]
    fn chromium_display_scheduler_root_resources_locked_waits_for_deadline() {
        // Chromium DisplayScheduler keeps the late deadline while root surface
        // resources are locked and does not draw early. In our compositor
        // scheduler, pending scene jobs represent locked/unavailable resources:
        // scene-ready is ignored until all jobs are available, and the deadline
        // still commits old ready state.
        let mut scheduler = CompositorFrameScheduler::new();
        scheduler.request_frame(FrameDemand::Animation);
        let frame = match scheduler.on_begin_frame(begin_timing(0), drawable_work()) {
            CompositorFrameAction::StartFrame(frame) => frame,
            other => panic!("expected start, got {other:?}"),
        };
        assert!(matches!(
            scheduler.on_frame_submitted(Time::from_nanos(10), pending_scene_work()),
            CompositorFrameAction::ArmDeadline(_)
        ));

        assert_eq!(
            scheduler.on_scene_ready(
                Time::from_nanos(9),
                CompositorWorkStatus {
                    can_draw: true,
                    needs_frame_work: false,
                    scene_jobs_pending: true,
                    commit_ready: true,
                },
            ),
            CompositorFrameAction::Idle
        );
        assert_eq!(
            scheduler.on_deadline(),
            CompositorFrameAction::Commit {
                frame,
                reason: CompositorCommitReason::Deadline,
            }
        );
    }

    #[test]
    fn compositor_scheduler_keeps_frame_source_active_until_finish() {
        let mut scheduler = CompositorFrameScheduler::new();
        let idle_status = CompositorWorkStatus {
            can_draw: true,
            needs_frame_work: false,
            scene_jobs_pending: false,
            commit_ready: false,
        };
        assert!(!scheduler.needs_frame_source(idle_status));

        scheduler.request_frame(FrameDemand::Animation);
        assert!(scheduler.needs_frame_source(idle_status));

        assert!(matches!(
            scheduler.on_begin_frame(begin_timing(0), drawable_work()),
            CompositorFrameAction::StartFrame(_)
        ));
        assert!(scheduler.needs_frame_source(idle_status));

        assert_eq!(
            scheduler.on_commit_complete(),
            CompositorFrameAction::FinishFrame { pending: None }
        );
        assert!(!scheduler.needs_frame_source(idle_status));
    }

    #[test]
    fn compositor_scheduler_does_not_request_frame_source_when_cannot_draw() {
        let mut scheduler = CompositorFrameScheduler::new();
        scheduler.request_frame(FrameDemand::Animation);

        assert!(!scheduler.needs_frame_source(CompositorWorkStatus {
            can_draw: false,
            needs_frame_work: true,
            scene_jobs_pending: true,
            commit_ready: true,
        }));
    }

    #[test]
    fn compositor_scheduler_reset_clears_active_pending_deadline_and_demand() {
        let mut scheduler = CompositorFrameScheduler::new();
        scheduler.request_frame(FrameDemand::Animation);
        assert!(matches!(
            scheduler.on_begin_frame(begin_timing(0), drawable_work()),
            CompositorFrameAction::StartFrame(_)
        ));
        scheduler.request_frame(FrameDemand::ContinuousInput);
        assert!(matches!(
            scheduler.on_begin_frame(begin_timing(1), drawable_work()),
            CompositorFrameAction::Coalesced { .. }
        ));
        assert!(matches!(
            scheduler.on_frame_submitted(Time::from_nanos(10), pending_scene_work()),
            CompositorFrameAction::ArmDeadline(_)
        ));

        scheduler.reset();

        assert!(!scheduler.has_active_frame());
        assert!(!scheduler.has_pending_frame());
        assert_eq!(scheduler.active_deadline(), None);
        assert_eq!(
            scheduler.on_begin_frame(
                begin_timing(2),
                CompositorWorkStatus {
                    can_draw: true,
                    needs_frame_work: false,
                    scene_jobs_pending: false,
                    commit_ready: false,
                },
            ),
            CompositorFrameAction::Idle
        );
    }

    #[test]
    fn fixed_rate_slow_work_starts_earliest_feasible_slot() {
        let display = DisplayTiming::fixed(Duration::from_hz(60));
        let mut pacer = FramePacer::new(display);
        pacer.set_estimate(FrameTimingEstimate {
            pre_surface_work: Duration::from_millis(8),
            surface_work: Duration::from_millis(6),
            gpu_work: Duration::from_millis(9),
            safety_margin: Duration::from_millis(1),
        });
        pacer.request_frame(FrameDemand::Animation, Time::ZERO);
        let action = pacer.next_action(Time::ZERO);
        let plan = action_plan(&mut pacer, action);
        assert_eq!(plan.frame_interval, Duration::from_hz(60));
        assert_eq!(plan.target_present_time, Time::from_nanos(33_333_332));
        assert_eq!(plan.pre_surface_work_start, Time::from_nanos(9_333_332));
    }

    #[test]
    fn variable_rate_without_granularity_uses_fastest_source_divisors() {
        let display = DisplayTiming::variable(Duration::from_hz(120), Duration::from_hz(40), None);
        let mut pacer = FramePacer::new(display);
        pacer.set_estimate(FrameTimingEstimate {
            pre_surface_work: Duration::from_millis(3),
            surface_work: Duration::from_millis(2),
            gpu_work: Duration::from_millis(6),
            safety_margin: Duration::from_millis(1),
        });
        pacer.request_frame(FrameDemand::Animation, Time::ZERO);
        let action = pacer.next_action(Time::ZERO);
        let plan = action_plan(&mut pacer, action);
        assert_eq!(
            plan.frame_interval,
            Duration::from_hz(120).saturating_mul(2)
        );
    }

    #[test]
    fn phase_estimate_jumps_up_to_slower_observation() {
        let mut pacer = FramePacer::new(DisplayTiming::fixed(Duration::from_hz(60)));
        pacer.set_estimate(FrameTimingEstimate {
            pre_surface_work: Duration::from_millis(2),
            surface_work: Duration::from_millis(2),
            gpu_work: Duration::from_millis(2),
            safety_margin: Duration::ZERO,
        });
        pacer.request_frame(FrameDemand::Animation, Time::ZERO);
        let action = pacer.next_action(Time::ZERO);
        let plan = action_plan(&mut pacer, action);

        pacer.report_phase(FramePhaseReport::pre_surface_work(
            plan.id,
            Time::ZERO,
            Time::ZERO + Duration::from_millis(9),
        ));

        assert_eq!(pacer.estimate().pre_surface_work, Duration::from_millis(9));
    }

    #[test]
    fn phase_estimate_decays_slowly_after_faster_observation() {
        let mut pacer = FramePacer::new(DisplayTiming::fixed(Duration::from_hz(60)));
        pacer.set_estimate(FrameTimingEstimate {
            pre_surface_work: Duration::from_millis(8),
            surface_work: Duration::from_millis(2),
            gpu_work: Duration::from_millis(2),
            safety_margin: Duration::ZERO,
        });
        pacer.request_frame(FrameDemand::Animation, Time::ZERO);
        let action = pacer.next_action(Time::ZERO);
        let plan = action_plan(&mut pacer, action);

        pacer.report_phase(FramePhaseReport::pre_surface_work(
            plan.id,
            Time::ZERO,
            Time::ZERO + Duration::ZERO,
        ));

        assert_eq!(pacer.estimate().pre_surface_work, Duration::from_millis(7));
    }

    #[test]
    fn input_presents_as_soon_as_ready() {
        let mut pacer = FramePacer::new(DisplayTiming::fixed(Duration::from_hz(60)));
        pacer.request_frame(FrameDemand::Input, Time::ZERO);
        let action = pacer.next_action(Time::ZERO);
        let plan = action_plan(&mut pacer, action);
        pacer.report_phase(FramePhaseReport::pre_surface_work(
            plan.id,
            Time::ZERO,
            Time::ZERO + Duration::from_millis(2),
        ));
        let acquire_at = match pacer.next_action(Time::ZERO) {
            Action::SleepUntil(t) => t,
            other => panic!("expected sleep, got {other:?}"),
        };
        let _ = pacer.next_action(acquire_at);
        pacer.report_phase(FramePhaseReport::surface_work(
            plan.id,
            acquire_at,
            acquire_at + Duration::from_millis(2),
        ));
        let present = pacer.next_action(acquire_at + Duration::from_millis(2));
        assert!(matches!(
            present,
            Action::Present {
                presentation: Presentation::AsSoonAsReady,
                ..
            }
        ));
    }

    #[test]
    fn surface_acquisition_is_delayed_until_surface_work_window() {
        let mut pacer = FramePacer::new(DisplayTiming::fixed(Duration::from_hz(60)));
        pacer.set_estimate(FrameTimingEstimate {
            pre_surface_work: Duration::from_millis(2),
            surface_work: Duration::from_millis(1),
            gpu_work: Duration::from_millis(1),
            safety_margin: Duration::from_millis(1),
        });
        pacer.request_frame(FrameDemand::Animation, Time::ZERO);
        let action = pacer.next_action(Time::ZERO);
        let plan = action_plan(&mut pacer, action);
        pacer.report_phase(FramePhaseReport::pre_surface_work(
            plan.id,
            Time::ZERO,
            Time::ZERO + Duration::from_millis(2),
        ));
        assert_eq!(
            pacer.next_action(Time::ZERO + Duration::from_millis(2)),
            Action::SleepUntil(plan.acquire_surface_at)
        );
    }

    #[test]
    fn animation_on_fixed_display_uses_minimum_duration_present() {
        let mut pacer = FramePacer::new(DisplayTiming::fixed(Duration::from_hz(60)));
        pacer.request_frame(FrameDemand::Animation, Time::ZERO);
        let action = pacer.next_action(Time::ZERO);
        let plan = action_plan(&mut pacer, action);
        pacer.report_phase(FramePhaseReport::pre_surface_work(
            plan.id,
            plan.pre_surface_work_start,
            plan.acquire_surface_at,
        ));
        let _ = pacer.next_action(plan.acquire_surface_at);
        pacer.report_phase(FramePhaseReport::surface_work(
            plan.id,
            plan.acquire_surface_at,
            plan.submit_deadline,
        ));
        let action = pacer.next_action(plan.submit_deadline);
        assert!(matches!(
            action,
            Action::Present {
                presentation: Presentation::AfterMinimumDuration(_),
                ..
            }
        ));
    }

    #[test]
    fn stateless_plan_slow_fixed_animation_keeps_current_opportunity() {
        let decision = plan_frame(
            DisplayTiming::fixed(Duration::from_hz(75)),
            FrameTimingEstimate {
                pre_surface_work: Duration::from_millis(9),
                surface_work: Duration::from_millis(2),
                gpu_work: Duration::from_millis(2),
                safety_margin: Duration::from_millis(1),
            },
            FrameDemand::Animation,
            FrameOpportunity {
                now: Time::ZERO,
                predicted_present_time: Some(Time::from_nanos(13_333_333)),
                frame_interval: Some(Duration::from_nanos(13_333_333)),
                last_present_time: None,
                pending_target_present_time: None,
            },
        );

        assert_eq!(decision.frame_interval, Duration::from_nanos(13_333_333));
        assert_eq!(decision.target_present_time, Time::from_nanos(26_666_666));
        assert_eq!(
            decision.pre_surface_work_start,
            Time::from_nanos(12_666_666)
        );
        assert!(matches!(
            decision.presentation,
            Presentation::AfterMinimumDuration(Duration(13_333_333))
        ));
    }

    #[test]
    fn stateless_plan_keeps_slow_fixed_input_asap() {
        let decision = plan_frame(
            DisplayTiming::fixed(Duration::from_hz(75)),
            FrameTimingEstimate {
                pre_surface_work: Duration::from_millis(9),
                surface_work: Duration::from_millis(2),
                gpu_work: Duration::from_millis(2),
                safety_margin: Duration::from_millis(1),
            },
            FrameDemand::Input,
            FrameOpportunity {
                now: Time::ZERO,
                predicted_present_time: Some(Time::from_nanos(13_333_333)),
                frame_interval: Some(Duration::from_nanos(13_333_333)),
                last_present_time: None,
                pending_target_present_time: None,
            },
        );

        assert_eq!(decision.frame_interval, Duration::from_nanos(13_333_333));
        assert_eq!(decision.target_present_time, Time::from_nanos(14_000_000));
        assert!(matches!(decision.presentation, Presentation::AsSoonAsReady));
    }

    #[test]
    fn stateless_plan_slow_fixed_continuous_input_keeps_current_opportunity() {
        let decision = plan_frame(
            DisplayTiming::fixed(Duration::from_hz(75)),
            FrameTimingEstimate {
                pre_surface_work: Duration::from_millis(9),
                surface_work: Duration::from_millis(2),
                gpu_work: Duration::from_millis(2),
                safety_margin: Duration::from_millis(1),
            },
            FrameDemand::ContinuousInput,
            FrameOpportunity {
                now: Time::ZERO,
                predicted_present_time: Some(Time::from_nanos(13_333_333)),
                frame_interval: Some(Duration::from_nanos(13_333_333)),
                last_present_time: None,
                pending_target_present_time: None,
            },
        );

        assert_eq!(decision.frame_interval, Duration::from_nanos(13_333_333));
        assert_eq!(decision.target_present_time, Time::from_nanos(26_666_666));
        assert_eq!(
            decision.pre_surface_work_start,
            Time::from_nanos(12_666_666)
        );
        assert!(matches!(decision.presentation, Presentation::AsSoonAsReady));
    }

    #[test]
    fn stateless_plan_continuous_input_does_not_reuse_pending_animation_target() {
        let decision = plan_frame(
            DisplayTiming::fixed(Duration::from_hz(75)),
            FrameTimingEstimate {
                pre_surface_work: Duration::from_millis(9),
                surface_work: Duration::from_millis(2),
                gpu_work: Duration::from_millis(2),
                safety_margin: Duration::from_millis(1),
            },
            FrameDemand::ContinuousInput,
            FrameOpportunity {
                now: Time::from_nanos(20_000_000),
                predicted_present_time: Some(Time::from_nanos(26_666_666)),
                frame_interval: Some(Duration::from_nanos(13_333_333)),
                last_present_time: None,
                pending_target_present_time: Some(Time::from_nanos(40_000_000)),
            },
        );

        assert_eq!(decision.target_present_time, Time::from_nanos(39_999_999));
    }

    #[test]
    fn stateless_plan_keeps_light_fixed_input_asap() {
        let decision = plan_frame(
            DisplayTiming::fixed(Duration::from_hz(75)),
            FrameTimingEstimate {
                pre_surface_work: Duration::from_millis(4),
                surface_work: Duration::from_millis(1),
                gpu_work: Duration::from_millis(1),
                safety_margin: Duration::from_millis(1),
            },
            FrameDemand::Input,
            FrameOpportunity {
                now: Time::ZERO,
                predicted_present_time: Some(Time::from_nanos(13_333_333)),
                frame_interval: Some(Duration::from_nanos(13_333_333)),
                last_present_time: None,
                pending_target_present_time: None,
            },
        );

        assert_eq!(decision.frame_interval, Duration::from_nanos(13_333_333));
        assert_eq!(decision.target_present_time, Time::from_nanos(13_333_333));
        assert!(matches!(decision.presentation, Presentation::AsSoonAsReady));
    }

    #[test]
    fn stateless_plan_late_starts_light_fixed_animation_from_predicted_present() {
        let decision = plan_frame(
            DisplayTiming::fixed(Duration::from_hz(75)),
            FrameTimingEstimate {
                pre_surface_work: Duration::from_micros(500),
                surface_work: Duration::ZERO,
                gpu_work: Duration::ZERO,
                safety_margin: Duration::from_millis(1),
            },
            FrameDemand::Animation,
            FrameOpportunity {
                now: Time::ZERO,
                predicted_present_time: Some(Time::from_nanos(13_333_333)),
                frame_interval: Some(Duration::from_nanos(13_333_333)),
                last_present_time: None,
                pending_target_present_time: None,
            },
        );

        assert_eq!(decision.target_present_time, Time::from_nanos(13_333_333));
        assert_eq!(
            decision.pre_surface_work_start,
            Time::from_nanos(11_833_333)
        );
        assert!(matches!(
            decision.presentation,
            Presentation::AfterMinimumDuration(Duration(13_333_333))
        ));
    }

    #[test]
    fn stateless_plan_late_starts_light_fixed_continuous_input_from_predicted_present() {
        let decision = plan_frame(
            DisplayTiming::fixed(Duration::from_hz(75)),
            FrameTimingEstimate {
                pre_surface_work: Duration::from_micros(500),
                surface_work: Duration::ZERO,
                gpu_work: Duration::ZERO,
                safety_margin: Duration::from_millis(1),
            },
            FrameDemand::ContinuousInput,
            FrameOpportunity {
                now: Time::ZERO,
                predicted_present_time: Some(Time::from_nanos(13_333_333)),
                frame_interval: Some(Duration::from_nanos(13_333_333)),
                last_present_time: None,
                pending_target_present_time: None,
            },
        );

        assert_eq!(decision.target_present_time, Time::from_nanos(13_333_333));
        assert_eq!(
            decision.pre_surface_work_start,
            Time::from_nanos(11_833_333)
        );
        assert!(matches!(decision.presentation, Presentation::AsSoonAsReady));
    }

    #[test]
    fn stateless_plan_rolls_smooth_work_past_too_close_prediction() {
        let decision = plan_frame(
            DisplayTiming::fixed(Duration::from_hz(75)),
            FrameTimingEstimate {
                pre_surface_work: Duration::from_micros(500),
                surface_work: Duration::ZERO,
                gpu_work: Duration::ZERO,
                safety_margin: Duration::from_millis(1),
            },
            FrameDemand::ContinuousInput,
            FrameOpportunity {
                now: Time::from_nanos(12_500_000),
                predicted_present_time: Some(Time::from_nanos(13_333_333)),
                frame_interval: Some(Duration::from_nanos(13_333_333)),
                last_present_time: None,
                pending_target_present_time: None,
            },
        );

        assert_eq!(decision.target_present_time, Time::from_nanos(26_666_666));
        assert_eq!(
            decision.pre_surface_work_start,
            Time::from_nanos(25_166_666)
        );
        assert!(matches!(decision.presentation, Presentation::AsSoonAsReady));
    }

    #[test]
    fn stateless_plan_larger_fixed_continuous_input_estimate_starts_earlier_not_later() {
        let now = Time::ZERO;
        let predicted = Time::from_nanos(13_333_333);
        let interval = Duration::from_nanos(13_333_333);
        let light = plan_frame(
            DisplayTiming::fixed(interval),
            FrameTimingEstimate {
                pre_surface_work: Duration::from_millis(2),
                surface_work: Duration::ZERO,
                gpu_work: Duration::ZERO,
                safety_margin: Duration::from_millis(1),
            },
            FrameDemand::ContinuousInput,
            FrameOpportunity {
                now,
                predicted_present_time: Some(predicted),
                frame_interval: Some(interval),
                last_present_time: None,
                pending_target_present_time: None,
            },
        );
        let heavier = plan_frame(
            DisplayTiming::fixed(interval),
            FrameTimingEstimate {
                pre_surface_work: Duration::from_millis(6),
                surface_work: Duration::ZERO,
                gpu_work: Duration::ZERO,
                safety_margin: Duration::from_millis(1),
            },
            FrameDemand::ContinuousInput,
            FrameOpportunity {
                now,
                predicted_present_time: Some(predicted),
                frame_interval: Some(interval),
                last_present_time: None,
                pending_target_present_time: None,
            },
        );

        assert_eq!(light.target_present_time, predicted);
        assert_eq!(heavier.target_present_time, predicted);
        assert!(heavier.pre_surface_work_start < light.pre_surface_work_start);
    }

    #[test]
    fn stateless_plan_keeps_variable_input_asap_even_when_slow() {
        let decision = plan_frame(
            DisplayTiming::variable(Duration::from_hz(75), Duration::from_hz(24), None),
            FrameTimingEstimate {
                pre_surface_work: Duration::from_millis(80),
                surface_work: Duration::from_millis(1),
                gpu_work: Duration::ZERO,
                safety_margin: Duration::from_millis(1),
            },
            FrameDemand::Input,
            FrameOpportunity {
                now: Time::ZERO,
                predicted_present_time: Some(Time::from_nanos(13_333_333)),
                frame_interval: Some(Duration::from_nanos(13_333_333)),
                last_present_time: None,
                pending_target_present_time: None,
            },
        );

        assert_eq!(decision.frame_interval, Duration::from_nanos(13_333_333));
        assert_eq!(decision.target_present_time, Time::from_nanos(82_000_000));
        assert!(matches!(decision.presentation, Presentation::AsSoonAsReady));
    }

    #[test]
    fn stateless_plan_clamps_stale_prediction_to_now() {
        let decision = plan_frame(
            DisplayTiming::fixed(Duration::from_hz(60)),
            FrameTimingEstimate {
                pre_surface_work: Duration::from_millis(1),
                surface_work: Duration::from_millis(1),
                gpu_work: Duration::from_millis(1),
                safety_margin: Duration::from_millis(1),
            },
            FrameDemand::Input,
            FrameOpportunity {
                now: Time::from_nanos(20_000_000),
                predicted_present_time: Some(Time::from_nanos(10_000_000)),
                frame_interval: Some(Duration::from_nanos(16_666_667)),
                last_present_time: None,
                pending_target_present_time: None,
            },
        );

        assert_eq!(decision.target_present_time, Time::from_nanos(24_000_000));
    }

    #[test]
    fn stateless_plan_replans_infeasible_pending_target() {
        let decision = plan_frame(
            DisplayTiming::fixed(Duration::from_hz(75)),
            FrameTimingEstimate {
                pre_surface_work: Duration::from_millis(9),
                surface_work: Duration::from_millis(2),
                gpu_work: Duration::from_millis(2),
                safety_margin: Duration::from_millis(1),
            },
            FrameDemand::Animation,
            FrameOpportunity {
                now: Time::from_nanos(20_000_000),
                predicted_present_time: Some(Time::from_nanos(26_666_666)),
                frame_interval: Some(Duration::from_nanos(13_333_333)),
                last_present_time: None,
                pending_target_present_time: Some(Time::from_nanos(25_000_000)),
            },
        );

        assert!(decision.target_present_time >= Time::from_nanos(34_000_000));
    }

    #[test]
    fn stateless_plan_replans_pending_target_that_violates_cadence() {
        let decision = plan_frame(
            DisplayTiming::fixed(Duration::from_hz(75)),
            FrameTimingEstimate {
                pre_surface_work: Duration::from_millis(9),
                surface_work: Duration::from_millis(2),
                gpu_work: Duration::from_millis(2),
                safety_margin: Duration::from_millis(1),
            },
            FrameDemand::Animation,
            FrameOpportunity {
                now: Time::from_nanos(20_000_000),
                predicted_present_time: Some(Time::from_nanos(26_666_666)),
                frame_interval: Some(Duration::from_nanos(13_333_333)),
                last_present_time: Some(Time::from_nanos(13_333_333)),
                pending_target_present_time: Some(Time::from_nanos(30_000_000)),
            },
        );

        assert!(decision.target_present_time >= Time::from_nanos(39_999_999));
    }
}
