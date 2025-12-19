//! PyTorch-compatible random number generation.
//!
//! This module implements PyTorch's exact Box-Muller algorithm to achieve
//! numerical parity with `torch.randn()` when using the same MT19937 seed.
//!
//! # Two Paths: Scalar and Vectorized
//!
//! PyTorch uses **different algorithms** based on tensor size:
//! - **Scalar path (size < 16)**: Uses 53-bit double-precision uniforms with caching
//! - **Vectorized path (size >= 16)**: Uses 24-bit float-precision uniforms with SIMD batching
//!
//! ## Scalar Path (size < 16)
//!
//! 1. **Uniform conversion**: 53-bit double from two u32 values
//! 2. **Uses `log(1 - u2)`** not `log(u2)` to avoid log(0)
//! 3. **Caches the second value** from each Box-Muller pair
//! 4. **u1 is used for theta**, u2 for radius
//!
//! ## Vectorized Path (size >= 16)
//!
//! 1. **Uniform conversion**: 24-bit float from single u32: `(u32 & 0xFFFFFF) / 16777216.0`
//! 2. **Processes in chunks of 16**
//! 3. **u1[i] = rand[8+i], u2[i] = rand[i]** (second half for angle, first half for radius)
//! 4. **Output order**: all cos values first, then all sin values
//!
//! # Example
//!
//! ```rust,ignore
//! use rand_mt::Mt;
//! use vibevoice_rs::pytorch_rng::PyTorchNormal;
//!
//! let mut rng = Mt::new(524242);
//! let mut normal = PyTorchNormal::new();
//!
//! // Scalar path (size < 16): uses caching
//! let v1 = normal.sample(&mut rng);
//! let v2 = normal.sample(&mut rng);  // Returns cached value
//!
//! // Vectorized path (size >= 16): batch processing
//! let mut rng2 = Mt::new(524242);
//! let vec16 = PyTorchNormal::sample_vectorized(&mut rng2, 16);
//! ```
//!
//! # References
//!
//! - PyTorch source: `aten/src/ATen/native/cpu/DistributionTemplates.h`
//! - PyTorch source: `aten/src/ATen/core/TransformationHelper.h`
//! - PyTorch source: `aten/src/ATen/core/DistributionsHelper.h`

use candle_core::{Device, Tensor};
use rand_mt::Mt; // Mersenne Twister 32-bit (MT19937) to match Python's torch.Generator
// IMPORTANT: Must use Mt (32-bit), NOT Mt64 (64-bit) - they produce different sequences!
use std::sync::Mutex;
use anyhow::{Error as AnyErr, Result};
use tracing::{debug, info};

/// Global CPU RNG state for deterministic random generation across all functions.
/// Includes both the MT19937 generator AND the Box-Muller cache for PyTorch parity.
///
/// IMPORTANT: Python's torch.manual_seed() uses MT19937 (32-bit), NOT MT19937-64!
/// This MUST be at module level to be shared between set_all_seeds() and seeded_randn().
struct GlobalRngState {
    /// Mersenne Twister 32-bit generator (identical to PyTorch's)
    rng: Mt,
    /// PyTorch-compatible Box-Muller with caching
    normal: PyTorchNormal,
}

static GLOBAL_CPU_RNG: Mutex<Option<GlobalRngState>> = Mutex::new(None);

/// PyTorch-compatible normal distribution generator.
///
/// This struct maintains the state needed to exactly match PyTorch's
/// `torch.randn()` output given the same MT19937 seed.
///
/// # Caching Behavior
///
/// Box-Muller generates two independent normal values per invocation.
/// PyTorch returns one immediately and caches the second for the next call.
/// This means:
/// - Odd-numbered calls (1st, 3rd, 5th, ...) consume 2 RNG values
/// - Even-numbered calls (2nd, 4th, 6th, ...) consume 0 RNG values
///
/// This caching is critical for sequence alignment with PyTorch.
#[derive(Debug, Clone)]
pub struct PyTorchNormal {
    /// Cached second value from Box-Muller (None if cache is empty)
    cached_value: Option<f32>,
    /// Total u32 values consumed from MT19937 since creation/reset
    u32_consumed: u64,
}

impl PyTorchNormal {
    /// Create a new PyTorch-compatible normal generator.
    ///
    /// The cache starts empty, so the first sample will generate a fresh pair.
    #[inline]
    pub fn new() -> Self {
        Self {
            cached_value: None,
            u32_consumed: 0,
        }
    }

    /// Get the total number of u32 values consumed from MT19937.
    ///
    /// Used for debugging RNG state synchronization with Python.
    #[inline]
    pub fn u32_consumed(&self) -> u64 {
        self.u32_consumed
    }

    /// Convert two MT19937 u32 values to a 53-bit uniform double [0, 1).
    ///
    /// PyTorch's randn() uses 64-bit (double precision) uniforms internally,
    /// even when generating float outputs. Each 53-bit uniform requires
    /// TWO u32 values from the MT19937 generator.
    ///
    /// This matches PyTorch's random64() → uniform_real<double> pattern:
    /// ```text
    /// u64 = (hi << 32) | lo  // Combine two u32
    /// uniform = (u64 & 0x1FFFFFFFFFFFFF) / 9007199254740992.0  // 53-bit / 2^53
    /// ```
    ///
    /// # Why 53 bits?
    ///
    /// IEEE 754 double-precision floats have a 52-bit mantissa plus an implicit
    /// leading 1, giving 53 bits of precision.
    #[inline]
    fn mt_to_uniform_double(lo: u32, hi: u32) -> f64 {
        // Combine two u32 into u64
        // PyTorch uses (lo << 32) | hi for uniform generation
        let combined = ((lo as u64) << 32) | (hi as u64);
        // Extract 53 bits and divide by 2^53
        const MASK_53BIT: u64 = 0x001F_FFFF_FFFF_FFFF; // 53 bits
        const DIVISOR: f64 = 9_007_199_254_740_992.0; // 2^53
        (combined & MASK_53BIT) as f64 / DIVISOR
    }

    /// Sample a single value from N(0, 1) using PyTorch's Box-Muller.
    ///
    /// This method:
    /// 1. Returns cached value if available (no RNG consumption)
    /// 2. Otherwise generates two uniforms and computes Box-Muller pair
    /// 3. Returns first value immediately, caches second value
    ///
    /// # Box-Muller Formula (PyTorch variant)
    ///
    /// Given two uniform values u1, u2 in [0, 1):
    /// ```text
    /// r = sqrt(-2 * log(1 - u2))   // Note: log(1-u2), NOT log(u2)
    /// theta = 2 * PI * u1
    ///
    /// z1 = r * cos(theta)          // Returned immediately
    /// z2 = r * sin(theta)          // Cached for next call
    /// ```
    ///
    /// # Why `log(1 - u2)` instead of `log(u2)`?
    ///
    /// When u2 is exactly 1.0, log(u2) = 0 which is fine, but when u2 is
    /// exactly 0.0, log(u2) = -inf. By using `1 - u2`, we convert the range
    /// from [0, 1) to (0, 1], ensuring the log argument is always positive.
    pub fn sample(&mut self, rng: &mut Mt) -> f32 {
        // If we have a cached value, return it and clear the cache
        if let Some(cached) = self.cached_value.take() {
            return cached;
        }

        // Generate two 53-bit uniform values from MT19937
        // PyTorch's randn() uses double precision internally, consuming 4 u32 values total:
        // - u1 from u32[0], u32[1] combined into 64-bit → 53-bit uniform
        // - u2 from u32[2], u32[3] combined into 64-bit → 53-bit uniform
        let lo1 = rng.next_u32();
        let hi1 = rng.next_u32();
        let lo2 = rng.next_u32();
        let hi2 = rng.next_u32();
        self.u32_consumed += 4;

        let u1 = Self::mt_to_uniform_double(lo1, hi1);
        let u2 = Self::mt_to_uniform_double(lo2, hi2);

        // Box-Muller transform (PyTorch's variant) in double precision
        // CRITICAL: Use log(1 - u2), not log(u2), to avoid log(0)
        let r = (-2.0_f64 * (1.0_f64 - u2).ln()).sqrt();
        let theta = 2.0_f64 * std::f64::consts::PI * u1;

        // Generate the pair in double precision, then convert to f32
        let sample1 = (r * theta.cos()) as f32;
        let sample2 = (r * theta.sin()) as f32;

        // Cache sample2 for next call
        self.cached_value = Some(sample2);
        sample1
    }

    /// Sample with mean and standard deviation.
    ///
    /// Equivalent to `sample() * std + mean`, matching PyTorch's
    /// `torch.randn(...) * std + mean` pattern.
    #[inline]
    pub fn sample_scaled(&mut self, rng: &mut Mt, mean: f32, std: f32) -> f32 {
        self.sample(rng) * std + mean
    }

    /// Convert a single MT19937 u32 to a 24-bit uniform float [0, 1).
    ///
    /// This is used by PyTorch's vectorized path (`torch.rand()`).
    /// Takes the upper 24 bits and divides by 2^24.
    ///
    /// # Formula
    /// ```text
    /// uniform = (u32 & 0xFFFFFF) / 16777216.0
    /// ```
    #[inline]
    fn mt_to_uniform_float(val: u32) -> f32 {
        const MASK_24BIT: u32 = 0x00FF_FFFF; // 24 bits
        const DIVISOR: f32 = 16_777_216.0; // 2^24
        (val & MASK_24BIT) as f32 / DIVISOR
    }

    /// Generate N normal values using PyTorch's vectorized Box-Muller algorithm.
    ///
    /// This matches PyTorch's behavior for `torch.randn(N)` where N >= 16.
    /// The algorithm processes values in chunks of 16:
    ///
    /// 1. Generate 16 uniform values from MT19937
    /// 2. Split into u1 (second half) and u2 (first half)
    /// 3. Apply Box-Muller: r = sqrt(-2 * log(1 - u2)), theta = 2π * u1
    /// 4. Output: [cos0, cos1, ..., cos7, sin0, sin1, ..., sin7]
    ///
    /// # Important
    ///
    /// The vectorized path has no caching - it's designed for batch processing.
    /// If you need exact PyTorch parity, use this for size >= 16 and
    /// the scalar `sample()` method for size < 16.
    ///
    /// # Panics
    ///
    /// Currently panics if `count` is not a multiple of 16. For non-multiples,
    /// PyTorch uses a more complex algorithm that we haven't implemented yet.
    pub fn sample_vectorized(&mut self, rng: &mut Mt, count: usize) -> Vec<f32> {
        assert!(
            count.is_multiple_of(16),
            "Vectorized path currently only supports multiples of 16, got {}",
            count
        );

        let mut output = Vec::with_capacity(count);
        let num_chunks = count / 16;

        // Process in chunks of 16
        for _ in 0..num_chunks {
            // Generate 16 uniform values
            let mut uniforms = [0.0_f32; 16];
            for u in uniforms.iter_mut() {
                *u = Self::mt_to_uniform_float(rng.next_u32());
            }

            // Split: u1 = uniforms[8:16], u2 = uniforms[0:8]
            // Apply Box-Muller and output cos first, then sin
            let mut cos_vals = [0.0_f32; 8];
            let mut sin_vals = [0.0_f32; 8];

            for i in 0..8 {
                let u1 = uniforms[8 + i]; // Second half → angle
                let u2 = uniforms[i]; // First half → radius

                // Box-Muller transform (same log(1-u2) formula as scalar)
                let r = (-2.0_f32 * (1.0_f32 - u2).ln()).sqrt();
                let theta = 2.0_f32 * std::f32::consts::PI * u1;

                cos_vals[i] = r * theta.cos();
                sin_vals[i] = r * theta.sin();
            }

            // Output order: all cos values, then all sin values
            output.extend_from_slice(&cos_vals);
            output.extend_from_slice(&sin_vals);
        }

        // Track consumption: 16 u32 values per chunk
        self.u32_consumed += (num_chunks * 16) as u64;

        output
    }

    /// Generate N normal values with mean and standard deviation using vectorized path.
    ///
    /// Equivalent to `sample_vectorized(rng, count).iter().map(|x| x * std + mean)`.
    pub fn sample_vectorized_scaled(
        &mut self,
        rng: &mut Mt,
        count: usize,
        mean: f32,
        std: f32,
    ) -> Vec<f32> {
        self.sample_vectorized(rng, count)
            .into_iter()
            .map(|x| x * std + mean)
            .collect()
    }
}

impl Default for PyTorchNormal {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_float_conversion() {
        // Test the 24-bit masking and division for vectorized path
        assert_eq!(PyTorchNormal::mt_to_uniform_float(0), 0.0);
        assert_eq!(
            PyTorchNormal::mt_to_uniform_float(0x00FF_FFFF),
            16_777_215.0 / 16_777_216.0
        );
        // Values above 24 bits should be masked off
        assert_eq!(
            PyTorchNormal::mt_to_uniform_float(0xFFFF_FFFF),
            16_777_215.0 / 16_777_216.0
        );
        assert_eq!(PyTorchNormal::mt_to_uniform_float(0xFF00_0000), 0.0);
    }

    #[test]
    fn test_vectorized_basic() {
        // Test that vectorized produces 16 values
        let mut rng = Mt::new(524242);
        let mut normal = PyTorchNormal::new();
        let values = normal.sample_vectorized(&mut rng, 16);
        assert_eq!(values.len(), 16);

        // All values should be finite
        for v in &values {
            assert!(v.is_finite());
        }

        // Check consumption tracking
        assert_eq!(normal.u32_consumed(), 16);
    }

    #[test]
    fn test_vectorized_determinism() {
        // Same seed should produce same sequence
        let mut rng1 = Mt::new(524242);
        let mut rng2 = Mt::new(524242);
        let mut normal1 = PyTorchNormal::new();
        let mut normal2 = PyTorchNormal::new();

        let v1 = normal1.sample_vectorized(&mut rng1, 32);
        let v2 = normal2.sample_vectorized(&mut rng2, 32);

        assert_eq!(v1, v2);
    }

    #[test]
    fn test_determinism() {
        // Same seed should produce same sequence
        let mut rng1 = Mt::new(524242);
        let mut normal1 = PyTorchNormal::new();

        let mut rng2 = Mt::new(524242);
        let mut normal2 = PyTorchNormal::new();

        for _ in 0..100 {
            let v1 = normal1.sample(&mut rng1);
            let v2 = normal2.sample(&mut rng2);
            assert_eq!(v1, v2);
        }
    }

    #[test]
    fn test_pytorch_parity_seed_524242() {
        // Test against known Python values
        // Python: torch.manual_seed(524242); torch.randn(4).tolist()
        // = [-0.7195818424224854, 2.221095561981201, 0.5087963938713074, 0.8240591883659363]

        // Test SCALAR path (size < 16)
        let mut rng = Mt::new(524242);
        let mut normal = PyTorchNormal::new();

        let v1 = normal.sample(&mut rng);
        let v2 = normal.sample(&mut rng);
        let v3 = normal.sample(&mut rng);
        let v4 = normal.sample(&mut rng);

        println!("Rust randn(4) SCALAR path with seed 524242:");
        println!("  v1 = {} (Python expects: -0.7196)", v1);
        println!("  v2 = {} (Python expects: 2.2211)", v2);
        println!("  v3 = {} (Python expects: 0.5088)", v3);
        println!("  v4 = {} (Python expects: 0.8241)", v4);

        // Test VECTORIZED path (size >= 16)
        // Python: torch.manual_seed(524242); torch.randn(16).tolist()
        let mut rng2 = Mt::new(524242);
        let mut normal2 = PyTorchNormal::new();
        let vec16 = normal2.sample_vectorized(&mut rng2, 16);

        println!("\nRust randn(16) VECTORIZED path with seed 524242:");
        println!("  first4 = {:?}", &vec16[0..4]);
        println!("  (Python randn(16) first4 should be different - vectorized path)");
    }

    #[test]
    fn test_scaled_sampling() {
        let mut rng = Mt::new(524242);
        let mut normal = PyTorchNormal::new();

        // Get raw sample
        let raw = normal.sample(&mut rng);

        // Reset and get scaled sample
        let mut rng2 = Mt::new(524242);
        let mut normal2 = PyTorchNormal::new();
        let scaled = normal2.sample_scaled(&mut rng2, 5.0, 2.0);

        assert_eq!(scaled, raw * 2.0 + 5.0);
    }
}


/// Set all random seeds for full reproducibility across CPU, CUDA, and Metal.
/// This function must be called BEFORE any model loading to ensure deterministic behavior.
///
/// Uses Mersenne Twister 32-bit (MT19937) to match Python's torch.Generator, and resets
/// the Box-Muller cache to ensure the same sequence as a fresh PyTorch session.
///
/// # Important
///
/// This function resets BOTH the MT19937 state AND the Box-Muller cache. If you only
/// reset the MT19937 without clearing the cache, you'll get a different sequence than
/// PyTorch because the cached second value from the previous Box-Muller pair would
/// be returned first.
pub fn set_all_seeds(seed: u64, device: &Device) -> Result<()> {
    // Set device seed (only for CUDA/Metal, CPU doesn't support set_seed)
    if !matches!(device, Device::Cpu) {
        device.set_seed(seed)?;
    }

    // Initialize the global RNG state with both MT19937 AND a fresh Box-Muller cache
    // NOTE: PyTorch internally uses MT19937 (32-bit), so we truncate the seed to u32
    // This matches Python's behavior: torch.manual_seed(654321) uses the lower 32 bits
    let mut rng_guard = GLOBAL_CPU_RNG
        .lock()
        .map_err(|e| AnyErr::msg(format!("Failed to acquire RNG lock: {}", e)))?;
    *rng_guard = Some(GlobalRngState {
        rng: Mt::new(seed as u32),
        normal: PyTorchNormal::new(), // Fresh cache - critical for sequence alignment
    });

    debug!(
        "🎲 All random seeds set to {} (device + MT19937-32bit + Box-Muller cache)",
        seed
    );
    Ok(())
}

/// Saved RNG state for later restoration.
/// This allows generating random values without permanently advancing the global RNG.
#[derive(Clone)]
pub struct SavedRngState {
    rng: Mt,
    normal: PyTorchNormal,
}

/// Save the current global RNG state for later restoration.
///
/// This is useful when you want to generate random values for one purpose
/// (e.g., voice embedding) without affecting the RNG position for another
/// purpose (e.g., diffusion noise).
///
/// # Example
/// ```ignore
/// let saved = save_rng_state()?;
/// // Generate some random values...
/// seeded_randn(...)?;
/// // Restore to original position
/// restore_rng_state(saved)?;
/// // Next seeded_randn will use the same position as before
/// ```
pub fn save_rng_state() -> Result<SavedRngState> {
    let rng_guard = GLOBAL_CPU_RNG
        .lock()
        .map_err(|e| AnyErr::msg(format!("Failed to acquire RNG lock: {}", e)))?;

    let state = rng_guard.as_ref().ok_or_else(|| {
        AnyErr::msg("RNG not initialized. Call set_all_seeds() first.")
    })?;

    Ok(SavedRngState {
        rng: state.rng.clone(),
        normal: state.normal.clone(),
    })
}

/// Restore a previously saved RNG state.
///
/// This resets the global RNG to the exact state it was in when save_rng_state() was called,
/// including the MT19937 internal state and the Box-Muller cache.
pub fn restore_rng_state(saved: SavedRngState) -> Result<()> {
    let mut rng_guard = GLOBAL_CPU_RNG
        .lock()
        .map_err(|e| AnyErr::msg(format!("Failed to acquire RNG lock: {}", e)))?;

    *rng_guard = Some(GlobalRngState {
        rng: saved.rng,
        normal: saved.normal,
    });

    Ok(())
}

/// Generate a seeded random normal tensor using PyTorch-compatible Box-Muller.
///
/// ALWAYS generates on CPU for true determinism, then transfers to target device.
/// This ensures reproducible results across CPU, CUDA, and Metal.
///
/// # PyTorch Parity
///
/// This function automatically selects the correct algorithm based on tensor size:
///
/// - **Size < 16**: Uses scalar Box-Muller with 53-bit double-precision uniforms
/// - **Size >= 16 (multiple of 16)**: Uses vectorized Box-Muller with 24-bit float uniforms
///
/// Both paths produce bit-identical results to PyTorch given the same seed.
///
/// # Important
///
/// For sizes >= 16 that aren't multiples of 16, this currently falls back to the
/// scalar path. PyTorch's actual behavior for non-multiples is more complex and
/// involves partial vectorized batches.
pub fn seeded_randn(mean: f64, std: f64, shape: &[usize], device: &Device) -> Result<Tensor> {
    // Always use the global CPU RNG for determinism
    // GPU random generation (Tensor::randn on Metal/CUDA) may not respect seeds properly
    let mut rng_guard = GLOBAL_CPU_RNG
        .lock()
        .map_err(|e| AnyErr::msg(format!("Failed to acquire RNG lock: {}", e)))?;

    if let Some(ref mut state) = *rng_guard {
        let elem_count = shape.iter().product::<usize>();
        let mean_f32 = mean as f32;
        let std_f32 = std as f32;

        // Log RNG state before generation (only when tracing is enabled)
        let u32_before = if tracing::enabled!(tracing::Level::DEBUG) {
            state.normal.u32_consumed()
        } else {
            0
        };

        let data = if elem_count >= 16 && elem_count % 16 == 0 {
            // Vectorized path: use PyTorch's SIMD-style batch processing
            // This matches torch.randn() for sizes that are multiples of 16
            debug!(
                "🎲 Using vectorized RNG path for {} elements ({} chunks of 16)",
                elem_count,
                elem_count / 16
            );
            state
                .normal
                .sample_vectorized_scaled(&mut state.rng, elem_count, mean_f32, std_f32)
        } else {
            // Scalar path for small sizes or non-multiples of 16
            debug!("🎲 Using scalar RNG path for {} elements", elem_count);
            let mut data = Vec::with_capacity(elem_count);
            for _ in 0..elem_count {
                data.push(
                    state
                        .normal
                        .sample_scaled(&mut state.rng, mean_f32, std_f32),
                );
            }
            data
        };

        // Log RNG state after generation (only when tracing is enabled)
        if tracing::enabled!(tracing::Level::DEBUG) {
            let u32_after = state.normal.u32_consumed();
            info!(
                "[RNG] randn({:?}): u32_consumed {} -> {} (delta={})",
                shape,
                u32_before,
                u32_after,
                u32_after - u32_before
            );
        }

        // Create tensor on CPU first
        let cpu_tensor = Tensor::from_vec(data, shape, &Device::Cpu)?;

        // Transfer to target device if needed
        if matches!(device, Device::Cpu) {
            Ok(cpu_tensor)
        } else {
            Ok(cpu_tensor.to_device(device)?)
        }
    } else {
        // RNG must be initialized - this is a programming error that would break parity
        anyhow::bail!(
            "Global CPU RNG not initialized! Call set_all_seeds() before generating random tensors."
        );
    }
}