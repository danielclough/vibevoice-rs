/*!
 * Test helpers for Rust-Python parity validation
 *
 * Provides utilities for:
 * - Loading/saving .npz checkpoint files
 * - Converting between ndarray and Candle tensors
 * - Comparing outputs and printing results
 * - File-based logging (no stdout)
 */

use anyhow::{Context, Result, anyhow};
use candle_core::{Device, Tensor};
use ndarray::{Array1, Array2, Array3, Array4, ArrayD};
use ndarray_npy::{NpzReader, NpzWriter};
use std::fs::File;
use std::path::Path;
use tracing::info;

/// Load a checkpoint from a .npz file
pub struct Checkpoint {
    reader: NpzReader<File>,
}

impl Checkpoint {
    /// Open a checkpoint file
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())
            .with_context(|| format!("Failed to open checkpoint: {:?}", path.as_ref()))?;
        let reader = NpzReader::new(file)?;
        Ok(Self { reader })
    }

    /// Load a 1D array
    pub fn load_array1(&mut self, name: &str) -> Result<Array1<f32>> {
        self.reader
            .by_name(name)
            .with_context(|| format!("Failed to load array: {}", name))
    }

    /// Load a 2D array
    pub fn load_array2(&mut self, name: &str) -> Result<Array2<f32>> {
        self.reader
            .by_name(name)
            .with_context(|| format!("Failed to load array: {}", name))
    }

    /// Load a 3D array
    pub fn load_array3(&mut self, name: &str) -> Result<Array3<f32>> {
        self.reader
            .by_name(name)
            .with_context(|| format!("Failed to load array: {}", name))
    }

    /// Load a 4D array
    pub fn load_array4(&mut self, name: &str) -> Result<Array4<f32>> {
        self.reader
            .by_name(name)
            .with_context(|| format!("Failed to load array: {}", name))
    }

    /// Load a dynamic array (any number of dimensions)
    pub fn load_array_dyn(&mut self, name: &str) -> Result<ArrayD<f32>> {
        self.reader
            .by_name(name)
            .with_context(|| format!("Failed to load array: {}", name))
    }

    /// Load a scalar (single value)
    pub fn load_scalar(&mut self, name: &str) -> Result<f32> {
        let arr: Array1<f32> = self.load_array1(name)?;
        Ok(arr[0])
    }

    /// Load a scalar (single value) as f64
    pub fn load_scalar_f64(&mut self, name: &str) -> Result<f64> {
        let arr: Array1<f64> = self
            .reader
            .by_name(name)
            .with_context(|| format!("Failed to load array: {}", name))?;
        Ok(arr[0])
    }

    /// Load a 1D array as i64 (for integer arrays like token IDs)
    pub fn load_array1_i64(&mut self, name: &str) -> Result<Array1<i64>> {
        self.reader
            .by_name(name)
            .with_context(|| format!("Failed to load array: {}", name))
    }

    /// Load a 1D array as i32 (for integer arrays)
    pub fn load_array1_i32(&mut self, name: &str) -> Result<Array1<i32>> {
        self.reader
            .by_name(name)
            .with_context(|| format!("Failed to load array: {}", name))
    }

    /// Load a 2D array as i64 (for token IDs, attention masks)
    pub fn load_array2_i64(&mut self, name: &str) -> Result<Array2<i64>> {
        self.reader
            .by_name(name)
            .with_context(|| format!("Failed to load array: {}", name))
    }

    /// Load a 1D array as f64 and convert to f32
    pub fn load_array1_f64_as_f32(&mut self, name: &str) -> Result<Array1<f32>> {
        let arr: Array1<f64> = self
            .reader
            .by_name(name)
            .with_context(|| format!("Failed to load array: {}", name))?;
        Ok(arr.mapv(|x| x as f32))
    }

    /// Load a 1D array that could be f32 or f64, returning f32
    pub fn load_array1_auto(&mut self, name: &str) -> Result<Array1<f32>> {
        // Try f32 first
        if let Ok(arr) = self.load_array1(name) {
            return Ok(arr);
        }
        // Try f64 and convert
        self.load_array1_f64_as_f32(name)
    }

    /// Load a 2D array as f64 and convert to f32
    pub fn load_array2_f64_as_f32(&mut self, name: &str) -> Result<Array2<f32>> {
        let arr: Array2<f64> = self
            .reader
            .by_name(name)
            .with_context(|| format!("Failed to load array: {}", name))?;
        Ok(arr.mapv(|x| x as f32))
    }

    /// Load a 3D array as f64 and convert to f32
    pub fn load_array3_f64_as_f32(&mut self, name: &str) -> Result<Array3<f32>> {
        let arr: Array3<f64> = self
            .reader
            .by_name(name)
            .with_context(|| format!("Failed to load array: {}", name))?;
        Ok(arr.mapv(|x| x as f32))
    }

    /// Load a scalar as i64
    pub fn load_scalar_i64(&mut self, name: &str) -> Result<i64> {
        let arr: Array1<i64> = self.load_array1_i64(name)?;
        Ok(arr[0])
    }

    /// Load a scalar that might be stored as int64 and convert to usize
    pub fn load_scalar_as_usize(&mut self, name: &str) -> Result<usize> {
        // Try i64 first (most common for Python integers)
        if let Ok(arr) = self.load_array1_i64(name) {
            return Ok(arr[0] as usize);
        }
        // Try i32
        if let Ok(arr) = self.load_array1_i32(name) {
            return Ok(arr[0] as usize);
        }
        // Try f64 (sometimes Python saves ints as floats)
        if let Ok(arr) = self.load_array1_f64_as_f32(name) {
            return Ok(arr[0] as usize);
        }
        // Try f32
        if let Ok(arr) = self.load_array1(name) {
            return Ok(arr[0] as usize);
        }
        Err(anyhow!("Failed to load {} as any integer type", name))
    }

    /// Try to load a scalar that could be f32, f64, i64, or i32
    pub fn load_scalar_auto(&mut self, name: &str) -> Result<f32> {
        // Try f32 first
        if let Ok(arr) = self.load_array1(name) {
            return Ok(arr[0]);
        }
        // Try f64 and convert
        if let Ok(arr) = self.load_array1_f64_as_f32(name) {
            return Ok(arr[0]);
        }
        // Try i64 (Python often saves integers as i64)
        if let Ok(arr) = self.load_array1_i64(name) {
            return Ok(arr[0] as f32);
        }
        // Try i32
        if let Ok(arr) = self.load_array1_i32(name) {
            return Ok(arr[0] as f32);
        }
        Err(anyhow!("Failed to load {} as numeric type", name))
    }

    /// Load a 1D boolean array
    pub fn load_array1_bool(&mut self, name: &str) -> Result<Array1<bool>> {
        self.reader
            .by_name(name)
            .with_context(|| format!("Failed to load array: {}", name))
    }

    /// Load a 2D boolean array
    pub fn load_array2_bool(&mut self, name: &str) -> Result<Array2<bool>> {
        self.reader
            .by_name(name)
            .with_context(|| format!("Failed to load array: {}", name))
    }
}

/// Save arrays to a .npz file
pub struct CheckpointWriter {
    writer: NpzWriter<File>,
}

impl CheckpointWriter {
    /// Create a new checkpoint writer
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::create(path.as_ref())
            .with_context(|| format!("Failed to create checkpoint: {:?}", path.as_ref()))?;
        let writer = NpzWriter::new(file);
        Ok(Self { writer })
    }

    /// Add a 1D array
    pub fn add_array1(&mut self, name: &str, array: &Array1<f32>) -> Result<()> {
        self.writer
            .add_array(name, array)
            .with_context(|| format!("Failed to add array: {}", name))
    }

    /// Add a 2D array
    pub fn add_array2(&mut self, name: &str, array: &Array2<f32>) -> Result<()> {
        self.writer
            .add_array(name, array)
            .with_context(|| format!("Failed to add array: {}", name))
    }

    /// Add a 3D array
    pub fn add_array3(&mut self, name: &str, array: &Array3<f32>) -> Result<()> {
        self.writer
            .add_array(name, array)
            .with_context(|| format!("Failed to add array: {}", name))
    }

    /// Add a 4D array
    pub fn add_array4(&mut self, name: &str, array: &Array4<f32>) -> Result<()> {
        self.writer
            .add_array(name, array)
            .with_context(|| format!("Failed to add array: {}", name))
    }

    /// Finish writing and close the file
    pub fn finish(self) -> Result<()> {
        self.writer
            .finish()
            .context("Failed to finish writing")
            .map(|_| ())
    }
}

/// Convert ndarray to Candle tensor
pub trait ToTensor {
    fn to_tensor(&self, device: &Device) -> Result<Tensor>;
}

impl ToTensor for Array1<f32> {
    fn to_tensor(&self, device: &Device) -> Result<Tensor> {
        // Ensure standard layout for as_slice() to work
        let arr = self.as_standard_layout();
        Tensor::from_slice(arr.as_slice().unwrap(), arr.shape(), device)
            .context("Failed to convert Array1 to Tensor")
    }
}

impl ToTensor for Array2<f32> {
    fn to_tensor(&self, device: &Device) -> Result<Tensor> {
        // Ensure standard layout for as_slice() to work
        let arr = self.as_standard_layout();
        Tensor::from_slice(arr.as_slice().unwrap(), arr.shape(), device)
            .context("Failed to convert Array2 to Tensor")
    }
}

impl ToTensor for Array3<f32> {
    fn to_tensor(&self, device: &Device) -> Result<Tensor> {
        // Ensure standard layout for as_slice() to work
        let arr = self.as_standard_layout();
        Tensor::from_slice(arr.as_slice().unwrap(), arr.shape(), device)
            .context("Failed to convert Array3 to Tensor")
    }
}

impl ToTensor for Array4<f32> {
    fn to_tensor(&self, device: &Device) -> Result<Tensor> {
        // Ensure standard layout for as_slice() to work
        let arr = self.as_standard_layout();
        Tensor::from_slice(arr.as_slice().unwrap(), arr.shape(), device)
            .context("Failed to convert Array4 to Tensor")
    }
}

impl ToTensor for Array2<i64> {
    fn to_tensor(&self, device: &Device) -> Result<Tensor> {
        // Ensure standard layout for as_slice() to work
        let arr = self.as_standard_layout();
        Tensor::from_slice(arr.as_slice().unwrap(), arr.shape(), device)
            .context("Failed to convert Array2<i64> to Tensor")
    }
}

/// Trait for converting bool arrays to tensors
pub trait ToTensorBool {
    fn to_tensor_bool(&self, device: &Device) -> Result<Tensor>;
}

impl ToTensorBool for Array2<bool> {
    fn to_tensor_bool(&self, device: &Device) -> Result<Tensor> {
        // Convert bool to u8 (0 or 1) for tensor creation
        let arr = self.as_standard_layout();
        let u8_data: Vec<u8> = arr.iter().map(|&b| if b { 1u8 } else { 0u8 }).collect();
        Tensor::from_slice(&u8_data, arr.shape(), device)
            .context("Failed to convert Array2<bool> to Tensor")
    }
}

/// Convert Candle tensor to ndarray
pub trait ToNdarray {
    fn to_array1(&self) -> Result<Array1<f32>>;
    fn to_array2(&self) -> Result<Array2<f32>>;
    fn to_array3(&self) -> Result<Array3<f32>>;
    fn to_array4(&self) -> Result<Array4<f32>>;
}

impl ToNdarray for Tensor {
    fn to_array1(&self) -> Result<Array1<f32>> {
        let data = self.flatten_all().and_then(|t| t.to_vec1::<f32>())?;
        let shape = self.shape().dims1()?;
        Array1::from_shape_vec(shape, data).context("Failed to convert Tensor to Array1")
    }

    fn to_array2(&self) -> Result<Array2<f32>> {
        let data = self.flatten_all().and_then(|t| t.to_vec1::<f32>())?;
        let (d0, d1) = self.shape().dims2()?;
        Array2::from_shape_vec((d0, d1), data).context("Failed to convert Tensor to Array2")
    }

    fn to_array3(&self) -> Result<Array3<f32>> {
        let data = self.flatten_all().and_then(|t| t.to_vec1::<f32>())?;
        let (d0, d1, d2) = self.shape().dims3()?;
        Array3::from_shape_vec((d0, d1, d2), data).context("Failed to convert Tensor to Array3")
    }

    fn to_array4(&self) -> Result<Array4<f32>> {
        let data = self.flatten_all().and_then(|t| t.to_vec1::<f32>())?;
        let dims = self.shape().dims4()?;
        Array4::from_shape_vec((dims.0, dims.1, dims.2, dims.3), data)
            .context("Failed to convert Tensor to Array4")
    }
}

/// Print formatted test header
pub fn print_header(phase: &str, title: &str) {
    info!("{}", "=".repeat(70));
    info!("{}: {}", phase, title);
    info!("{}", "=".repeat(70));
}

/// Print formatted section
pub fn print_section(title: &str) {
    info!("\n🔹 {}", title);
}

/// Print tensor info
pub fn print_tensor_info(name: &str, tensor: &Tensor) -> Result<()> {
    info!("  {}: {:?}", name, tensor.shape());

    // Print first few values
    let data = tensor.flatten_all().and_then(|t| t.to_vec1::<f32>())?;
    let sample_size = 5.min(data.len());
    if sample_size > 0 {
        info!("  {}[:{}]: {:?}", name, sample_size, &data[..sample_size]);
    }

    // Print statistics
    let mean = tensor.mean_all()?.to_scalar::<f32>()?;
    info!("  Mean: {:.6}", mean);

    Ok(())
}

/// Print comparison result
pub fn print_comparison(name: &str, max_diff: f32, tolerance: f32) {
    let symbol = if max_diff < tolerance { "✅" } else { "❌" };
    info!(
        "{} {}: max_diff={:.2e} (tolerance={:.2e})",
        symbol, name, max_diff, tolerance
    );
}

/// Print manual verification of operation
pub fn verify_operation<F>(name: &str, inputs: &[f32], outputs: &[f32], expected_fn: F)
where
    F: Fn(f32) -> f32,
{
    info!("\n🔍 Manual Verification ({}):", name);
    for i in 0..5.min(inputs.len()) {
        let input = inputs[i];
        let expected = expected_fn(input);
        let actual = outputs[i];
        let diff = (expected - actual).abs();
        let match_symbol = if diff < 1e-6 { "✅" } else { "❌" };
        info!(
            "  [{}] input={:+.6}, expected={:.8}, actual={:.8} {}",
            i, input, expected, actual, match_symbol
        );
        if diff >= 1e-6 {
            info!("      Diff: {:.2e}", diff);
        }
    }
}

/// Get device (prefer Metal, fallback to CPU)
pub fn get_device() -> Result<Device> {
    let device = Device::new_metal(0).unwrap_or(Device::Cpu);
    info!("\nUsing device: {:?}", device);
    Ok(device)
}

/// Print success message with next steps
pub fn print_success(test_name: &str, python_path: &str, rust_path: &str, tolerance: f64) {
    info!("\n{}", "=".repeat(70));
    info!("✅ {} complete!", test_name);
    info!("{}", "=".repeat(70));
    info!("\n🎯 Next: Run comparison:");
    info!("   python debug/tools/compare.py \\");
    info!("       {} \\", python_path);
    info!("       {} \\", rust_path);
    info!("       --tolerance {:.0e}", tolerance);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_conversion() -> Result<()> {
        let device = Device::Cpu;

        // Test 1D
        let arr1 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let tensor = arr1.to_tensor(&device)?;
        let arr1_back = tensor.to_array1()?;
        assert_eq!(arr1, arr1_back);

        // Test 3D
        let arr3 = Array3::from_shape_vec((1, 2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
        let tensor = arr3.to_tensor(&device)?;
        let arr3_back = tensor.to_array3()?;
        assert_eq!(arr3, arr3_back);

        Ok(())
    }
}
