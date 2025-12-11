# Test Helpers - Rust-Python Parity Validation

This module provides utilities to simplify writing Rust tests that validate parity with Python checkpoints.

## Overview

The test helpers eliminate boilerplate for:
- **Loading** `.npz` checkpoint files
- **Converting** between `ndarray` and Candle `Tensor`
- **Saving** test outputs as `.npz` files
- **Formatting** test output

## Quick Example

**Before (Manual):**
```rust
// 100+ lines of boilerplate
let file = File::open("checkpoint.npz")?;
let mut npz = NpzReader::new(file)?;
let arr: Array3<f32> = npz.by_name("input.npy")?;
let tensor = Tensor::from_slice(arr.as_slice().unwrap(), arr.shape(), &device)?;
// ... operations ...
let data = tensor.flatten_all()?.to_vec1::<f32>()?;
let (d0, d1, d2) = tensor.shape().dims3()?;
let output_arr = Array3::from_shape_vec((d0, d1, d2), data)?;
let mut writer = NpzWriter::new(File::create("output.npz")?);
writer.add_array("output", &output_arr)?;
writer.finish()?;
```

**After (With Helpers):**
```rust
// ~10 lines - clean and readable
use vibevoice_rs::test_helpers::*;

let device = get_device()?;
let mut checkpoint = Checkpoint::open("checkpoint.npz")?;

let input_np = checkpoint.load_array3("input")?;
let input = input_np.to_tensor(&device)?;

// ... your test operations ...

let mut writer = CheckpointWriter::create("output.npz")?;
writer.add_array3("output", &output.to_array3()?)?;
writer.finish()?;
```

## API Reference

### Loading Checkpoints

```rust
// Open a checkpoint file
let mut checkpoint = Checkpoint::open("path/to/file.npz")?;

// Load arrays of different dimensions
let arr1 = checkpoint.load_array1("name")?;  // 1D array
let arr2 = checkpoint.load_array2("name")?;  // 2D array
let arr3 = checkpoint.load_array3("name")?;  // 3D array
let arr4 = checkpoint.load_array4("name")?;  // 4D array

// Load scalar values
let scalar = checkpoint.load_scalar("bias")?;  // Single f32 value
```

### Converting Arrays ↔ Tensors

```rust
use vibevoice_rs::test_helpers::ToTensor;

// ndarray → Candle Tensor
let arr = Array3::from_shape_vec((1, 2, 3), vec![...])?;
let tensor = arr.to_tensor(&device)?;

// Candle Tensor → ndarray
let arr1 = tensor.to_array1()?;
let arr2 = tensor.to_array2()?;
let arr3 = tensor.to_array3()?;
let arr4 = tensor.to_array4()?;
```

### Saving Checkpoints

```rust
let mut writer = CheckpointWriter::create("output.npz")?;
writer.add_array1("bias", &bias_arr)?;
writer.add_array3("output", &output_arr)?;
writer.finish()?;  // Don't forget to call finish()!
```

### Formatting Helpers

```rust
// Print test header
print_header("PHASE 1.1", "NORMALIZATION FORMULA TEST");
// Output: ======================================================================
//         PHASE 1.1: NORMALIZATION FORMULA TEST
//         ======================================================================

// Print section
print_section("Test Input");
// Output: 🔹 Test Input

// Print tensor info
print_tensor_info("my_tensor", &tensor)?;
// Output:   my_tensor: [1, 70, 1536]
//           my_tensor[:5]: [0.123, -0.456, ...]
//           Mean: 0.004521

// Print comparison result
print_comparison("output", 1.5e-5, 1e-4);
// Output: ✅ output: max_diff=1.50e-05 (tolerance=1.00e-04)

// Manual verification of operation
verify_operation("Normalization", &inputs, &outputs, |x| (x + bias) * scale);
// Output: 🔍 Manual Verification (Normalization):
//           [0] input=+1.000000, expected=0.18660879, actual=0.18660879 ✅
//           [1] input=+2.000000, expected=0.38289785, actual=0.38289785 ✅

// Print success message with next steps
print_success("Test name", "python.npz", "rust.npz", 1e-6);
// Output: ======================================================================
//         ✅ Test name complete!
//         ======================================================================
//
//         🎯 Next: Run comparison:
//            python debug/tools/compare.py \
//                python.npz \
//                rust.npz \
//                --tolerance 1e-6
```

## Complete Test Template

```rust
use anyhow::Result;
use vibevoice_rs::test_helpers::*;

fn main() -> Result<()> {
    print_header("PHASE X.Y", "YOUR TEST NAME");

    let device = get_device()?;

    // 1. Load Python checkpoint
    info!("\n📥 Loading Python checkpoint...");
    let mut checkpoint = Checkpoint::open("debug/checkpoints/your_test.npz")?;

    let input_np = checkpoint.load_array3("input")?;
    let param = checkpoint.load_scalar("param")?;

    // 2. Convert to Candle tensors
    let input = input_np.to_tensor(&device)?;

    print_section("Input");
    print_tensor_info("input", &input)?;

    // 3. Run your test operations
    let output = your_operation(&input, param)?;

    print_section("Output");
    print_tensor_info("output", &output)?;

    // 4. Save Rust output
    info!("\n💾 Saving Rust output...");
    let mut writer = CheckpointWriter::create("rust_your_test.npz")?;
    writer.add_array3("input", &input_np)?;
    writer.add_array3("output", &output.to_array3()?)?;
    writer.finish()?;

    print_success(
        "Your test",
        "debug/checkpoints/your_test.npz",
        "rust_your_test.npz",
        1e-4,  // tolerance
    );

    Ok(())
}
```

## Design Patterns

### Pattern 1: Test with Manual Verification

Use when you want to verify the formula step-by-step:

```rust
// Save input before it's moved
let input_slice = input.flatten_all().and_then(|t| t.to_vec1::<f32>())?;

// Run operation (consumes input)
let output = operation(input)?;

// Verify against expected formula
let output_slice = output.flatten_all().and_then(|t| t.to_vec1::<f32>())?;
verify_operation("Operation", &input_slice, &output_slice, |x| expected_fn(x));
```

### Pattern 2: Test with Layer-by-Layer Outputs

For multi-layer operations (like connector):

```rust
let x1 = layer1.forward(&input)?;
writer.add_array3("after_layer1", &x1.to_array3()?)?;

let x2 = layer2.forward(&x1)?;
writer.add_array3("after_layer2", &x2.to_array3()?)?;

let x3 = layer3.forward(&x2)?;
writer.add_array3("after_layer3", &x3.to_array3()?)?;
```

### Pattern 3: Test with Comparison Against Python

```rust
// Load expected output from Python
let expected_np = checkpoint.load_array3("expected_output")?;
let expected = expected_np.to_tensor(&device)?;

// Compare
let diff = (output - expected)?.abs()?;
let max_diff = diff.max_all()?.to_scalar::<f32>()?;

print_comparison("output", max_diff, tolerance);
```

## Tips

1. **Call `finish()` on CheckpointWriter** - Required to actually write the file
2. **Extract data before moving tensors** - Operations consume tensors, so extract slices first if needed for verification
3. **Use consistent naming** - Match Python checkpoint keys exactly
4. **Print intermediate steps** - Helps debug where divergence occurs
5. **Save all intermediate outputs** - Layer-by-layer for complex operations

## Example Tests

See these files for complete examples:
- `src/bin/test_normalization.rs` - Simplest example
- `src/bin/test_connector.rs` - Multi-layer example (coming soon)
- `src/bin/test_vae.rs` - Complex operation example (coming soon)

## Common Errors

### Error: "borrow of moved value"
```rust
// ❌ Wrong - tensor moved by operation
let output = operation(tensor)?;
let slice = tensor.flatten_all()?;  // ERROR!

// ✅ Correct - extract slice before moving
let slice = tensor.flatten_all().and_then(|t| t.to_vec1::<f32>())?;
let output = operation(tensor)?;
```

### Error: "dimensions don't match"
```rust
// ❌ Wrong - using wrong array type
let arr = checkpoint.load_array2("my_3d_tensor")?;  // ERROR!

// ✅ Correct - match the actual dimensions
let arr = checkpoint.load_array3("my_3d_tensor")?;
```

### Error: "file not found"
```rust
// ❌ Wrong - forgot to call Python test first
cargo run --bin test_connector

// ✅ Correct - run Python first to generate checkpoint
./debug/run_python.sh 04_test_connector.py
cargo run --bin test_connector
```

## Testing the Helpers

The module includes unit tests:
```bash
cargo test --lib test_helpers
```
