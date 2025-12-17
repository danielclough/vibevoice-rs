#!/usr/bin/env python3
"""Convert PyTorch voice cache (.pt) to Safetensors format for Rust inference.

This script converts the voice cache files used by VibeVoice Realtime (0.5B)
from PyTorch checkpoint format to Safetensors format that can be loaded by Rust.

Input format (from Python VibeVoice):
{
    'lm': {'last_hidden_state': Tensor, 'past_key_values': DynamicCache or list},
    'tts_lm': {'last_hidden_state': Tensor, 'past_key_values': DynamicCache or list},
    'neg_lm': {'last_hidden_state': Tensor, 'past_key_values': DynamicCache or list},
    'neg_tts_lm': {'last_hidden_state': Tensor, 'past_key_values': DynamicCache or list}
}

Output format (Safetensors):
{
    'lm/last_hidden_state': Tensor,
    'lm/past_key_values/0/key': Tensor,
    'lm/past_key_values/0/value': Tensor,
    ...
    'tts_lm/last_hidden_state': Tensor,
    'tts_lm/past_key_values/0/key': Tensor,
    ...
}

Usage:
    python scripts/convert_voice_cache.py input.pt output.safetensors
    python scripts/convert_voice_cache.py --batch voices/*.pt --output-dir converted/
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from safetensors.torch import save_file


def extract_kv_pairs(past_kv: Any) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Extract (key, value) pairs from various cache formats.

    Handles:
    - transformers DynamicCache (has .key_cache, .value_cache)
    - List of (key, value) tuples
    - List of objects with __getitem__
    """
    pairs = []

    if hasattr(past_kv, 'key_cache') and hasattr(past_kv, 'value_cache'):
        # transformers DynamicCache object (newer format)
        for k, v in zip(past_kv.key_cache, past_kv.value_cache):
            pairs.append((k, v))
    elif hasattr(past_kv, '__iter__'):
        # Legacy format: list of tuples or nested structures
        for i, layer_cache in enumerate(past_kv):
            if isinstance(layer_cache, tuple) and len(layer_cache) == 2:
                k, v = layer_cache
            elif hasattr(layer_cache, '__getitem__'):
                k, v = layer_cache[0], layer_cache[1]
            else:
                raise ValueError(f"Unknown KV cache format at layer {i}: {type(layer_cache)}")
            pairs.append((k, v))
    else:
        raise ValueError(f"Unknown past_key_values format: {type(past_kv)}")

    return pairs


def convert_cache_entry(entry: Dict, prefix: str, preserve_dtype: bool = True) -> Dict[str, torch.Tensor]:
    """Convert a single cache entry (lm, tts_lm, etc.) to flat tensor dict.

    Args:
        entry: Dict containing 'last_hidden_state' and 'past_key_values'
        prefix: Prefix for tensor keys (e.g., 'lm', 'tts_lm')
        preserve_dtype: If True, keep original dtype (bf16). If False, convert to f32.
                       NOTE: bf16 is REQUIRED for correct long-text generation!
    """
    tensors = {}

    # Extract last_hidden_state
    if 'last_hidden_state' not in entry:
        raise ValueError(f"Missing 'last_hidden_state' in {prefix}")

    hidden = entry['last_hidden_state']
    if hasattr(hidden, 'contiguous'):
        hidden = hidden.contiguous()

    if preserve_dtype:
        # Keep original dtype (bf16) - IMPORTANT for long text generation
        tensors[f'{prefix}/last_hidden_state'] = hidden
    else:
        # Legacy: convert to f32 (causes garbled audio for long text!)
        tensors[f'{prefix}/last_hidden_state'] = hidden.to(torch.float32)

    # Extract past_key_values
    if 'past_key_values' not in entry:
        raise ValueError(f"Missing 'past_key_values' in {prefix}")

    kv_pairs = extract_kv_pairs(entry['past_key_values'])

    for i, (k, v) in enumerate(kv_pairs):
        k_tensor = k.contiguous()
        v_tensor = v.contiguous()

        if not preserve_dtype:
            # Legacy: convert to f32
            k_tensor = k_tensor.to(torch.float32)
            v_tensor = v_tensor.to(torch.float32)

        tensors[f'{prefix}/past_key_values/{i}/key'] = k_tensor
        tensors[f'{prefix}/past_key_values/{i}/value'] = v_tensor

    return tensors


def convert_voice_cache(input_path: Path, output_path: Path, verbose: bool = True) -> Dict[str, Any]:
    """Convert a single voice cache file.

    Returns metadata about the conversion.
    """
    if verbose:
        print(f"Loading {input_path}...")

    # Load PyTorch checkpoint
    cache = torch.load(input_path, map_location='cpu', weights_only=False)

    # Expected prefixes
    prefixes = ['lm', 'tts_lm', 'neg_lm', 'neg_tts_lm']

    # Validate structure
    for prefix in prefixes:
        if prefix not in cache:
            raise ValueError(f"Missing required key '{prefix}' in voice cache")

    # Convert all entries
    tensors = {}
    metadata = {'layers': {}}

    for prefix in prefixes:
        entry_tensors = convert_cache_entry(cache[prefix], prefix)
        tensors.update(entry_tensors)

        # Count layers
        num_layers = sum(1 for k in entry_tensors.keys() if '/key' in k)
        metadata['layers'][prefix] = num_layers

        if verbose:
            hidden = entry_tensors[f'{prefix}/last_hidden_state']
            hidden_shape = list(hidden.shape)
            hidden_dtype = hidden.dtype
            print(f"  {prefix}: {num_layers} layers, hidden_state shape={hidden_shape}, dtype={hidden_dtype}")

    if verbose:
        print(f"Saving to {output_path}...")
        print(f"  Total tensors: {len(tensors)}")

    # Save as safetensors
    save_file(tensors, output_path)

    if verbose:
        print("Done!")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description='Convert PyTorch voice cache to Safetensors format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Single file mode
    parser.add_argument('input', nargs='?', type=Path,
                        help='Input .pt file (single file mode)')
    parser.add_argument('output', nargs='?', type=Path,
                        help='Output .safetensors file (single file mode)')

    # Batch mode
    parser.add_argument('--batch', nargs='+', type=Path,
                        help='Input .pt files for batch conversion')
    parser.add_argument('--output-dir', type=Path,
                        help='Output directory for batch conversion')

    # Options
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress progress output')

    args = parser.parse_args()
    verbose = not args.quiet

    # Determine mode
    if args.batch:
        # Batch mode
        if not args.output_dir:
            parser.error('--output-dir required for batch mode')

        args.output_dir.mkdir(parents=True, exist_ok=True)

        for input_path in args.batch:
            output_path = args.output_dir / input_path.with_suffix('.safetensors').name
            try:
                convert_voice_cache(input_path, output_path, verbose)
            except Exception as e:
                print(f"Error converting {input_path}: {e}", file=sys.stderr)
                continue

    elif args.input and args.output:
        # Single file mode
        convert_voice_cache(args.input, args.output, verbose)

    else:
        parser.error('Provide input/output files or use --batch mode')


if __name__ == '__main__':
    main()
