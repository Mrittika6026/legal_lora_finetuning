#!/usr/bin/env python3
"""
Convert pretty-printed JSONL to standard JSONL format (one JSON object per line).
"""
import json
from pathlib import Path
from tqdm import tqdm

def convert_pretty_jsonl_to_standard(input_path: str, output_path: str):
    """Convert pretty-printed JSONL to standard compact JSONL."""
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Count total lines for progress bar
    print(f"Counting lines in {input_file.name}...")
    total_lines = sum(1 for _ in input_file.open("r", encoding="utf-8"))
    print(f"Found {total_lines} lines")
    
    buffer = []
    parsed_count = 0
    error_count = 0
    
    print(f"Converting {input_file.name} to standard JSONL format...")
    with input_file.open("r", encoding="utf-8") as f_in, \
         output_file.open("w", encoding="utf-8") as f_out:
        
        for line in tqdm(f_in, total=total_lines, desc="Processing", unit="lines"):
            stripped = line.strip()
            
            # Handle empty lines (separators between JSON objects)
            if not stripped:
                if buffer:
                    # Try to parse accumulated buffer
                    try:
                        obj = json.loads("".join(buffer))
                        # Write as compact JSON on a single line
                        f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        parsed_count += 1
                        buffer = []
                    except json.JSONDecodeError as e:
                        error_count += 1
                        if error_count <= 5:  # Show first 5 errors
                            print(f"\nWarning: Failed to parse JSON object: {e}")
                        buffer = []
                continue
            
            # Add line to buffer
            buffer.append(line)
            
            # Try parsing after each line (handles both single-line and multi-line JSON)
            try:
                obj = json.loads("".join(buffer))
                # Successfully parsed! Write it
                f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                parsed_count += 1
                buffer = []
            except json.JSONDecodeError:
                # Not yet complete, continue accumulating
                pass
        
        # Handle any remaining buffer
        if buffer:
            try:
                obj = json.loads("".join(buffer))
                f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                parsed_count += 1
            except json.JSONDecodeError as e:
                error_count += 1
                print(f"\nWarning: Failed to parse final JSON object: {e}")
    
    print(f"\n✓ Conversion complete!")
    print(f"  Parsed: {parsed_count} JSON objects")
    print(f"  Errors: {error_count}")
    print(f"  Output: {output_file}")
    
    # Show file sizes
    input_size = input_file.stat().st_size / (1024 * 1024)  # MB
    output_size = output_file.stat().st_size / (1024 * 1024)  # MB
    print(f"  Input size: {input_size:.2f} MB")
    print(f"  Output size: {output_size:.2f} MB")


if __name__ == "__main__":
    import sys
    
    # Default paths
    data_dir = Path(__file__).parent / "data"
    train_input = data_dir / "train_instruct.jsonl"
    train_output = data_dir / "train_instruct_standard.jsonl"
    val_input = data_dir / "val_instruct.jsonl"
    val_output = data_dir / "val_instruct_standard.jsonl"
    
    # Convert training file
    if train_input.exists():
        print("=" * 60)
        print("Converting training file...")
        print("=" * 60)
        convert_pretty_jsonl_to_standard(str(train_input), str(train_output))
        print()
    
    # Convert validation file
    if val_input.exists():
        print("=" * 60)
        print("Converting validation file...")
        print("=" * 60)
        convert_pretty_jsonl_to_standard(str(val_input), str(val_output))
        print()
    
    print("=" * 60)
    print("All conversions complete!")
    print("=" * 60)
    print("\nTo use the converted files, update your config:")
    print("  train_file: \"data/train_instruct_standard.jsonl\"")
    print("  eval_file: \"data/val_instruct_standard.jsonl\"")


