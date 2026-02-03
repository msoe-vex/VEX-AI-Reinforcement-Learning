import argparse

def remove_null_bytes(file_path):
    with open(file_path, 'rb') as f:
        content = f.read()
    
    occurrences = content.count(b'\x00')

    if occurrences == 0:
        print(f"No null bytes found in {file_path}")
        return

    cleaned_content = content.replace(b'\x00', b'')
    
    with open(file_path, 'wb') as f:
        f.write(cleaned_content)

    print(f"Found and removed {occurrences} null byte(s) from {file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove null bytes from a Python file.")
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to the Python file to clean."
    )

    args = parser.parse_args()
    remove_null_bytes(args.file)