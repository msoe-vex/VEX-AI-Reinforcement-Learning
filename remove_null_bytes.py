def remove_null_bytes(file_path):
    with open(file_path, 'rb') as f:
        content = f.read()
    
    cleaned_content = content.replace(b'\x00', b'')
    
    with open(file_path, 'wb') as f:
        f.write(cleaned_content)

if __name__ == "__main__":
    file_path = 'rl_environment.py'
    remove_null_bytes(file_path)
    print(f"Null bytes removed from {file_path}")
