import os

def compare_and_delete_redundant_files(dir1, dir2, delete=False):
    files1 = set(os.listdir(dir1))
    files2 = set(os.listdir(dir2))
    redundant_files = files2 - files1
    
    if not redundant_files:
        print("No redundant files found.")
        return
    
    print("Redundant files in", dir2, "are:")
    for file in redundant_files:
        print(file)
    
    if delete:
        for file in redundant_files:
            file_path = os.path.join(dir2, file)
            os.remove(file_path)
            print(f"Deleted: {file_path}")

if __name__ == "__main__":
    dir1 = input("Enter the first directory path: ")
    dir2 = input("Enter the second directory path: ")
    
    if not os.path.isdir(dir1) or not os.path.isdir(dir2):
        print("Invalid directory paths.")
    else:
        compare_and_delete_redundant_files(dir1, dir2, delete=True)

