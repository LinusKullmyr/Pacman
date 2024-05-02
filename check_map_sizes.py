import os


def read_lay_files(directory):
    # List all files in the directory
    files = os.listdir(directory)

    # Filter and process each .lay file
    for file in files:
        if file.endswith(".lay"):
            file_path = os.path.join(directory, file)
            with open(file_path, "r") as f:
                lines = f.readlines()

            h = len(lines)
            w = len(lines[0])
            print(file, w, h)


# Example usage:
directory_path = os.path.join(os.getcwd(), "layouts")
read_lay_files(directory_path)
