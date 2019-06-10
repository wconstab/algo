import os 


def load_input(filename):
    input = []
    with open(filename) as f:
        for line in f:
            input.append(int(line))
    return input


def build_filename(dir_name_from_file, file_name):
    dir_path = os.path.dirname(os.path.realpath(dir_name_from_file))
    return os.path.join(dir_path, file_name)
