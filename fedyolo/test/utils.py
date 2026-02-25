import re
import os


def extract_results_path(file_path):
    with open(file_path, "r") as file:
        file_content = file.read()
    lines = file_content.split("\n")
    for line in reversed(lines):
        if line.startswith("Results saved to"):
            # Extract the path with or without ANSI escape sequences
            match = re.search(r"Results saved to (.+)", line)
            if match:
                # Remove ANSI escape sequences
                absolute_path = re.sub(
                    r"\x1b\[[0-9;]*[a-zA-Z]", "", match.group(1)
                ).strip()
                # Convert absolute path to relative path if needed
                relative_path = os.path.relpath(absolute_path, start=os.getcwd())
                return relative_path
    return None
