import os
import gzip
import json
import argparse
from collections import defaultdict

def process_files(folder_path):
    object_counts = defaultdict(int)

    # Iterate through all files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".json.gz"):
            file_path = os.path.join(folder_path, filename)
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                data = json.load(f)
                
                # Iterate through episodes and tasks
                for episode in data.get("episodes", []):
                    for task in episode.get("tasks", []):
                        if task[0]:  # Ensure there is an object in the task
                            object_counts[task[0]] += 1

    return dict(object_counts)

def save_results(results, output_file):
    sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sorted_results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and count objects in json.gz files.")
    parser.add_argument('--split', type=str, default='train')
    args = parser.parse_args()
    folder_path = f"data/datasets/goat_bench/hm3d/v1/{args.split}/content"  # Replace with your folder path
    output_file = f"data/object_count_info/{args.split}_object_counts.json"  # Replace with your desired output file path
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results = process_files(folder_path)
    save_results(results, output_file)

    print(f"Results saved to {output_file}")
