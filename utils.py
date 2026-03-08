import json

def output_to_json_file(json_dict, output_file):
    print(f"=== Writing to JSON ===\nFile: {output_file}\nContent: {json_dict}\n")
    with open(output_file, "w") as fp:
        json.dump(json_dict, fp, indent=4)
