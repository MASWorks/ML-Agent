import json
import sys

def add_datasets(data_name, data_path,  dataset_info_path='LLaMA-Factory/data/dataset_info.json'):
    with open(dataset_info_path, 'r') as f:
        dataset_info = json.load(f)
    
    dataset_info[data_name] = {
        "file_name": data_path,
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations"
        },
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt"
        }
        }
    with open(dataset_info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    print(f"Datasets added to {dataset_info_path}")
    return True


def main():


    data_name = sys.argv[1]
    data_path =sys.argv[2]
    add_datasets(data_name, data_path)
    print("Datasets have been successfully added!")

if __name__ == '__main__':
    main()
