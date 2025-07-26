import argparse
import os
from huggingface_hub import snapshot_download

# Define the list of all available dataset names
ALL_DATASETS = ['Movies', 'Toys', 'RedditS', 'Grocery', 'GroceryS', 'Photo', 'Arts', 'Reddit']

def get_required_files(dataset_name):
    """
    Generates the list of specific file paths to download for a given dataset.
    Note: These paths are relative to the Hugging Face repository root.
    """
    if dataset_name == "RedditS":
        text_file = f'{dataset_name}/TextFeature/{dataset_name}_gemma_7b_64_mean.npy'
    elif dataset_name == "Reddit":
        text_file = f'{dataset_name}/TextFeature/{dataset_name}_gemma_7b_100_mean.npy'
    else:
        text_file = f'{dataset_name}/TextFeature/{dataset_name}_gemma_7b_256_mean.npy'
    
    image_file = f'{dataset_name}/ImageFeature/{dataset_name}_openai_clip-vit-large-patch14.npy'
    graph_file = f'{dataset_name}/{dataset_name}Graph.pt'
    csv_file = f'{dataset_name}/{dataset_name}.csv'
    
    return [text_file, image_file, graph_file, csv_file]

def download_dataset(dataset_name, local_dir='./MAG'):
    """Download specific files for a single dataset from Hugging Face."""
    print(f"Downloading specified files for {dataset_name} dataset...")
    
    files_to_download = get_required_files(dataset_name)
    
    try:
        snapshot_download(
            repo_id='Sherirto/MAG', 
            allow_patterns=files_to_download, 
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print(f"Successfully downloaded specified files for {dataset_name} to {local_dir}")
    except Exception as e:
        print(f"Error downloading {dataset_name}: {str(e)}")
        print("Please make sure you have:")
        print("1. Requested access to the dataset at https://huggingface.co/Sherirto/MAG")
        print("2. Logged in to your Hugging Face account using 'huggingface-cli login'")

def download_all_datasets(local_dir='./MAG'):
    """Download specific files for all available datasets."""
    print("Downloading specified files for all datasets...")
    
    all_files_to_download = []
    for dataset in ALL_DATASETS:
        all_files_to_download.extend(get_required_files(dataset))
        
    try:
        snapshot_download(
            repo_id='Sherirto/MAG', 
            allow_patterns=all_files_to_download, 
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print(f"Successfully downloaded all specified dataset files to {local_dir}")
    except Exception as e:
        print(f"Error downloading datasets: {str(e)}")
        print("Please make sure you have:")
        print("1. Requested access to the dataset at https://huggingface.co/Sherirto/MAG")
        print("2. Logged in to your Hugging Face account using 'huggingface-cli login'")

def main():
    parser = argparse.ArgumentParser(description="Download MAG datasets from Hugging Face")
    parser.add_argument('--dataset', type=str, default='all', 
                        choices=['all'] + ALL_DATASETS,
                        help='Dataset to download (default: all)')
    parser.add_argument('--local_dir', type=str, default='./MAG',
                        help='Local directory to save datasets (default: ./MAG)')
    
    args = parser.parse_args()
    
    os.makedirs(args.local_dir, exist_ok=True)
    
    if args.dataset == 'all':
        download_all_datasets(args.local_dir)
    else:
        download_dataset(args.dataset, args.local_dir)

if __name__ == "__main__":
    main()