# DGF


## Requirements

```bash
Python==3.9.18
torch==1.11.0
pytorch-cluster==1.6.0
pytorch-scatter==2.0.9
pytorch-sparse==0.6.15
dgl==0.9.1.post1
numpy==1.26.1
pandas==1.5.3
scikit-learn==1.2.2
munkres==1.1.4
huggingface-hub==0.29.3
```
## Datasets

To download the datasets, follow these steps:

1. First, request access to the dataset from https://huggingface.co/Sherirto/MAG
2. Login to your Hugging Face account: `huggingface-cli login`
3. Use our download script:

```bash
# Download a specific dataset
python src/download_datasets.py --dataset Movies

# Download all datasets
python src/download_datasets.py --dataset all
```

Available datasets: Movies, Toys, Grocery, GroceryS, RedditS, Reddit, Photo, Arts


## Usage

### Run Example

```bash
python src/main.py --dataset Movies
```

## Project Structure

```
src/
├── main.py                     # Main entry point
├── model.py                    # Graph clustering model
├── trainer.py                  # Trainer
├── load_data.py                # Data loading
├── download_datasets.py        # Dataset download utility
├── clustering_metrics.py       # Clustering evaluation metrics
├── community_contrastive.py    # Community contrastive learning
├── cross_modal_contrastive.py  # Cross-modal contrastive learning
└── graph_contrastive.py        # Graph contrastive learning
```

## Reproductivity
To reproduce our empirical results, run the `run.sh` script.