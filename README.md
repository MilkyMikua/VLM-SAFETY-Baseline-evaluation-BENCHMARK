# VLM Safety Baseline Evaluation Benchmark

This repository contains code and resources for evaluating the safety of Vision Language Models (VLMs). It includes the implementation of **ETA (Evaluating Then Aligning)** and uses the **Hateful Memes** dataset for benchmarking.

## Project Structure

- **ETA/**: Implementation of "ETA: Evaluating Then Aligning Safety of Vision Language Models at Inference Time".
- **HatefulMemes/**: Directory structure for the Hateful Memes dataset (data files not included due to size).
- **scripts/**: Evaluation scripts.

## Setup

### Prerequisites

- Python 3.10+
- PyTorch
- CUDA (for GPU support)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/VLM-SAFETY-Baseline-evaluation-BENCHMARK.git
   cd VLM-SAFETY-Baseline-evaluation-BENCHMARK
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (Note: Check `ETA/requirements.txt` for specific dependencies related to ETA).

### Datasets

**Note:** The training datasets (e.g., Hateful Memes, `dataset.zip`) are **not** included in this repository due to their large size.

To run the benchmarks, please download the datasets separately:

- **Hateful Memes Challenge Dataset**: Download from [Facebook AI](https://ai.facebook.com/hatefulmemes) or the [DrivenData competition page](https://www.drivendata.org/competitions/64/hateful-memes/).
  - Extract the data into the `HatefulMemes/data/` directory.
  - Ensure the structure looks like:
    ```
    HatefulMemes/
      data/
        img/
        train.jsonl
        dev.jsonl
        test.jsonl
    ```

## Usage

(Add specific usage instructions here, for example:)

To run the ETA evaluation:

```bash
cd ETA
python eta_quick_use.py --gpu_id 0 --qs "your question here" --image_path "your image path here"
```

## References

- **ETA**: [arXiv:2410.06625](https://arxiv.org/abs/2410.06625)
- **Hateful Memes**: [arXiv:2005.04790](https://arxiv.org/abs/2005.04790)
