# DLP Benchmark for Indonesian PII Detection

A comprehensive benchmark comparing three Data Loss Prevention (DLP) paradigms for detecting Indonesian Personally Identifiable Information (PII) in shadow AI traffic.

## Overview

This benchmark evaluates three DLP approaches:
- **Pattern-Matching (Regex, CPU)**: Traditional rule-based detection
- **Discriminative (Fine-tuned BERT, GPU)**: Machine learning-based NER
- **Reasoning (Llama-3 8B, GPU)**: Large language model with structured output

### PII Types Detected
- **NIK**: 16-digit Indonesian National Identity Number
- **Phone**: Indonesian phone numbers (various formats)
- **Credit Card**: 16-digit credit card numbers
- **Bank Account**: 10-16 digit bank account numbers
- **Email**: Email addresses

### Writing Styles Evaluated
- **Formal**: Professional Indonesian text
- **Code-Mixed**: Indonesian-English mixed text
- **Slang/Noisy**: Informal text with typos and noise

## Project Structure

```
shadow-ai-sensitive-data/
├── main.py                      # Main orchestration script
├── config.py                    # Configuration constants
├── utils.py                     # Common utilities
├── data_generation.py           # Synthetic data generation
├── pattern_matching.py          # Regex-based detection
├── bert_detection.py            # BERT NER detection
├── llm_detection.py             # LLM reasoning detection
├── evaluation.py                # Metrics and statistical tests
├── reporting.py                 # Report generation
├── train_bert_pii_ner.py        # BERT fine-tuning script
└── README.md                    # This file
```

## Module Descriptions

### main.py
Main orchestration script that coordinates the complete benchmark workflow:
- Initializes random seeds for reproducibility
- Generates synthetic data
- Runs all three paradigms
- Computes metrics and statistical tests
- Generates reports
- Saves outputs

### config.py
Centralized configuration containing:
- Model names and paths
- Hardware specifications
- Prompt templates
- Regex patterns
- Evaluation parameters
- Random seed

### utils.py
Common utility functions:
- `print_banner()`: Formatted section headers
- `normalize_val()`: Value normalization
- `is_hit()`: PII matching logic

### data_generation.py
Synthetic data generation:
- `generate_nik()`: Generate valid NIK numbers
- `generate_phone()`: Generate Indonesian phone numbers
- `generate_credit_card()`: Generate credit card numbers
- `generate_bank_account()`: Generate bank account numbers
- `generate_email()`: Generate email addresses
- `inject_noise()`: Add typos and noise
- `wrap_pii_in_prompt()`: Embed PII in contextual prompts
- `build_ground_truth_dataframe()`: Create ground truth dataset
- `build_prompt_dataset()`: Generate prompts for all styles
- `get_eval_sample()`: Stratified sampling for evaluation

### pattern_matching.py
Regex-based detection:
- Regex patterns for all PII types
- `regex_detect()`: Apply regex patterns
- `run_regex_benchmark()`: Run complete regex evaluation

### bert_detection.py
BERT NER detection:
- `load_indobert_ner()`: Load fine-tuned BERT model
- `bert_detect()`: Extract entities using BIO tagging
- `run_bert_benchmark()`: Run complete BERT evaluation

### llm_detection.py
LLM reasoning detection:
- `load_llama3_4bit()`: Load quantized Llama-3 model
- `llm_detect()`: Extract PII using structured prompts
- `run_llm_benchmark()`: Run complete LLM evaluation

### evaluation.py
Metrics and statistical analysis:
- `compute_detection_metrics()`: Calculate Precision, Recall, F1, TP, FP, FN
- `bootstrap_recall_ci()`: Bootstrap confidence intervals (n=1000)
- `paired_permutation_test()`: Statistical significance testing (n=10,000)
- `compute_robustness_delta()`: Style robustness analysis
- `compute_latency_summary()`: Latency statistics

### reporting.py
Report generation and display:
- `generate_full_report()`: Comprehensive metrics report
- `generate_significance_report()`: Statistical significance results
- `generate_robustness_report()`: Robustness across styles
- `generate_latency_report()`: Latency comparison
- `display_results()`: Pretty-print all reports

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for BERT and LLM)
- 16GB+ RAM
- 24GB+ VRAM for LLM (or use 4-bit quantization)
- HuggingFace account and API token

### Step 1: Clone the Repository
```bash
git clone https://github.com/gredss/shadow-ai-sensitive-data.git
cd shadow-ai-sensitive-data
```

### Step 2: Set Up Environment Variables
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your HuggingFace token
# Get your token from: https://huggingface.co/settings/tokens
nano .env  # or use your preferred editor
```

Your `.env` file should contain:
```
HF_TOKEN=your_actual_huggingface_token_here
```

**Important**: Never commit the `.env` file to version control. It's already in `.gitignore`.

### Step 3: Install Dependencies
```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch with CUDA support (for GPU)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

For CPU-only installation:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "from config import HF_TOKEN; print('HF_TOKEN loaded successfully')"
```

## Usage

### Quick Start
Run the complete benchmark:
```bash
python main.py
```

### Command-Line Options
```bash
# Skip BERT evaluation
python main.py --skip-bert

# Skip LLM evaluation
python main.py --skip-llm

# Don't save output files
python main.py --no-save

# Combine options
python main.py --skip-llm --no-save
```

### Training Custom BERT Model
See `BERT_TRAINING_README.md` for detailed instructions:
```bash
python train_bert_pii_ner.py
```

## Output Files

The benchmark generates the following CSV files:
- `ground_truth.csv`: Ground truth PII data
- `prompt_dataset.csv`: Generated prompts with PII
- `eval_sample.csv`: Stratified evaluation sample
- `results_regex.csv`: Regex detection results
- `results_bert.csv`: BERT detection results
- `results_llm.csv`: LLM detection results
- `benchmark_report.csv`: Comprehensive metrics
- `significance_report.csv`: Statistical significance tests
- `robustness_report.csv`: Style robustness analysis
- `latency_report.csv`: Latency comparison

## Evaluation Metrics

### Detection Metrics
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **True Positives (TP)**: Correctly detected PII
- **False Positives (FP)**: Incorrectly detected non-PII
- **False Negatives (FN)**: Missed PII

### Statistical Tests
- **Bootstrap Confidence Intervals**: 1,000 resamples, 95% CI
- **Paired Permutation Test**: 10,000 shuffles, two-tailed

### Robustness Analysis
- **Delta (Δ)**: Recall difference between formal and noisy styles
- **Lower Δ**: More robust to style variations

## Key Features

### Statistical Rigor
- Bootstrap confidence intervals for uncertainty quantification
- Paired permutation tests for significance testing
- Stratified sampling ensures fair comparison

### Comprehensive Coverage
- All 5 Indonesian PII types
- 3 writing styles (formal, code-mixed, slang)
- 500 samples per style (1,500 total)

### Modular Design
- Clean separation of concerns
- Easy to extend with new paradigms
- Reusable components

### Reproducibility
- Fixed random seed (SEED=42)
- Identical evaluation sample for all paradigms
- Deterministic data generation

## Performance Expectations

### Typical Results (RTX 4090)
- **Regex**: ~0.1ms per sample, 85-90% recall
- **BERT**: ~50ms per sample, 90-95% recall
- **LLM**: ~500ms per sample, 95-98% recall

### Trade-offs
- **Regex**: Fast, deterministic, but brittle
- **BERT**: Balanced speed/accuracy, requires training
- **LLM**: Highest accuracy, but slowest and most expensive

## Customization

### Adding New PII Types
1. Add generation function in `data_generation.py`
2. Add regex pattern in `pattern_matching.py`
3. Update BERT training data in `train_bert_pii_ner.py`
4. Update LLM prompt in `config.py`

### Adding New Paradigms
1. Create new detection module (e.g., `transformer_detection.py`)
2. Implement detection function following existing pattern
3. Add benchmark runner function
4. Update `main.py` to include new paradigm
5. Update `reporting.py` to include in reports

### Modifying Evaluation
- Change `EVAL_SAMPLE_SIZE` in `config.py`
- Adjust bootstrap/permutation iterations in `evaluation.py`
- Modify style distributions in `data_generation.py`

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size in BERT/LLM detection
- Use 4-bit quantization for LLM (already enabled)
- Skip LLM evaluation: `python main.py --skip-llm`

### Slow Execution
- Use GPU for BERT and LLM
- Reduce `EVAL_SAMPLE_SIZE` in `config.py`
- Skip expensive paradigms during development

### Import Errors
- Ensure all dependencies are installed
- Check Python version (3.8+)
- Verify CUDA installation for GPU support

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.

## Acknowledgments

- IndoBERT model by IndoNLP
- Llama-3 by Meta AI
- Transformers library by Hugging Face
