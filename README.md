# Minimalist Implementation of a GPT 2 with Language Model Head

This repo is a minimalist implementation of a GPT 2 with Language Model Head.
This repo uses the following libraries as the main building blocks:

- [PyTorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/)
- [Transformers](https://huggingface.co/transformers/index.html)
- [PyTorch-NLP](https://pytorchnlp.readthedocs.io/en/latest/index.html)

You can also check this minimalist implementation for text classification: [Minimalist Implementation of a BERT Sentence Classifier](https://github.com/ricardorei/lightning-text-classification).

## Requirements:

This project uses Python 3.7

Create a virtual env with (outside the project folder):

```bash
virtualenv -p python3 gpt2-env
source gpt2-env/bin/activate
```

Install the requirements (inside the project folder):
```bash
pip install -r requirements.txt
```

## Getting Started:

### Train:
```bash
python training.py
```

Available commands:

Training arguments:
```bash
optional arguments:
  --seed                      Training seed.
  --distributed_backend       Supports three options: dp
  --use_16bit                 If true uses 16 bit precision
  --batch_size                Batch size to be used.
  --accumulate_grad_batches   Accumulated gradients runs K small batches of \
                              size N before doing a backwards pass.
  --log_gpu_memory            Uses the output of nvidia-smi to log GPU usage. \
                              Might slow performance.
  --val_percent_check         If you dont want to use the entire dev set, set \
                              how much of the dev set you want to use with this flag.      
```

Early Stopping/Checkpoint arguments:
```bash
optional arguments:
  --metric_mode             If we want to min/max the monitored quantity.
  --min_epochs              Limits training to a minimum number of epochs
  --max_epochs              Limits training to a max number number of epochs
  --save_top_k              The best k models according to the quantity \
                            monitored will be saved.
```

Model arguments:

```bash
optional arguments:
  --learning_rate             Learning rate.
  --train_csv                 Path to the file containing the train data.
  --dev_csv                   Path to the file containing the dev data.
  --test_csv                  Path to the file containing the test data.
  --loader_workers            How many subprocesses to use for data loading.
```

Training command example:
```bash
python training.py \
    --gpus 1 \
    --distributed_backend dp \
    --batch_size 6 \
    --accumulate_grad_batches 2 \
    --loader_workers 4 \
```

You can generate sentences with the model using (you may change the sampling parameters in the `generate` function in `gpt2_lm.py`):
```bash
python interact.py --experiment experiments/lightning_logs/version_{date}
```


### Tensorboard:

Launch tensorboard with:
```bash
tensorboard --logdir="experiments/lightning_logs/"
```

### Code Style:
To make sure all the code follows the same style we use [Black](https://github.com/psf/black).
