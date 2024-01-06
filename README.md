## Scheduled Sampling for Transformers

> In this repository, we directly implement a methodology to adapt Scheduled Sampling, commonly used in RNN models, for use in Transformer models. We empirically verify the performance changes based on the Sampling Ratio. The conventional Scheduled Sampling, which involves generation during the training process, is not suitable for Transformers. However, we conduct experiments using the methodology proposed in the Scheduled Sampling for Transformers paper, which addresses this issue. The model architecture for the experiments is Transformer, and we evaluate natural language generation capabilities on three tasks: machine translation, dialogue generation, and document summarization.

<br><br>



## Setup
The default values for experimental variables are set as follows, and each value can be modified by editing the config.yaml file. <br>

| **Tokenizer Setup**                         | **Model Setup**                   | **Training Setup**                |
| :---                                        | :---                              | :---                              |
| **`Tokenizer Type:`** &hairsp; `BPE`        | **`Input Dimension:`** `15,000`   | **`Epochs:`** `10`                |
| **`Vocab Size:`** &hairsp; `15,000`         | **`Output Dimension:`** `15,000`  | **`Batch Size:`** `32`            |
| **`PAD Idx, Token:`** &hairsp; `0`, `[PAD]` | **`Hidden Dimension:`** `256`     | **`Learning Rate:`** `5e-4`       |
| **`UNK Idx, Token:`** &hairsp; `1`, `[UNK]` | **`PFF Dimension:`** `512`        | **`iters_to_accumulate:`** `4`    |
| **`BOS Idx, Token:`** &hairsp; `2`, `[BOS]` | **`Num Layers:`** `3`             | **`Gradient Clip Max Norm:`** `1` |
| **`EOS Idx, Token:`** &hairsp; `3`, `[EOS]` | **`Num Heads:`** `8`              | **`Apply AMP:`** `True`           |

<br>To shorten the training speed, techiques below are used. <br> 
* **Accumulative Loss Update**, as shown in the table above, accumulative frequency has set 4. <br>
* **Application of AMP**, which enables to convert float32 type vector into float16 type vector.

<br><br>

## Result

### ⚫ Machine Translation
| Sampling Ratio | BLEU | Epoch Time | AVG GPU | Max GPU |
| :---: | :---: | :---: | :---: | :---: |
| 0.0 | 12.99 | 0m 44s | 0.20GB | 0.95GB |
| 0.1 | 2.11 | 1m 00s | 0.20GB | 0.95GB |
| 0.3 | 7.34 | 1m 00s | 0.20GB | 0.95GB |
| 0.5 | 6.94 | 1m 00s | 0.20GB | 0.95GB |

<br>

### ⚫ Dialogue Generation
| Sampling Ratio | ROUGE | Epoch Time | AVG GPU | Max GPU |
| :---: | :---: | :---: | :---: | :---: |
| 0.0 | 2.03 | 0m 43s | 0.20GB | 0.85GB |
| 0.1 | 2.31 | 0m 59s | 0.20GB | 0.85GB |
| 0.3 | 2.34 | 0m 59s | 0.20GB | 0.85GB |
| 0.5 | 0.82 | 0m 59s | 0.20GB | 0.85GB |

<br>

### ⚫ Text Summarization
| Sampling Ratio | ROUGE | Epoch Time | AVG GPU | Max GPU |
| :---: | :---: | :---: | :---: | :---: |
| 0.0 | 7.60 | 2m 44s | 0.21GB | 2.78GB |
| 0.1 | 2.95 | 3m 8s | 0.21GB | 2.78GB |
| 0.3 | 4.10 | 3m 8s | 0.21GB | 2.78GB |
| 0.5 | 5.39 | 3m 8s | 0.21GB | 2.78GB |
<br><br>


## How to Use
```
├── ckpt                    --this dir saves model checkpoints and training logs
├── config.yaml             --this file is for setting up arguments for model, training, and tokenizer 
├── data                    --this dir is for saving Training, Validataion and Test Datasets
├── model                   --this dir contains files for Deep Learning Model
│   ├── __init__.py
│   └── transformer.py
├── module                  --this dir contains a series of modules
│   ├── data.py
│   ├── generate.py
│   ├── __init__.py
│   ├── model.py
│   ├── test.py
│   └── train.py
├── README.md
├── run.py                 --this file includes codes for actual tasks such as training, testing, and inference to carry out the practical aspects of the work
└── setup.py               --this file contains a series of codes for preprocessing data, training a tokenizer, and saving the dataset.
```

<br>

**First clone git repo in your local env**
```
git clone https://github.com/moon23k/Scheduled Sampling
```

<br>

**Download and Process Dataset via setup.py**
```
bash setup.py -task [all, translation, dialogue, summarization]
```

<br>

**Execute the run file on your purpose (search is optional)**
```
python3 run.py -task [translation, dialogue, summarization] \
               -mode [train, test, inference] \
               -sampling_ratio [0.0 ~ 0.5] \
               -search [greedy, beam]
```


<br>

## Reference
* [**Attention is all you need**](https://arxiv.org/abs/1706.03762)
* [**Scheduled Sampling for Transformers**](https://arxiv.org/abs/1906.07651)
<br>
