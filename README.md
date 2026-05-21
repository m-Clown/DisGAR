DisGAR: Machine Generated Text Detection via Disentangled Generative Artifact Representation

# Dataset 

All data is located in the `dataset` directory. Additionally, we have provided the rewritten texts by DeepSeek-V3.2 in the dataset.

# model 

We use [`google/gemma-2-9b-it`](https://huggingface.co/google/gemma-2-9b-it) as our base model. Please download it to the `model` folder.
The trained model parameters are located in the `checkpoint` folder.

# Usage
### Train

```bash
python DisGAR.py --do_train
```

### eval

```bash
python DisGAR.py --do_eval
```

More detailed setting in the `DisGAR.py`
