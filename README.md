## NMT_ACT
Generally, when training a Transformer-based natural language generation model, the decoder is trained using the Teacher Forcing technique with Masking. Thanks to this, parallel processing becomes possible, enabling efficient learning. However, the relative ease of training comes at the cost of lower learning difficulty. Consequently, during inference when Teacher Forcing is not applied, the model may exhibit performance lower than what was expected during the training sessions.

To enhance the model's generation capabilities, there are approaches like increasing the model size and dataset or utilizing Generative Training on the entire dataset. However, the former requires substantial computational costs, and the latter often suffers from decreased training efficiency.

Therefore, this repository proposes an approach where training with Teacher Forcing is performed initially, and then Generative Training is applied only to address the deficiencies in the trained model.

<br><br>

## Training Strategy

**1. Train Initialized Transformer Model with Teacher Forcing** <br>
> This is the standard training approach. Cross-Entropy loss is used, enabling faster learning.

<br>

**2. Figure out Weak Points** <br>
> Using BLEU evaluation metric to measure the actual model performance, we calculate the average BLEU score and identify data points scoring below the average. We search for challenging data and samples with semantic similarity within the Training Dataset.

<br>

**3. Complementary Training** <br>
> Conduct additional training based on a supplementary dataset. 
  For comparison purposes, two different training methods are used: General training and Generative Training.

<br><br>

## Configurations

<br><br>

## Result

| Model | Greedy BLEU Score | Beam BLEU Score |
|---|---|---|
| Base Model |-|-|
| Complemented Model |-|-|
| Complemented Generative Model |-|-|

<br><br>
