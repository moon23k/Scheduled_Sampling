## NMT_GEN
Generally, when training a Transformer-based natural language generation model, the decoder is trained using the Teacher Forcing technique with Masking. Thanks to this, parallel processing becomes possible, enabling efficient learning. However, the relative ease of training comes at the cost of lower learning difficulty. Consequently, during inference when Teacher Forcing is not applied, the model may exhibit performance lower than what was expected during the training sessions.

To enhance the model's generation capabilities, there are approaches like increasing the model size and dataset or utilizing Generative Training on the entire dataset. However, the former requires substantial computational costs, and the latter often suffers from decreased training efficiency.

Therefore, this repository proposes an approach where training with Teacher Forcing is performed initially, and then Generative Training is applied only to address the deficiencies in the trained model.

<br><br>

## Training Strategy

**1. Standard Training** <br>
> This is the standard training approach. Cross-Entropy loss is used, enabling faster learning.

<br>

**2. Alternate Training** <br>
> Alternate Training is a hybrid approach that combines Standard Training and Generative Training. It predominantly employs Standard Training but incorporates Generative Training for specific data instances. This methodology efficiently blends the strengths and weaknesses of Standard Training, known for its speed yet mild learning drive, and Generative Training, recognized for its slower yet robust learning drive.

<br>

**3. Generative** <br>
> Generative Training involves generating sequences directly without Teacher Forcing, comparing the outcomes with actual labels, and updating the model's parameters accordingly. This learning approach employs Bleu Score in the Loss function, ensuring alignment between real-world usage and the learning process. However, it does come with the drawback of being time-consuming during the generation process. To address this, the Decoder utilizes caching while generating.

<br>

**4. Consecutive** <br>
>
 
<br>

**5. Complementary** <br>
> 

<br><br>

## Configurations

<br><br>

## Result

| Train Strategy | Greedy BLEU Score | Beam BLEU Score |
|:---:|:---:|:---:|
| Standard      | - | - |
| Alternate     | - | - |
| Generative    | - | - |
| Consecutive   | - | - |
| Complementary | - | - |

<br><br>

