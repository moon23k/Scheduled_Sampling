## NMT_GEN
In general, in the training of the Transformer-based Seq2Seq model, the decoder is trained through the Teacher Forcing technique with Masking. Thanks to this, parallel processing is possible, which makes whole training process way much efficient. However the difficulty of learning is relativelt low because of the intervention of Teacher Forcing. In the actual translation generation process where Teacher Forcing is not applied, it is common for the performance to be lower than expected in the training session.
Therefore, Generative Training without Teacher Forcing seems necessary for general users who have only small data but want excellent models. Of course, learning this way from scratch is not recommended because it is not only very inefficient, but also it costs a lot. However, if you are using a pre-trained model, it seems pretty doable.
In this repo, experiments are conducted on three methods to figure out the effectiveness of Generative Training in the FineTuning process. The details are as follows.

<br>
<br>

## Strategies

**General Way of FineTuning** <br>
> Fine Tunes Pretrained Encoder-Decoder Model. Although the learning speed is fast and efficient, it shows relatively poor performance in performance evaluation.

<br>

**Scheduled Sampling** <br>
> Half of it is based on teacher forcing, and half is based on generative learning technique.
It is a sort of compromise method to compensate for the inefficiency of Generative and the low performance of Teacher Forcing.

<br>

**Generative Training** <br>
> Apply Generative Training only


<br>
<br>

## Configurations

<br>
<br>

## Result

<br>
<br>

## Reference
[**Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks**](https://arxiv.org/abs/1506.03099)
<br>
