# AET

## Title
Auto-Embedding Transformer for Interpretable Few-Shot Fault Diagnosis of Rolling Bearings

## Abstract
Deep learning-based intelligent diagnosis is a popular method to ensure the safe operation of rolling bearings. However, practical diagnostic tasks are often subject to a lack of labeled data, resulting in poor performance in scenarios with insufficient training samples. Moreover, conventional intelligent diagnosis methods suffer from a deficiency in interpretability. In this paper, an auto-embedding Transformer (AET) method is proposed to implement the interpretable few-shot fault diagnosis of rolling bearings. First, an auto-embedding module is developed to improve the embedding quality of signal, which is designed based on a novel asymmetric convolutional encoder-decoder architecture. This module can leverage the merits of unsupervised learning in data mining and allow the Transformer to learn more diagnostic knowledge from limited data. Second, an attention scoring method is proposed that utilizes position-wise attention to quantify the importance of each signal embedding for diagnosis, thereby interpreting the AET method. Experimental results confirm that, even with limited training samples, the AET method outperforms various comparison methods in terms of recognition accuracy and convergence rate. Furthermore, the attention scores assigned to each embedding facilitate the interpretability of the AET method.

## Keywords
Autoencoder, few-shot diagnosis, interpretability, rolling bearings, transformer

## Paper
    @article{wang2023auto,
      title={Auto-embedding transformer for interpretable few-shot fault diagnosis of rolling bearings},
      author={Wang, Gang and Liu, Dongdong and Cui, Lingli},
      journal={IEEE Transactions on Reliability},
      year={2023},
      publisher={IEEE}
    }

    Online: https://ieeexplore.ieee.org/document/10315955
