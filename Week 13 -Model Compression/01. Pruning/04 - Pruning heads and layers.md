

[Paul Michel, Omer Levy, Graham Neubig, Are Sixteen Heads Really Better than One?, 2019](https://arxiv.org/abs/1905.10650)

![Untitled](Model%20compression/01.%20Pruning/img/Untitled.png)

Majority of attention heads can be removed without deviating too much from the original score. Surprisingly, in some cases removing an attention head results in an increase in BLEU/accuracy.

- Only 8 (out of 96) heads in 6-layer WMT NMT Transformer (16 heads / layer) cause a statistically significant change in performance when they are removed from the model, half of which actually result in a higher BLEU score.

![Untitled](Model%20compression/01.%20Pruning/img/Untitled%201.png)

- For most layers, one head is indeed sufficient at test time, even though the network was trained with 12 (BERT) or 16 (WMT Transformer) attention heads.

![Untitled](Model%20compression/01.%20Pruning/img/Untitled%202.png)

- What if we pruned heads across two or more different layers at the same time?
    - Sort all the attention heads, and prune.
    - Prune up to 20% and 40% of heads from WMT and BERT resp., without incurring any noticeable negative impact.
    
- [**Elena Voita, David Talbot**, Fedor Moiseev, Rico Sennrich, Ivan Titov, Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned 2019](https://arxiv.org/abs/1905.09418)

## Layer prunning

![[Pasted image 20221208134102.png]]
LayerDrop (right) randomly drops layers at training time. At test time, this allows for sub-network selection to any desired depth as the network has been trained to be robust to pruning. In contrast to standard approaches that must re-train a new model from scratch for each model size (left), our method trains only one network from which multiple shallow models can be extracted.

![[Pasted image 20221208134549.png]]

Reducing Transformer Depth on Demand with Structured Dropout, ICLR 2020,  [pdf](https://arxiv.org/pdf/1909.11556.pdf), [openreview](https://openreview.net/forum?id=SylO2yStDr)

## Prunning attention heads and MLP layers
For a finetuned BERT it is possible to find a subnetwork of elements, that achieves performance, comparable with the full model.
86% heads and 57% MLPs survive in less than 7 tasks, which rises concerns about the degree to which BERT relies on task-specific heuristics rather than general linguistic knowledge

![[Pasted image 20221208135153.png]]
The “good” subnetworks: self-attention heads and MLPs that survive pruning. Each cell gives the average number of GLUE tasks in which a given head/MLP survived, and the standard deviation across 5 finetuning initializations.

When BERT Plays the Lottery, All Tickets Are Winning, Anna Rumshisky, 2020  [pdf](https://arxiv.org/pdf/2005.00561.pdf)