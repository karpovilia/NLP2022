# Model compression
## Recommended articles:

- [An Overview of Model Compression Techniques for Deep Learning in Space](https://medium.com/gsi-technology/an-overview-of-model-compression-techniques-for-deep-learning-in-space-3fd8d4ce84e5)
- [Compressing Large-Scale Transformer-Based Models: A Case Study on BERT](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00413/107387/Compressing-Large-Scale-Transformer-Based-Models-A)
## Why should we compress models?
### Resource constraints
- RAM
- Prediction Latency - FLOPs
- Power dissipation
    - Battery drains fast, phone overheats

- Cloud problems
    - Network delay
    - Power consumption for communication
    - User Privacy
- Persistent Recurrent Neural Networks
    - Cache the weights in one-chip memory such as caches, block RAM, or register files across multiple timestamps
    - High model compression allows significatly large RNNs to be stored in on-chip memory
    - 146x speedup if entire RNN can be stored in registers rather than GPU DRAM

[Greg Diamos, Shubho Sengupta, Bryan Catanzaro, Mike Chrzanowski, Adam Coates, Erich Elsen, Jesse Engel, Awni Hannun, Sanjeev Satheesh Persistent RNNs: Stashing Recurrent Weights On-Chip](http://proceedings.mlr.press/v48/diamos16.html), 2017

[Song Han, Xingyu Liu, Huizi Mao, Jing Pu, Ardavan Pedram, Mark A. Horowitz, William J. Dally, EIE: Efficient Inference Engine on Compressed Deep Neural Network, 2016](https://arxiv.org/abs/1602.01528)

### Training/Finetuning large models is difficult
- Impossible to fine-tune pretrained BERT-large on GPU with 16 GB RAM
- This poses large barrier of entry for communities without the resources to purchase either several large GPUs or time on Google’s TPU’s
- Tuning various configuration parameters needs lots of resources
- By removing unimportant weights from a network, several improvements can be expected: better generalization, fewer training required, and improved speed of learning and/or classification

### Mobile apps limitations
App stores are very sensitive to the size of the binary files
Smaller models also decrease the communication overhead of distributed training of the models

### Is it possible?
- large amount of redundancy among the weights of the neural network
- a small subset of the weights are sufficient to reconstruct the entire network
- We can predict >95% of the weights without any drop in accuracy

[Misha Denil, Babak Shakibi, Laurent Dinh, Marc'Aurelio Ranzato, Nando de Freitas. Predicting Parameters in Deep Learning 2014](https://arxiv.org/abs/1306.0543)


## Model compression techniques

![Untitled](Overview%20of%20popular%20ways%20of%20model%20compression.png)

### Pruning: sparsifying weight matrices
- [Pruning: sparsifying weight matrice](Model%20compression/01.%20Pruning/Index.md)
Methods differ based on what is pruned and the actual logic used to prune

- Given a matrix, one can prune:
    - some weight entries
    - rows/columns
    - bloks
    - heads
    - layers
- Which weights/neurons/blocks/heads to prune?
- Should you prune large networks or build small networks?
- Iterative/gradual pruning? Iterative pruning and densification
- Interplay between pruning and regularization

### Quantization: Reducing bits to represent each weight
[02 - Quantization](02%20-%20Quantization.md)

- Weights can be quantized to two values (binary), three values(ternary) or multiple bits
- Uniform vs non-uniform
- Deterministic vs stochastic
- Loss-aware or not
- Trained vs tuned parameters
- How to quantize word vectors, RNN/LSTM weight matricies, transformers

### Knowledge distillation: Train student to mimic a pretrained, larger teacher
- [Knowledge Distillation: Train student to mimic a pre trained, larger teacher](03%20-%20Knowledge%20Distillation.md)
Distillation methods vary on:

- Different types of teacher model
- Different types of loss function
    - Squared error between the logits of the models
    - KL divergence between the predictive distributions, or
    - Some other measure of agreement between the model predictions.
- Different choices for what dataset the student model trains on.
    - A large unlabeled dataset
    - A held-out data set, or
    - The original training set.
- Mimic what?
    - Teacher's class probabilities
    - Teacher's feature representation
- Learn from whom?
Teacher, teacher assistant, other fellow students

### Parameter sharing
[04 Parameter sharing](04%20Parameter%20sharing.md)

Methods differ depending on:

- Which parameters are shared
- Technique used to share parameters
- Level at which sharing is performed

### Matrix decomposition: Factorize large matrices into multiple smaller components
[05 Matrix decomposition](05%20Matrix%20decomposition.md)
Methods differ		

- in the type of factorization technique
- matrices being factorized
- and the property of weight matrix being exploited

[Neural Architecture Search (NAS)](Neural%20Architecture%20Search%20(NAS).md)
