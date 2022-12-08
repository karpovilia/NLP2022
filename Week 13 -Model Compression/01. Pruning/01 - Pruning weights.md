## Optimal Brain Damage (OBD)
- [Yann LeCun, John Denker, Sara Solla, Optimal Brain Damage (NIPS 1989)](https://papers.nips.cc/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf)
- [Оптимальное прореживание нейронных сетей](http://www.machinelearning.ru/wiki/index.php?title=OBD)

Oldest work.
How to define saliency of a weight, besides simply using its magnitude?
Using change in the objective function, caused by deleting/changing the parameter.

Drawbacks
- Computationally prohibitive as second derivative computations are expensive.
- Cross terms in the Hessian matrix are ignored.

## Optimal Brain Surgeon
[Babak Hassibi, David Stork, Second order derivatives for network pruning: Optimal Brain Surgeon (NIPS 1992)](https://proceedings.neurips.cc/paper/1992/file/303ed4c69846ab36c2904d3ba8573050-Paper.pdf)

- Use information from all second order derivatives of the error function to perform network pruning.
- Also, unlike other methods (like OBD or magnitude pruning), OBS does not demand (typically slow) retraining after the pruning of a weight.
- Computationally prohibitive as second derivative computations are expensive.

## Deep Compression
[Song Han, Huizi Mao, William J. Dally, Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149)

A more computationally feasible method for pruning connections and relearning weights based solely on the magnitude of the original weights

![Untitled](Untitled%204.png)

## Pruning encoder-decoder models
[Abigail See, Minh-Thang Luong, Christopher D. Manning, Compression of Neural Machine Translation Models via Pruning](https://arxiv.org/abs/1606.09274)

![Untitled](Untitled%201%202.png)

### Pruning Schemes
- How do we distribute the pruning over the different weight classes of our model?
    - Class-blind: Take all parameters, sort them by magnitude and prune the x% with smallest magnitude, regardless of weight class.
    - Class-uniform: Within each class, sort the weights by magnitude and prune the x% with smallest magnitude.
    - Class-distribution: For each class c, weights with magnitude less than $\lambda$ are pruned.
- Retraining the sparse pruned network helps. Retrain with smaller learning rate (LR), simpler LR schedule (halve every 2 epochs), and train for fewer epochs.
- Class-blind pruning outperforms both other schemes.

## Iterative Pruning
[Song Han, Jeff Pool, John Tran, William J. Dally Learning both Weights and Connections for Efficient Neural Networks, 2015](https://arxiv.org/abs/1506.02626)

![Untitled](Untitled%202%202.png)

- Regularization (L1/L2) while training.
- Fixed threshold is used for magnitude pruning in every iteration.
- Dropout Ratio Adjustment
    - During retraining, the dropout ratio must be adjusted to account for the change in model capacity.
    - As pruning already reduced model capacity, the retraining dropout ratio should be smaller.

## Iterative Magnitude Pruning for Transformers
[Robin Cheong, Robel Daniel, Compressing Transformers with Pruning and Quantization, 2019](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/reports/custom/15763707.pdf)
[ERNIE github](https://github.com/robeld/ERNIE)

- For starting proportion X% and ending proportion Y%, our iterative magnitude pruning procedure pruned X% of each of the pre-trained Transformer layers, began re-training, and pruned (Y -X)/9 % of each of the layers every 1001 iterations.

![Untitled](Untitled%203%201.png)

- By the 10,000th iteration, we reached Y% pruning of the model iteratively.
- Do not factor in word embeddings in compression rate.

## Sparse BERT with improved pruning
[Fu-Ming Guo, Sijia Liu, Finlay S. Mungall, Xue Lin, Yanzhi Wang, Reweighted Proximal Pruning for Large-Scale Language Representation, 2019](https://arxiv.org/abs/1909.12486)

- Two problems with pruning
    - The larger weight $w_i$, is penalized more heavily than smaller weight $w_i$ in $l_1$ regularization, which violates the original intention of weight pruning, “removing the unimportant connections”.
    - Direct optimization of a regularization penalty term causes divergence from the original loss function and has negative effect on the effectiveness of gradient-based update.
- Solution using reweiehted proximal pruning (which depends on proximal operators)
    - Decouples the goals of high sparsity from minimizing loss.
- NIP: Progressive/gradual pruning without regularizers.

![Untitled](Untitled%204%201.png)

- Even when 90% of weights are pruned, next sentence prediction accuracy keeps above 95% in RPP. 80% pruning for most GLUE tasks and 41% for SQuAD 1.1 at 0 degradation for BERT BASE.

## Should we prune large networks or build small dense networks?

[Michael Zhu, Suyog Gupta, To prune, or not to prune: exploring the efficacy of pruning for model compression, 2017](https://arxiv.org/abs/1710.01878)

![Untitled](Untitled%205.png)

![Untitled](Untitled%206.png)

- Pruning involves extra processing plus sparse matrices need special handling - can we avoid it
- Large-sparse models consistently outperform small-dense models and achieve up to 10x reduction in number of non-zero parameters with minimal loss in accuracy.
- Models: stacked LSTMs for language modeling, and seq2seq models for NMT.