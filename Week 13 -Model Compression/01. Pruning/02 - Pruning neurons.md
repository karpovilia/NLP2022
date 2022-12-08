# Pruning neurons
## Node Importance Functions

[Tianxing He; Yuchen Fan; Yanmin Qian; Tian Tan; Kai Yu, Reshaping deep neural network for fast decoding by node-pruning, 2014](https://ieeexplore.ieee.org/document/6853595)

- After training, a score is calculated for each hidden node using one of these importance functions.
    - Let  $a_i$ be # instances with node output > 0.5 and $d_i$ be # instances with node output ≤ 0.5
    - $Entropy(i)=\dfrac{d_i}{a_i+d_i}log_2\dfrac{d_i}{a_i+d_i}+\dfrac{a_i}{a_i+d_i}log_2\dfrac{a_i}{a_i+d_i}$
    - The intuition is that if one node's outputs are almost identical on all training data, these outputs do not generate variations to later layers and consequently the node may not be useful.
    - Output-weights Norm (onorm): average L1-norm of the weights of its outgoing links
    - Input-weights norm (inorm): average L1-norm of the weights of its incoming links
- All the nodes are sorted by their scores and nodes with less importance values are removed.
- On switchboard speech recognition data, onorm was found to be the best.