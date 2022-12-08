# Knowledge Distillation
Train student to mimic a pre trained, larger teacher

![[Pasted image 20221208141148.png]]
Decoupled Knowledge Distillation, CVPR-2022, [pdf](https://arxiv.org/pdf/2203.08679.pdf), [git](https://github.com/megvii-research/mdistiller)

Distillation methods vary on:
- different types of teacher model
- different types of loss function
    - squared error between the logits of the models
    - KL divergence between the predictive distributions, or
    - some other measure of agreement between the model predictions.
- different choices for what dataset the student model trains on.
    - a large unlabeled dataset
    - a held-out data set, or
    - the original training set.
- Mimic what?
    - Teacher's class probabilities
    - Teacher's feature representation
- Learn from whom?
    - Teacher, teacher assistant, other fellow students
- Adversarial learning

## Distilling the Knowledge in a Neural Network
[Geoffrey **Hinton**, Oriol Vinyals, Jeff Dean Distilling the Knowledge in a Neural Network, 2015](https://arxiv.org/abs/1503.02531)

- The relative probabilities of incorrect answers tell us a lot about how the / teacher model tends to generalize.
- Softmax with temperature $q_i = \frac {exp(z_i/T)}{\sum_j(z_j/T)}$
- Use cross entropy (H) for both the soft and hard part of the loss function. Typically smaller weight for hard part works better.
- When using both hard and soft loss, since the magnitudes of the gradients produced by the soft targets scale as $1/T^2$ it is important to multiply them by $T^2$ when using both hard and soft targets.
    
     $L_{KD}(W_s) = H(y_true, P_s) + \lambda H(P_T, P_S)$
    

![Untitled](Model%20compression/img/Untitled%203.png)

## Sobolev Training for Neural Networks
![[Pasted image 20221208141608.png]]
a) Sobolev Training of order 2. Diamond nodes $m$ and $f$ indicate parameterised functions, where $m$ is trained to approximate $f$. Green nodes receive supervision. Solid lines indicate connections through which error signal from loss $l$, $l_1$, and $l_2$ are backpropagated through to train $m$. 
b) Stochastic Sobolev Training of order 2. If $f$ and $m$ are multivariate functions, the gradients are Jacobian matrices. To avoid computing these high dimensional objects, we can efficiently compute and fit their projections on a random vector $v_j$ sampled from the unit sphere

- Sobolev Training for Neural Networks, NIPS 2017, [pdf](https://dl.dropboxusercontent.com/s/x00bh3855h9xy8j/1706.04859v3.pdf)

## Noisy teachers
Let's include a noise-based regularizer while training the strudent from the teacher.
- Noise is Gaussian noise with mean 0 and std dev $\sigma$ . This noise is added to teachers logits.
- Noisy teachr is more helpful than a noisy student

![[Pasted image 20221208142049.png]]
Training Shallow Students using the proposed Logit Perturbation Method

Deep Model Compression: Distilling Knowledge from Noisy Teachers, 2016, [pdf](https://arxiv.org/pdf/1610.09650.pdf)
# Distillation architectures
- Learning students and teacher together
- Multiple teachers
- Adversarial methods

![Untitled](Model%20compression/img/Untitled%202%201.png)

- [https://huggingface.co/docs/transformers/model_doc/distilbert](https://huggingface.co/docs/transformers/model_doc/distilbert)
- [https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D](https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D)
- [Fine Tuning DistilBERT for MultiLabel Text Classification](https://colab.research.google.com/drive/1VjzdrWW1xnVYyViK-MMzr66QBR8ujVag?usp=sharing) colab

## Multi-step knowledge distillation
or chain distillation
- Student network performance degrades when the gap between student and teacher is large.
- Multi-step knowledge proposes to use intermediate sized network (teacher assistant) to bridge the gap between the student and the teacher
![[Pasted image 20221208142526.png]]
TA fills the gap between student & teacher
- Improved Knowledge Distillation via Teacher Assistant, 2019, [pdf](https://arxiv.org/pdf/1902.03393.pdf)

## Distillation from many teachers
or an ensemble of specialists
- Teacher model could be an ensemble that contains:
	- one generalist model, trained on all the data
	- many specialist models, each of which is trained on data, that is highly enriched in examples from a very confusable subset of the classes
- The softmax of this type of specialist can be made much smaller by combining all of the classes it doesn't care about into single $pad$ class
- Training the student
	- 1. For each instance, we find the $n$ most propable classes acording to the generalist model, ($K$ - known)
	- 2. We take all the specialist models $m$ whose subset of confusable classes $S^m$, has a non-empty intersection with $K$ and call this the active set of specialists $A_k$. Then, we find the full probability distribution $q$ over all the classes that minimizes $KL(gen, q) + \sum{KL(p^m,q)}$

## KD with Adversarial methods
[Knowledge Distillation with Adversarial Samples Supporting Decision Boundary](https://arxiv.org/abs/1805.05532), 2018, [pdf](https://arxiv.org/pdf/1805.05532.pdf)
![[Pasted image 20221208144130.png]]
The concept of knowledge distillation using samples close to the decision boundary. The dots in the figure represent the training sample and the circle around a dot represents the distance to the nearest decision boundary. The samples close to the decision boundary enable more accurate knowledge transfer.
![[Pasted image 20221208144252.png]]
Iterative scheme to find boundary supporting samples for a base sample
