
- Problems with weight prunning: Irregularity of sparse ,atricies limits the maximum performance and energy efficency achevable on hardware accelerators.
- Block-sparse formats store blocks continiously in memory reducing irregular memory accesses.
- If the maximum magnitude weight of the block is below the current threshold, we set all the weights to zero.
- If the maximum magnitude weight of a block is below the current threshold we set all the weights in that block to zeroz
- For block prunning we need to modify the starting slope to account for the number of elements in a block $(N_b)$
    - Start slope for weight prunning $\Theta_w=\Theta$
    - Start slope for weight prunning $\Theta_b=\Theta_w \times  \sqrt[4]{N_b}$

Bank-Ballanced Sparsity (BBS)
Split tata to banks and remove 50% of the weights within each bank

![[Pasted image 20221208131112.png]]
## References
- Efficient and Effective Sparse LSTM on FPGA with Bank-Balanced Sparsity, [pdf](https://www.microsoft.com/en-us/research/uploads/prod/2019/05/FPGA2019_final.pdf)
- BLOCK-SPARSE RECURRENT NEURAL NETWORKS, ICLR 2018, [pdf](https://openreview.net/pdf?id=HJaDJZ-0W)
- Exploring Sparsity in Recurrent Neural Networks, ICLR 2017, [pdf](https://arxiv.org/pdf/1704.05119.pdf)