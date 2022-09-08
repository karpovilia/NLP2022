```python
!pip install transformers
!pip install tensorboard
!pip install torch
```

    Looking in indexes: https://pypi.org/simple, https://packagecloud.io/github/git-lfs/pypi/simple
    Collecting transformers
      Downloading transformers-4.21.3-py3-none-any.whl (4.7 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m4.7/4.7 MB[0m [31m15.1 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting filelock
      Using cached filelock-3.8.0-py3-none-any.whl (10 kB)
    Collecting regex!=2019.12.17
      Using cached regex-2022.8.17-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (765 kB)
    Requirement already satisfied: packaging>=20.0 in /home/user/juputer_env/lib/python3.9/site-packages (from transformers) (21.3)
    Collecting tokenizers!=0.11.3,<0.13,>=0.11.1
      Downloading tokenizers-0.12.1-cp39-cp39-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (6.6 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m6.6/6.6 MB[0m [31m9.8 MB/s[0m eta [36m0:00:00[0m:00:01[0m00:01[0mm
    [?25hCollecting tqdm>=4.27
      Using cached tqdm-4.64.1-py2.py3-none-any.whl (78 kB)
    Collecting requests
      Using cached requests-2.28.1-py3-none-any.whl (62 kB)
    Collecting huggingface-hub<1.0,>=0.1.0
      Downloading huggingface_hub-0.9.1-py3-none-any.whl (120 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m120.7/120.7 kB[0m [31m4.5 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting numpy>=1.17
      Using cached numpy-1.23.2-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (17.1 MB)
    Collecting pyyaml>=5.1
      Downloading PyYAML-6.0-cp39-cp39-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl (661 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m661.8/661.8 kB[0m [31m29.5 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting typing-extensions>=3.7.4.3
      Downloading typing_extensions-4.3.0-py3-none-any.whl (25 kB)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /home/user/juputer_env/lib/python3.9/site-packages (from packaging>=20.0->transformers) (3.0.9)
    Collecting idna<4,>=2.5
      Using cached idna-3.3-py3-none-any.whl (61 kB)
    Collecting urllib3<1.27,>=1.21.1
      Downloading urllib3-1.26.12-py2.py3-none-any.whl (140 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m140.4/140.4 kB[0m [31m3.4 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hCollecting charset-normalizer<3,>=2
      Downloading charset_normalizer-2.1.1-py3-none-any.whl (39 kB)
    Collecting certifi>=2017.4.17
      Downloading certifi-2022.6.15-py3-none-any.whl (160 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m160.2/160.2 kB[0m [31m4.6 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: tokenizers, urllib3, typing-extensions, tqdm, regex, pyyaml, numpy, idna, filelock, charset-normalizer, certifi, requests, huggingface-hub, transformers
    Successfully installed certifi-2022.6.15 charset-normalizer-2.1.1 filelock-3.8.0 huggingface-hub-0.9.1 idna-3.3 numpy-1.23.2 pyyaml-6.0 regex-2022.8.17 requests-2.28.1 tokenizers-0.12.1 tqdm-4.64.1 transformers-4.21.3 typing-extensions-4.3.0 urllib3-1.26.12
    Looking in indexes: https://pypi.org/simple, https://packagecloud.io/github/git-lfs/pypi/simple
    Collecting torch
      Downloading torch-1.12.1-cp39-cp39-manylinux1_x86_64.whl (776.4 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m776.4/776.4 MB[0m [31m2.2 MB/s[0m eta [36m0:00:00[0m00:01[0m00:03[0m
    [?25hRequirement already satisfied: typing-extensions in /home/user/juputer_env/lib/python3.9/site-packages (from torch) (4.3.0)
    Installing collected packages: torch
    Successfully installed torch-1.12.1



```python
import numpy as np
import torch
```


```python
!git clone https://github.com/priya-dwivedi/Deep-Learning/
```

    Cloning into 'Deep-Learning'...
    remote: Enumerating objects: 3461, done.[K
    remote: Counting objects: 100% (60/60), done.[K
    remote: Compressing objects: 100% (43/43), done.[K
    remote: Total 3461 (delta 43), reused 21 (delta 17), pack-reused 3401[K
    Receiving objects: 100% (3461/3461), 424.80 MiB | 11.10 MiB/s, done.
    Resolving deltas: 100% (940/940), done.
    Checking out files: 100% (3469/3469), done.



```python
!pwd
```

    /home/user/ki/Deep-Learning/GPT2-HarryPotter-Training/examples



```python
!cd Deep-Learning/GPT2-HarryPotter-Training/examples
```

    /bin/bash: line 0: cd: Deep-Learning/GPT2-HarryPotter-Training/examples: No such file or directory



```python
!python run_lm_finetuning.py \
    --output_dir=output \
    --model_type=gpt2 \
    --model_name_or_path=gpt2-medium \
    --do_train \
    --train_data_file='input_data/train_harry.txt' \
    --do_eval \
    --eval_data_file='input_data/val_harry.txt'\
    --overwrite_output_dir\
    --block_size=200\
    --per_gpu_train_batch_size=1\
    --save_steps 5000\
    --num_train_epochs=2
```

    /home/user/juputer_env/lib/python3.9/site-packages/torch/cuda/__init__.py:83: UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 10010). Please update your GPU driver by downloading and installing a new version from the URL: http://www.nvidia.com/Download/index.aspx Alternatively, go to: https://pytorch.org to install a PyTorch version that has been compiled with your version of the CUDA driver. (Triggered internally at  ../c10/cuda/CUDAFunctions.cpp:109.)
      return torch._C._cuda_getDeviceCount() > 0
    09/08/2022 14:20:22 - WARNING - __main__ -   Process rank: -1, device: cpu, n_gpu: 0, distributed training: False, 16-bits training: False
    Downloading config.json: 100%|██████████████████| 718/718 [00:00<00:00, 537kB/s]
    Downloading vocab.json: 100%|██████████████| 0.99M/0.99M [00:00<00:00, 1.40MB/s]
    Downloading merges.txt: 100%|█████████████████| 446k/446k [00:00<00:00, 736kB/s]
    Downloading pytorch_model.bin: 100%|███████| 1.42G/1.42G [02:05<00:00, 12.1MB/s]
    09/08/2022 14:22:40 - INFO - __main__ -   Training/evaluation parameters Namespace(train_data_file='input_data/train_harry.txt', output_dir='output', eval_data_file='input_data/val_harry.txt', model_type='gpt2', model_name_or_path='gpt2-medium', mlm=False, mlm_probability=0.15, config_name='', tokenizer_name='', cache_dir='', block_size=200, do_train=True, do_eval=True, evaluate_during_training=False, do_lower_case=False, per_gpu_train_batch_size=1, per_gpu_eval_batch_size=4, gradient_accumulation_steps=1, learning_rate=5e-05, weight_decay=0.0, adam_epsilon=1e-08, max_grad_norm=1.0, num_train_epochs=2.0, max_steps=-1, warmup_steps=0, logging_steps=100, save_steps=5000, save_total_limit=2, eval_all_checkpoints=False, no_cuda=False, overwrite_output_dir=True, overwrite_cache=False, seed=42, fp16=False, fp16_opt_level='O1', local_rank=-1, server_ip='', server_port='', n_gpu=0, device=device(type='cpu'))
    09/08/2022 14:22:40 - INFO - __main__ -   Loading features from cached file input_data/gpt2-medium_cached_lm_200_train_harry.txt
    /home/user/juputer_env/lib/python3.9/site-packages/transformers/optimization.py:306: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
      warnings.warn(
    09/08/2022 14:22:40 - INFO - __main__ -   ***** Running training *****
    09/08/2022 14:22:40 - INFO - __main__ -     Num examples = 1184
    09/08/2022 14:22:40 - INFO - __main__ -     Num Epochs = 2
    09/08/2022 14:22:40 - INFO - __main__ -     Instantaneous batch size per GPU = 1
    09/08/2022 14:22:40 - INFO - __main__ -     Total train batch size (w. parallel, distributed & accumulation) = 1
    09/08/2022 14:22:40 - INFO - __main__ -     Gradient Accumulation steps = 1
    09/08/2022 14:22:40 - INFO - __main__ -     Total optimization steps = 2368
    Epoch:   0%|                                              | 0/2 [00:00<?, ?it/s]
    Iteration:   0%|                                       | 0/1184 [00:00<?, ?it/s][A
    Iteration:   0%|                               | 1/1184 [00:01<38:06,  1.93s/it][A
    Iteration:   0%|                               | 2/1184 [00:05<55:13,  2.80s/it][A
    Iteration:   0%|                             | 3/1184 [00:08<1:00:39,  3.08s/it][A
    Iteration:   0%|                             | 4/1184 [00:12<1:03:22,  3.22s/it][A
    Iteration:   0%|                             | 5/1184 [00:15<1:01:48,  3.15s/it][A
    Iteration:   1%|▏                              | 6/1184 [00:17<57:11,  2.91s/it][A
    Iteration:   1%|▏                              | 7/1184 [00:20<53:25,  2.72s/it][A
    Iteration:   1%|▏                              | 8/1184 [00:22<51:42,  2.64s/it][A
    Iteration:   1%|▏                              | 9/1184 [00:24<49:12,  2.51s/it][A
    Iteration:   1%|▎                             | 10/1184 [00:26<46:33,  2.38s/it][A
    Iteration:   1%|▎                             | 11/1184 [00:28<45:20,  2.32s/it][A
    Iteration:   1%|▎                             | 12/1184 [00:31<43:52,  2.25s/it][A
    Iteration:   1%|▎                             | 13/1184 [00:32<39:53,  2.04s/it][A
    Iteration:   1%|▎                             | 14/1184 [00:34<37:04,  1.90s/it][A
    Iteration:   1%|▍                             | 15/1184 [00:36<38:27,  1.97s/it][A
    Iteration:   1%|▍                             | 16/1184 [00:38<40:02,  2.06s/it][A
    Iteration:   1%|▍                             | 17/1184 [00:40<41:04,  2.11s/it][A
    Iteration:   2%|▍                             | 18/1184 [00:42<41:20,  2.13s/it][A
    Iteration:   2%|▍                             | 19/1184 [00:45<41:43,  2.15s/it][A
    Iteration:   2%|▌                             | 20/1184 [00:47<41:38,  2.15s/it][A
    Iteration:   2%|▌                             | 21/1184 [00:49<42:03,  2.17s/it][A
    Iteration:   2%|▌                             | 22/1184 [00:51<43:26,  2.24s/it][A
    Iteration:   2%|▌                             | 23/1184 [00:54<43:57,  2.27s/it][A
    Iteration:   2%|▌                             | 24/1184 [00:56<44:16,  2.29s/it][A
    Iteration:   2%|▋                             | 25/1184 [00:58<44:26,  2.30s/it][A
    Iteration:   2%|▋                             | 26/1184 [01:01<43:47,  2.27s/it][A
    Iteration:   2%|▋                             | 27/1184 [01:03<43:39,  2.26s/it][A
    Iteration:   2%|▋                             | 28/1184 [01:05<43:22,  2.25s/it][A
    Iteration:   2%|▋                             | 29/1184 [01:07<43:45,  2.27s/it][A
    Iteration:   3%|▊                             | 30/1184 [01:10<42:46,  2.22s/it][A
    Iteration:   3%|▊                             | 31/1184 [01:12<42:02,  2.19s/it][A
    Iteration:   3%|▊                             | 32/1184 [01:14<41:26,  2.16s/it][A
    Iteration:   3%|▊                             | 33/1184 [01:16<41:53,  2.18s/it][A
    Iteration:   3%|▊                             | 34/1184 [01:18<42:05,  2.20s/it][A
    Iteration:   3%|▉                             | 35/1184 [01:21<42:47,  2.23s/it][A
    Iteration:   3%|▉                             | 36/1184 [01:23<43:26,  2.27s/it][A
    Iteration:   3%|▉                             | 37/1184 [01:25<43:27,  2.27s/it][A
    Iteration:   3%|▉                             | 38/1184 [01:28<44:35,  2.33s/it][A
    Iteration:   3%|▉                             | 39/1184 [01:30<44:25,  2.33s/it][A
    Iteration:   3%|█                             | 40/1184 [01:33<46:03,  2.42s/it][A
    Iteration:   3%|█                             | 41/1184 [01:35<46:12,  2.43s/it][A
    Iteration:   4%|█                             | 42/1184 [01:37<44:55,  2.36s/it][A
    Iteration:   4%|█                             | 43/1184 [01:39<43:31,  2.29s/it][A
    Iteration:   4%|█                             | 44/1184 [01:42<42:51,  2.26s/it][A
    Iteration:   4%|█▏                            | 45/1184 [01:44<42:23,  2.23s/it][A
    Iteration:   4%|█▏                            | 46/1184 [01:46<41:59,  2.21s/it][A
    Iteration:   4%|█▏                            | 47/1184 [01:48<41:45,  2.20s/it][A
    Iteration:   4%|█▏                            | 48/1184 [01:50<41:38,  2.20s/it][A
    Iteration:   4%|█▏                            | 49/1184 [01:53<42:01,  2.22s/it][A
    Iteration:   4%|█▎                            | 50/1184 [01:55<42:38,  2.26s/it][A
    Iteration:   4%|█▎                            | 51/1184 [01:57<43:07,  2.28s/it][A
    Iteration:   4%|█▎                            | 52/1184 [02:00<43:14,  2.29s/it][A
    Iteration:   4%|█▎                            | 53/1184 [02:02<43:28,  2.31s/it][A
    Iteration:   5%|█▎                            | 54/1184 [02:04<43:17,  2.30s/it][A
    Iteration:   5%|█▍                            | 55/1184 [02:06<43:11,  2.30s/it][A
    Iteration:   5%|█▍                            | 56/1184 [02:09<43:10,  2.30s/it][A
    Iteration:   5%|█▍                            | 57/1184 [02:11<43:33,  2.32s/it][A
    Iteration:   5%|█▍                            | 58/1184 [02:13<43:23,  2.31s/it][A
    Iteration:   5%|█▍                            | 59/1184 [02:16<43:00,  2.29s/it][A
    Iteration:   5%|█▌                            | 60/1184 [02:18<42:46,  2.28s/it][A
    Iteration:   5%|█▌                            | 61/1184 [02:21<45:18,  2.42s/it][A
    Iteration:   5%|█▌                            | 62/1184 [02:23<44:09,  2.36s/it][A
    Iteration:   5%|█▌                            | 63/1184 [02:25<43:24,  2.32s/it][A
    Iteration:   5%|█▌                            | 64/1184 [02:27<42:46,  2.29s/it][A
    Iteration:   5%|█▋                            | 65/1184 [02:30<42:18,  2.27s/it][A
    Iteration:   6%|█▋                            | 66/1184 [02:32<42:00,  2.25s/it][A
    Iteration:   6%|█▋                            | 67/1184 [02:34<41:45,  2.24s/it][A
    Iteration:   6%|█▋                            | 68/1184 [02:36<41:51,  2.25s/it][A
    Iteration:   6%|█▋                            | 69/1184 [02:38<41:41,  2.24s/it][A
    Iteration:   6%|█▊                            | 70/1184 [02:41<41:30,  2.24s/it][A
    Iteration:   6%|█▊                            | 71/1184 [02:43<41:27,  2.23s/it][A
    Iteration:   6%|█▊                            | 72/1184 [02:45<41:19,  2.23s/it][A
    Iteration:   6%|█▊                            | 73/1184 [02:47<41:12,  2.23s/it][A
    Iteration:   6%|█▉                            | 74/1184 [02:50<40:50,  2.21s/it][A
    Iteration:   6%|█▉                            | 75/1184 [02:52<40:50,  2.21s/it][A
    Iteration:   6%|█▉                            | 76/1184 [02:54<40:49,  2.21s/it][A
    Iteration:   7%|█▉                            | 77/1184 [02:56<41:09,  2.23s/it][A
    Iteration:   7%|█▉                            | 78/1184 [02:58<40:59,  2.22s/it][A
    Iteration:   7%|██                            | 79/1184 [03:01<40:47,  2.22s/it][A
    Iteration:   7%|██                            | 80/1184 [03:03<40:47,  2.22s/it][A
    Iteration:   7%|██                            | 81/1184 [03:05<40:22,  2.20s/it][A
    Iteration:   7%|██                            | 82/1184 [03:07<40:43,  2.22s/it][A
    Iteration:   7%|██                            | 83/1184 [03:10<41:26,  2.26s/it][A
    Iteration:   7%|██▏                           | 84/1184 [03:12<41:16,  2.25s/it][A
    Iteration:   7%|██▏                           | 85/1184 [03:14<41:24,  2.26s/it][A
    Iteration:   7%|██▏                           | 86/1184 [03:16<41:01,  2.24s/it][A
    Iteration:   7%|██▏                           | 87/1184 [03:19<40:40,  2.22s/it][A
    Iteration:   7%|██▏                           | 88/1184 [03:21<40:53,  2.24s/it][A
    Iteration:   8%|██▎                           | 89/1184 [03:23<41:33,  2.28s/it][A
    Iteration:   8%|██▎                           | 90/1184 [03:25<40:58,  2.25s/it][A
    Iteration:   8%|██▎                           | 91/1184 [03:28<40:38,  2.23s/it][A
    Iteration:   8%|██▎                           | 92/1184 [03:30<40:33,  2.23s/it][A
    Iteration:   8%|██▎                           | 93/1184 [03:32<40:20,  2.22s/it][A
    Iteration:   8%|██▍                           | 94/1184 [03:34<40:24,  2.22s/it][A
    Iteration:   8%|██▍                           | 95/1184 [03:36<40:09,  2.21s/it][A
    Iteration:   8%|██▍                           | 96/1184 [03:39<39:58,  2.20s/it][A
    Iteration:   8%|██▍                           | 97/1184 [03:41<39:57,  2.21s/it][A
    Iteration:   8%|██▍                           | 98/1184 [03:43<40:01,  2.21s/it][A
    Iteration:   8%|██▌                           | 99/1184 [03:45<40:07,  2.22s/it][A/home/user/juputer_env/lib/python3.9/site-packages/torch/optim/lr_scheduler.py:249: UserWarning: To get the last learning rate computed by the scheduler, please use `get_last_lr()`.
      warnings.warn("To get the last learning rate computed by the scheduler, "
    
    Iteration:   8%|██▍                          | 100/1184 [03:47<39:46,  2.20s/it][A
    Iteration:   9%|██▍                          | 101/1184 [03:50<39:31,  2.19s/it][A
    Iteration:   9%|██▍                          | 102/1184 [03:52<39:47,  2.21s/it][A
    Iteration:   9%|██▌                          | 103/1184 [03:54<40:03,  2.22s/it][A
    Iteration:   9%|██▌                          | 104/1184 [03:56<40:09,  2.23s/it][A
    Iteration:   9%|██▌                          | 105/1184 [03:59<40:19,  2.24s/it][A
    Iteration:   9%|██▌                          | 106/1184 [04:01<41:07,  2.29s/it][A
    Iteration:   9%|██▌                          | 107/1184 [04:03<40:51,  2.28s/it][A
    Iteration:   9%|██▋                          | 108/1184 [04:05<40:39,  2.27s/it][A
    Iteration:   9%|██▋                          | 109/1184 [04:08<40:51,  2.28s/it][A
    Iteration:   9%|██▋                          | 110/1184 [04:10<40:50,  2.28s/it][A
    Iteration:   9%|██▋                          | 111/1184 [04:12<40:38,  2.27s/it][A
    Iteration:   9%|██▋                          | 112/1184 [04:15<40:30,  2.27s/it][A
    Iteration:  10%|██▊                          | 113/1184 [04:17<40:51,  2.29s/it][A
    Iteration:  10%|██▊                          | 114/1184 [04:19<40:32,  2.27s/it][A
    Iteration:  10%|██▊                          | 115/1184 [04:21<40:34,  2.28s/it][A
    Iteration:  10%|██▊                          | 116/1184 [04:24<40:29,  2.27s/it][A
    Iteration:  10%|██▊                          | 117/1184 [04:26<40:46,  2.29s/it][A
    Iteration:  10%|██▉                          | 118/1184 [04:28<40:54,  2.30s/it][A
    Iteration:  10%|██▉                          | 119/1184 [04:31<40:44,  2.30s/it][A
    Iteration:  10%|██▉                          | 120/1184 [04:33<40:16,  2.27s/it][A
    Iteration:  10%|██▉                          | 121/1184 [04:35<40:07,  2.26s/it][A
    Iteration:  10%|██▉                          | 122/1184 [04:37<40:03,  2.26s/it][A
    Iteration:  10%|███                          | 123/1184 [04:40<39:48,  2.25s/it][A
    Iteration:  10%|███                          | 124/1184 [04:42<39:18,  2.23s/it][A
    Iteration:  11%|███                          | 125/1184 [04:44<38:36,  2.19s/it][A
    Iteration:  11%|███                          | 126/1184 [04:46<38:04,  2.16s/it][A
    Iteration:  11%|███                          | 127/1184 [04:48<38:21,  2.18s/it][A
    Iteration:  11%|███▏                         | 128/1184 [04:50<38:52,  2.21s/it][A
    Iteration:  11%|███▏                         | 129/1184 [04:53<38:54,  2.21s/it][A
    Iteration:  11%|███▏                         | 130/1184 [04:55<38:54,  2.21s/it][A
    Iteration:  11%|███▏                         | 131/1184 [04:57<38:55,  2.22s/it][A
    Iteration:  11%|███▏                         | 132/1184 [04:59<38:37,  2.20s/it][A
    Iteration:  11%|███▎                         | 133/1184 [05:02<39:04,  2.23s/it][A
    Iteration:  11%|███▎                         | 134/1184 [05:04<39:08,  2.24s/it][A
    Iteration:  11%|███▎                         | 135/1184 [05:06<38:45,  2.22s/it][A
    Iteration:  11%|███▎                         | 136/1184 [05:08<38:27,  2.20s/it][A
    Iteration:  12%|███▎                         | 137/1184 [05:10<38:18,  2.19s/it][A
    Iteration:  12%|███▍                         | 138/1184 [05:13<38:34,  2.21s/it][A
    Iteration:  12%|███▍                         | 139/1184 [05:15<38:13,  2.19s/it][A
    Iteration:  12%|███▍                         | 140/1184 [05:17<37:58,  2.18s/it][A
    Iteration:  12%|███▍                         | 141/1184 [05:19<38:08,  2.19s/it][A
    Iteration:  12%|███▍                         | 142/1184 [05:21<38:02,  2.19s/it][A
    Iteration:  12%|███▌                         | 143/1184 [05:24<38:44,  2.23s/it][A
    Iteration:  12%|███▌                         | 144/1184 [05:26<39:04,  2.25s/it][A
    Iteration:  12%|███▌                         | 145/1184 [05:28<38:49,  2.24s/it][A
    Iteration:  12%|███▌                         | 146/1184 [05:30<39:00,  2.25s/it][A
    Iteration:  12%|███▌                         | 147/1184 [05:33<38:30,  2.23s/it][A
    Iteration:  12%|███▋                         | 148/1184 [05:35<38:14,  2.22s/it][A
    Iteration:  13%|███▋                         | 149/1184 [05:38<41:49,  2.42s/it][A
    Iteration:  13%|███▋                         | 150/1184 [05:40<40:49,  2.37s/it][A
    Iteration:  13%|███▋                         | 151/1184 [05:42<39:48,  2.31s/it][A
    Iteration:  13%|███▋                         | 152/1184 [05:44<38:58,  2.27s/it][A
    Iteration:  13%|███▋                         | 153/1184 [05:46<38:27,  2.24s/it][A
    Iteration:  13%|███▊                         | 154/1184 [05:49<38:10,  2.22s/it][A
    Iteration:  13%|███▊                         | 155/1184 [05:51<37:55,  2.21s/it][A
    Iteration:  13%|███▊                         | 156/1184 [05:53<37:42,  2.20s/it][A
    Iteration:  13%|███▊                         | 157/1184 [05:55<37:43,  2.20s/it][A
    Iteration:  13%|███▊                         | 158/1184 [05:57<37:23,  2.19s/it][A
    Iteration:  13%|███▉                         | 159/1184 [06:00<37:12,  2.18s/it][A
    Iteration:  14%|███▉                         | 160/1184 [06:02<37:10,  2.18s/it][A
    Iteration:  14%|███▉                         | 161/1184 [06:04<37:39,  2.21s/it][A
    Iteration:  14%|███▉                         | 162/1184 [06:06<37:47,  2.22s/it][A
    Iteration:  14%|███▉                         | 163/1184 [06:08<37:22,  2.20s/it][A
    Iteration:  14%|████                         | 164/1184 [06:11<37:02,  2.18s/it][A
    Iteration:  14%|████                         | 165/1184 [06:13<36:57,  2.18s/it][A
    Iteration:  14%|████                         | 166/1184 [06:15<36:57,  2.18s/it][A
    Iteration:  14%|████                         | 167/1184 [06:17<36:58,  2.18s/it][A
    Iteration:  14%|████                         | 168/1184 [06:19<37:30,  2.21s/it][A
    Iteration:  14%|████▏                        | 169/1184 [06:22<37:46,  2.23s/it][A
    Iteration:  14%|████▏                        | 170/1184 [06:24<38:19,  2.27s/it][A
    Iteration:  14%|████▏                        | 171/1184 [06:26<37:41,  2.23s/it][A
    Iteration:  15%|████▏                        | 172/1184 [06:28<37:29,  2.22s/it][A
    Iteration:  15%|████▏                        | 173/1184 [06:31<37:56,  2.25s/it][A
    Iteration:  15%|████▎                        | 174/1184 [06:33<39:14,  2.33s/it][A
    Iteration:  15%|████▎                        | 175/1184 [06:35<38:16,  2.28s/it][A
    Iteration:  15%|████▎                        | 176/1184 [06:38<38:00,  2.26s/it][A
    Iteration:  15%|████▎                        | 177/1184 [06:40<41:11,  2.45s/it][A
    Iteration:  15%|████▎                        | 178/1184 [06:43<39:42,  2.37s/it][A
    Iteration:  15%|████▍                        | 179/1184 [06:45<38:47,  2.32s/it][A
    Iteration:  15%|████▍                        | 180/1184 [06:47<38:19,  2.29s/it][A
    Iteration:  15%|████▍                        | 181/1184 [06:49<38:00,  2.27s/it][A
    Iteration:  15%|████▍                        | 182/1184 [06:51<37:44,  2.26s/it][A
    Iteration:  15%|████▍                        | 183/1184 [06:54<37:32,  2.25s/it][A
    Iteration:  16%|████▌                        | 184/1184 [06:56<37:21,  2.24s/it][A
    Iteration:  16%|████▌                        | 185/1184 [06:58<36:52,  2.21s/it][A
    Iteration:  16%|████▌                        | 186/1184 [07:00<36:42,  2.21s/it][A
    Iteration:  16%|████▌                        | 187/1184 [07:02<36:35,  2.20s/it][A
    Iteration:  16%|████▌                        | 188/1184 [07:05<36:40,  2.21s/it][A
    Iteration:  16%|████▋                        | 189/1184 [07:07<36:53,  2.22s/it][A
    Iteration:  16%|████▋                        | 190/1184 [07:09<36:58,  2.23s/it][A
    Iteration:  16%|████▋                        | 191/1184 [07:11<37:12,  2.25s/it][A
    Iteration:  16%|████▋                        | 192/1184 [07:14<37:05,  2.24s/it][A
    Iteration:  16%|████▋                        | 193/1184 [07:16<37:01,  2.24s/it][A
    Iteration:  16%|████▊                        | 194/1184 [07:18<36:58,  2.24s/it][A
    Iteration:  16%|████▊                        | 195/1184 [07:20<36:47,  2.23s/it][A
    Iteration:  17%|████▊                        | 196/1184 [07:23<36:30,  2.22s/it][A
    Iteration:  17%|████▊                        | 197/1184 [07:25<36:13,  2.20s/it][A
    Iteration:  17%|████▊                        | 198/1184 [07:27<36:27,  2.22s/it][A
    Iteration:  17%|████▊                        | 199/1184 [07:29<36:22,  2.22s/it][A
    Iteration:  17%|████▉                        | 200/1184 [07:31<36:06,  2.20s/it][A
    Iteration:  17%|████▉                        | 201/1184 [07:34<36:33,  2.23s/it][A
    Iteration:  17%|████▉                        | 202/1184 [07:36<36:18,  2.22s/it][A
    Iteration:  17%|████▉                        | 203/1184 [07:38<36:04,  2.21s/it][A
    Iteration:  17%|████▉                        | 204/1184 [07:40<36:01,  2.21s/it][A
    Iteration:  17%|█████                        | 205/1184 [07:43<36:13,  2.22s/it][A
    Iteration:  17%|█████                        | 206/1184 [07:45<36:18,  2.23s/it][A
    Iteration:  17%|█████                        | 207/1184 [07:47<36:16,  2.23s/it][A
    Iteration:  18%|█████                        | 208/1184 [07:49<36:13,  2.23s/it][A
    Iteration:  18%|█████                        | 209/1184 [07:51<36:11,  2.23s/it][A
    Iteration:  18%|█████▏                       | 210/1184 [07:54<35:54,  2.21s/it][A
    Iteration:  18%|█████▏                       | 211/1184 [07:56<35:39,  2.20s/it][A
    Iteration:  18%|█████▏                       | 212/1184 [07:58<35:26,  2.19s/it][A
    Iteration:  18%|█████▏                       | 213/1184 [08:00<35:35,  2.20s/it][A
    Iteration:  18%|█████▏                       | 214/1184 [08:03<36:11,  2.24s/it][A
    Iteration:  18%|█████▎                       | 215/1184 [08:05<36:08,  2.24s/it][A
    Iteration:  18%|█████▎                       | 216/1184 [08:07<36:51,  2.28s/it][A
    Iteration:  18%|█████▎                       | 217/1184 [08:09<37:04,  2.30s/it][A
    Iteration:  18%|█████▎                       | 218/1184 [08:12<36:22,  2.26s/it][A
    Iteration:  18%|█████▎                       | 219/1184 [08:14<36:18,  2.26s/it][A
    Iteration:  19%|█████▍                       | 220/1184 [08:16<35:48,  2.23s/it][A
    Iteration:  19%|█████▍                       | 221/1184 [08:18<35:47,  2.23s/it][A
    Iteration:  19%|█████▍                       | 222/1184 [08:21<35:46,  2.23s/it][A
    Iteration:  19%|█████▍                       | 223/1184 [08:23<35:53,  2.24s/it][A
    Iteration:  19%|█████▍                       | 224/1184 [08:25<37:35,  2.35s/it][A
    Iteration:  19%|█████▌                       | 225/1184 [08:28<37:15,  2.33s/it][A
    Iteration:  19%|█████▌                       | 226/1184 [08:30<36:56,  2.31s/it][A
    Iteration:  19%|█████▌                       | 227/1184 [08:32<36:28,  2.29s/it][A
    Iteration:  19%|█████▌                       | 228/1184 [08:34<35:52,  2.25s/it][A
    Iteration:  19%|█████▌                       | 229/1184 [08:37<35:47,  2.25s/it][A
    Iteration:  19%|█████▋                       | 230/1184 [08:39<36:02,  2.27s/it][A
    Iteration:  20%|█████▋                       | 231/1184 [08:41<36:00,  2.27s/it][A
    Iteration:  20%|█████▋                       | 232/1184 [08:43<35:47,  2.26s/it][A
    Iteration:  20%|█████▋                       | 233/1184 [08:46<35:54,  2.27s/it][A
    Iteration:  20%|█████▋                       | 234/1184 [08:48<35:36,  2.25s/it][A
    Iteration:  20%|█████▊                       | 235/1184 [08:50<35:33,  2.25s/it][A
    Iteration:  20%|█████▊                       | 236/1184 [08:52<35:32,  2.25s/it][A
    Iteration:  20%|█████▊                       | 237/1184 [08:55<35:33,  2.25s/it][A
    Iteration:  20%|█████▊                       | 238/1184 [08:57<35:26,  2.25s/it][A
    Iteration:  20%|█████▊                       | 239/1184 [08:59<35:14,  2.24s/it][A
    Iteration:  20%|█████▉                       | 240/1184 [09:01<34:48,  2.21s/it][A
    Iteration:  20%|█████▉                       | 241/1184 [09:03<34:33,  2.20s/it][A
    Iteration:  20%|█████▉                       | 242/1184 [09:06<34:34,  2.20s/it][A
    Iteration:  21%|█████▉                       | 243/1184 [09:08<34:15,  2.18s/it][A
    Iteration:  21%|█████▉                       | 244/1184 [09:10<34:08,  2.18s/it][A
    Iteration:  21%|██████                       | 245/1184 [09:12<34:13,  2.19s/it][A
    Iteration:  21%|██████                       | 246/1184 [09:14<34:50,  2.23s/it][A
    Iteration:  21%|██████                       | 247/1184 [09:17<35:19,  2.26s/it][A
    Iteration:  21%|██████                       | 248/1184 [09:19<35:40,  2.29s/it][A
    Iteration:  21%|██████                       | 249/1184 [09:21<35:04,  2.25s/it][A
    Iteration:  21%|██████                       | 250/1184 [09:23<34:39,  2.23s/it][A
    Iteration:  21%|██████▏                      | 251/1184 [09:26<34:21,  2.21s/it][A
    Iteration:  21%|██████▏                      | 252/1184 [09:28<34:07,  2.20s/it][A
    Iteration:  21%|██████▏                      | 253/1184 [09:30<34:03,  2.20s/it][A
    Iteration:  21%|██████▏                      | 254/1184 [09:32<33:55,  2.19s/it][A
    Iteration:  22%|██████▏                      | 255/1184 [09:34<33:51,  2.19s/it][A
    Iteration:  22%|██████▎                      | 256/1184 [09:37<33:43,  2.18s/it][A
    Iteration:  22%|██████▎                      | 257/1184 [09:39<33:41,  2.18s/it][A
    Iteration:  22%|██████▎                      | 258/1184 [09:41<33:36,  2.18s/it][A
    Iteration:  22%|██████▎                      | 259/1184 [09:43<33:44,  2.19s/it][A
    Iteration:  22%|██████▎                      | 260/1184 [09:45<33:53,  2.20s/it][A
    Iteration:  22%|██████▍                      | 261/1184 [09:48<33:56,  2.21s/it][A
    Iteration:  22%|██████▍                      | 262/1184 [09:50<36:44,  2.39s/it][A
    Iteration:  22%|██████▍                      | 263/1184 [09:53<35:54,  2.34s/it][A
    Iteration:  22%|██████▍                      | 264/1184 [09:55<35:13,  2.30s/it][A
    Iteration:  22%|██████▍                      | 265/1184 [09:57<35:25,  2.31s/it][A
    Iteration:  22%|██████▌                      | 266/1184 [09:59<34:55,  2.28s/it][A
    Iteration:  23%|██████▌                      | 267/1184 [10:02<34:43,  2.27s/it][A
    Iteration:  23%|██████▌                      | 268/1184 [10:04<34:27,  2.26s/it][A
    Iteration:  23%|██████▌                      | 269/1184 [10:06<33:56,  2.23s/it][A
    Iteration:  23%|██████▌                      | 270/1184 [10:08<34:05,  2.24s/it][A
    Iteration:  23%|██████▋                      | 271/1184 [10:10<33:50,  2.22s/it][A
    Iteration:  23%|██████▋                      | 272/1184 [10:13<33:33,  2.21s/it][A
    Iteration:  23%|██████▋                      | 273/1184 [10:15<33:19,  2.19s/it][A
    Iteration:  23%|██████▋                      | 274/1184 [10:17<33:25,  2.20s/it][A
    Iteration:  23%|██████▋                      | 275/1184 [10:19<33:36,  2.22s/it][A
    Iteration:  23%|██████▊                      | 276/1184 [10:22<33:53,  2.24s/it][A
    Iteration:  23%|██████▊                      | 277/1184 [10:24<33:51,  2.24s/it][A
    Iteration:  23%|██████▊                      | 278/1184 [10:26<32:13,  2.13s/it][A
    Iteration:  24%|██████▊                      | 279/1184 [10:28<32:40,  2.17s/it][A
    Iteration:  24%|██████▊                      | 280/1184 [10:30<32:47,  2.18s/it][A
    Iteration:  24%|██████▉                      | 281/1184 [10:32<33:17,  2.21s/it][A
    Iteration:  24%|██████▉                      | 282/1184 [10:35<33:15,  2.21s/it][A
    Iteration:  24%|██████▉                      | 283/1184 [10:37<33:02,  2.20s/it][A
    Iteration:  24%|██████▉                      | 284/1184 [10:39<33:10,  2.21s/it][A
    Iteration:  24%|██████▉                      | 285/1184 [10:41<33:05,  2.21s/it][A
    Iteration:  24%|███████                      | 286/1184 [10:43<33:09,  2.22s/it][A
    Iteration:  24%|███████                      | 287/1184 [10:46<33:28,  2.24s/it][A
    Iteration:  24%|███████                      | 288/1184 [10:48<33:24,  2.24s/it][A
    Iteration:  24%|███████                      | 289/1184 [10:50<33:17,  2.23s/it][A
    Iteration:  24%|███████                      | 290/1184 [10:52<33:12,  2.23s/it][A
    Iteration:  25%|███████▏                     | 291/1184 [10:55<33:02,  2.22s/it][A
    Iteration:  25%|███████▏                     | 292/1184 [10:57<33:04,  2.23s/it][A
    Iteration:  25%|███████▏                     | 293/1184 [10:59<33:04,  2.23s/it][A
    Iteration:  25%|███████▏                     | 294/1184 [11:01<33:03,  2.23s/it][A
    Iteration:  25%|███████▏                     | 295/1184 [11:04<33:21,  2.25s/it][A
    Iteration:  25%|███████▎                     | 296/1184 [11:06<32:46,  2.21s/it][A
    Iteration:  25%|███████▎                     | 297/1184 [11:08<32:30,  2.20s/it][A
    Iteration:  25%|███████▎                     | 298/1184 [11:10<32:37,  2.21s/it][A
    Iteration:  25%|███████▎                     | 299/1184 [11:12<32:39,  2.21s/it][A
    Iteration:  25%|███████▎                     | 300/1184 [11:15<33:13,  2.26s/it][A
    Iteration:  25%|███████▎                     | 301/1184 [11:17<33:05,  2.25s/it][A
    Iteration:  26%|███████▍                     | 302/1184 [11:19<33:34,  2.28s/it][A
    Iteration:  26%|███████▍                     | 303/1184 [11:22<33:42,  2.30s/it][A
    Iteration:  26%|███████▍                     | 304/1184 [11:24<33:30,  2.28s/it][A
    Iteration:  26%|███████▍                     | 305/1184 [11:26<32:48,  2.24s/it][A
    Iteration:  26%|███████▍                     | 306/1184 [11:28<32:21,  2.21s/it][A
    Iteration:  26%|███████▌                     | 307/1184 [11:30<31:58,  2.19s/it][A
    Iteration:  26%|███████▌                     | 308/1184 [11:33<32:06,  2.20s/it][A
    Iteration:  26%|███████▌                     | 309/1184 [11:35<31:48,  2.18s/it][A
    Iteration:  26%|███████▌                     | 310/1184 [11:37<32:36,  2.24s/it][A
    Iteration:  26%|███████▌                     | 311/1184 [11:39<32:29,  2.23s/it][A
    Iteration:  26%|███████▋                     | 312/1184 [11:41<32:13,  2.22s/it][A
    Iteration:  26%|███████▋                     | 313/1184 [11:44<32:11,  2.22s/it][A
    Iteration:  27%|███████▋                     | 314/1184 [11:46<32:05,  2.21s/it][A
    Iteration:  27%|███████▋                     | 315/1184 [11:48<32:07,  2.22s/it][A
    Iteration:  27%|███████▋                     | 316/1184 [11:50<32:00,  2.21s/it][A
    Iteration:  27%|███████▊                     | 317/1184 [11:53<31:56,  2.21s/it][A
    Iteration:  27%|███████▊                     | 318/1184 [11:55<32:24,  2.25s/it][A
    Iteration:  27%|███████▊                     | 319/1184 [11:57<32:49,  2.28s/it][A
    Iteration:  27%|███████▊                     | 320/1184 [11:59<32:50,  2.28s/it][A
    Iteration:  27%|███████▊                     | 321/1184 [12:02<32:36,  2.27s/it][A
    Iteration:  27%|███████▉                     | 322/1184 [12:04<32:57,  2.29s/it][A
    Iteration:  27%|███████▉                     | 323/1184 [12:06<32:42,  2.28s/it][A
    Iteration:  27%|███████▉                     | 324/1184 [12:09<33:05,  2.31s/it][A
    Iteration:  27%|███████▉                     | 325/1184 [12:11<33:08,  2.31s/it][A
    Iteration:  28%|███████▉                     | 326/1184 [12:13<32:23,  2.27s/it][A
    Iteration:  28%|████████                     | 327/1184 [12:15<32:33,  2.28s/it][A
    Iteration:  28%|████████                     | 328/1184 [12:18<32:37,  2.29s/it][A
    Iteration:  28%|████████                     | 329/1184 [12:20<32:35,  2.29s/it][A
    Iteration:  28%|████████                     | 330/1184 [12:22<32:10,  2.26s/it][A
    Iteration:  28%|████████                     | 331/1184 [12:25<32:19,  2.27s/it][A
    Iteration:  28%|████████▏                    | 332/1184 [12:27<31:59,  2.25s/it][A
    Iteration:  28%|████████▏                    | 333/1184 [12:29<31:45,  2.24s/it][A
    Iteration:  28%|████████▏                    | 334/1184 [12:31<31:26,  2.22s/it][A
    Iteration:  28%|████████▏                    | 335/1184 [12:33<31:23,  2.22s/it][A
    Iteration:  28%|████████▏                    | 336/1184 [12:36<31:24,  2.22s/it][A
    Iteration:  28%|████████▎                    | 337/1184 [12:38<31:25,  2.23s/it][A
    Iteration:  29%|████████▎                    | 338/1184 [12:40<31:50,  2.26s/it][A
    Iteration:  29%|████████▎                    | 339/1184 [12:42<31:30,  2.24s/it][A
    Iteration:  29%|████████▎                    | 340/1184 [12:45<31:26,  2.24s/it][A
    Iteration:  29%|████████▎                    | 341/1184 [12:47<31:19,  2.23s/it][A
    Iteration:  29%|████████▍                    | 342/1184 [12:49<30:56,  2.20s/it][A
    Iteration:  29%|████████▍                    | 343/1184 [12:51<30:41,  2.19s/it][A
    Iteration:  29%|████████▍                    | 344/1184 [12:53<30:32,  2.18s/it][A
    Iteration:  29%|████████▍                    | 345/1184 [12:55<30:17,  2.17s/it][A
    Iteration:  29%|████████▍                    | 346/1184 [12:58<30:36,  2.19s/it][A
    Iteration:  29%|████████▍                    | 347/1184 [13:00<33:07,  2.37s/it][A
    Iteration:  29%|████████▌                    | 348/1184 [13:03<32:56,  2.36s/it][A
    Iteration:  29%|████████▌                    | 349/1184 [13:05<31:56,  2.30s/it][A
    Iteration:  30%|████████▌                    | 350/1184 [13:07<31:11,  2.24s/it][A
    Iteration:  30%|████████▌                    | 351/1184 [13:09<30:56,  2.23s/it][A
    Iteration:  30%|████████▌                    | 352/1184 [13:11<30:54,  2.23s/it][A
    Iteration:  30%|████████▋                    | 353/1184 [13:14<30:51,  2.23s/it][A
    Iteration:  30%|████████▋                    | 354/1184 [13:16<30:50,  2.23s/it][A
    Iteration:  30%|████████▋                    | 355/1184 [13:18<30:51,  2.23s/it][A
    Iteration:  30%|████████▋                    | 356/1184 [13:20<30:50,  2.24s/it][A
    Iteration:  30%|████████▋                    | 357/1184 [13:23<30:49,  2.24s/it][A
    Iteration:  30%|████████▊                    | 358/1184 [13:25<31:10,  2.26s/it][A
    Iteration:  30%|████████▊                    | 359/1184 [13:27<31:07,  2.26s/it][A
    Iteration:  30%|████████▊                    | 360/1184 [13:30<31:02,  2.26s/it][A
    Iteration:  30%|████████▊                    | 361/1184 [13:32<30:58,  2.26s/it][A
    Iteration:  31%|████████▊                    | 362/1184 [13:34<31:00,  2.26s/it][A
    Iteration:  31%|████████▉                    | 363/1184 [13:36<30:46,  2.25s/it][A
    Iteration:  31%|████████▉                    | 364/1184 [13:38<30:16,  2.22s/it][A
    Iteration:  31%|████████▉                    | 365/1184 [13:41<30:06,  2.21s/it][A
    Iteration:  31%|████████▉                    | 366/1184 [13:43<30:03,  2.20s/it][A
    Iteration:  31%|████████▉                    | 367/1184 [13:45<30:21,  2.23s/it][A
    Iteration:  31%|█████████                    | 368/1184 [13:47<30:41,  2.26s/it][A
    Iteration:  31%|█████████                    | 369/1184 [13:50<31:08,  2.29s/it][A
    Iteration:  31%|█████████                    | 370/1184 [13:52<31:16,  2.31s/it][A
    Iteration:  31%|█████████                    | 371/1184 [13:54<30:43,  2.27s/it][A
    Iteration:  31%|█████████                    | 372/1184 [13:57<30:46,  2.27s/it][A
    Iteration:  32%|█████████▏                   | 373/1184 [13:59<30:13,  2.24s/it][A
    Iteration:  32%|█████████▏                   | 374/1184 [14:01<30:03,  2.23s/it][A
    Iteration:  32%|█████████▏                   | 375/1184 [14:03<30:09,  2.24s/it][A
    Iteration:  32%|█████████▏                   | 376/1184 [14:05<29:57,  2.23s/it][A
    Iteration:  32%|█████████▏                   | 377/1184 [14:08<29:38,  2.20s/it][A
    Iteration:  32%|█████████▎                   | 378/1184 [14:10<29:20,  2.18s/it][A
    Iteration:  32%|█████████▎                   | 379/1184 [14:12<29:02,  2.17s/it][A
    Iteration:  32%|█████████▎                   | 380/1184 [14:14<29:00,  2.16s/it][A
    Iteration:  32%|█████████▎                   | 381/1184 [14:16<28:55,  2.16s/it][A
    Iteration:  32%|█████████▎                   | 382/1184 [14:18<28:49,  2.16s/it][A
    Iteration:  32%|█████████▍                   | 383/1184 [14:20<28:47,  2.16s/it][A
    Iteration:  32%|█████████▍                   | 384/1184 [14:23<28:41,  2.15s/it][A
    Iteration:  33%|█████████▍                   | 385/1184 [14:25<28:48,  2.16s/it][A
    Iteration:  33%|█████████▍                   | 386/1184 [14:27<29:07,  2.19s/it][A
    Iteration:  33%|█████████▍                   | 387/1184 [14:29<29:39,  2.23s/it][A
    Iteration:  33%|█████████▌                   | 388/1184 [14:32<29:46,  2.24s/it][A
    Iteration:  33%|█████████▌                   | 389/1184 [14:34<29:45,  2.25s/it][A
    Iteration:  33%|█████████▌                   | 390/1184 [14:36<29:31,  2.23s/it][A
    Iteration:  33%|█████████▌                   | 391/1184 [14:38<29:16,  2.22s/it][A
    Iteration:  33%|█████████▌                   | 392/1184 [14:41<29:32,  2.24s/it][A
    Iteration:  33%|█████████▋                   | 393/1184 [14:43<29:21,  2.23s/it][A
    Iteration:  33%|█████████▋                   | 394/1184 [14:45<29:29,  2.24s/it][A
    Iteration:  33%|█████████▋                   | 395/1184 [14:47<29:43,  2.26s/it][A
    Iteration:  33%|█████████▋                   | 396/1184 [14:50<29:42,  2.26s/it][A
    Iteration:  34%|█████████▋                   | 397/1184 [14:52<29:41,  2.26s/it][A
    Iteration:  34%|█████████▋                   | 398/1184 [14:54<29:22,  2.24s/it][A
    Iteration:  34%|█████████▊                   | 399/1184 [14:56<29:31,  2.26s/it][A
    Iteration:  34%|█████████▊                   | 400/1184 [14:59<29:34,  2.26s/it][A
    Iteration:  34%|█████████▊                   | 401/1184 [15:01<29:18,  2.25s/it][A
    Iteration:  34%|█████████▊                   | 402/1184 [15:03<29:11,  2.24s/it][A
    Iteration:  34%|█████████▊                   | 403/1184 [15:05<29:23,  2.26s/it][A
    Iteration:  34%|█████████▉                   | 404/1184 [15:08<29:14,  2.25s/it][A
    Iteration:  34%|█████████▉                   | 405/1184 [15:10<29:27,  2.27s/it][A
    Iteration:  34%|█████████▉                   | 406/1184 [15:12<29:37,  2.28s/it][A
    Iteration:  34%|█████████▉                   | 407/1184 [15:14<29:01,  2.24s/it][A
    Iteration:  34%|█████████▉                   | 408/1184 [15:17<29:10,  2.26s/it][A
    Iteration:  35%|██████████                   | 409/1184 [15:19<29:02,  2.25s/it][A
    Iteration:  35%|██████████                   | 410/1184 [15:21<28:43,  2.23s/it][A
    Iteration:  35%|██████████                   | 411/1184 [15:23<28:44,  2.23s/it][A
    Iteration:  35%|██████████                   | 412/1184 [15:25<28:32,  2.22s/it][A
    Iteration:  35%|██████████                   | 413/1184 [15:28<28:38,  2.23s/it][A
    Iteration:  35%|██████████▏                  | 414/1184 [15:30<28:31,  2.22s/it][A
    Iteration:  35%|██████████▏                  | 415/1184 [15:32<28:44,  2.24s/it][A
    Iteration:  35%|██████████▏                  | 416/1184 [15:34<28:23,  2.22s/it][A
    Iteration:  35%|██████████▏                  | 417/1184 [15:37<28:24,  2.22s/it][A
    Iteration:  35%|██████████▏                  | 418/1184 [15:39<28:07,  2.20s/it][A
    Iteration:  35%|██████████▎                  | 419/1184 [15:41<27:55,  2.19s/it][A
    Iteration:  35%|██████████▎                  | 420/1184 [15:43<27:57,  2.20s/it][A
    Iteration:  36%|██████████▎                  | 421/1184 [15:45<28:13,  2.22s/it][A
    Iteration:  36%|██████████▎                  | 422/1184 [15:48<28:13,  2.22s/it][A
    Iteration:  36%|██████████▎                  | 423/1184 [15:50<28:00,  2.21s/it][A
    Iteration:  36%|██████████▍                  | 424/1184 [15:52<28:04,  2.22s/it][A
    Iteration:  36%|██████████▍                  | 425/1184 [15:54<28:08,  2.23s/it][A
    Iteration:  36%|██████████▍                  | 426/1184 [15:56<28:00,  2.22s/it][A
    Iteration:  36%|██████████▍                  | 427/1184 [15:59<27:51,  2.21s/it][A
    Iteration:  36%|██████████▍                  | 428/1184 [16:01<27:51,  2.21s/it][A
    Iteration:  36%|██████████▌                  | 429/1184 [16:03<27:39,  2.20s/it][A
    Iteration:  36%|██████████▌                  | 430/1184 [16:05<27:26,  2.18s/it][A
    Iteration:  36%|██████████▌                  | 431/1184 [16:07<27:13,  2.17s/it][A
    Iteration:  36%|██████████▌                  | 432/1184 [16:10<27:24,  2.19s/it][A
    Iteration:  37%|██████████▌                  | 433/1184 [16:12<30:02,  2.40s/it][A
    Iteration:  37%|██████████▋                  | 434/1184 [16:15<29:07,  2.33s/it][A
    Iteration:  37%|██████████▋                  | 435/1184 [16:17<28:52,  2.31s/it][A
    Iteration:  37%|██████████▋                  | 436/1184 [16:19<28:34,  2.29s/it][A
    Iteration:  37%|██████████▋                  | 437/1184 [16:21<28:17,  2.27s/it][A
    Iteration:  37%|██████████▋                  | 438/1184 [16:23<27:41,  2.23s/it][A
    Iteration:  37%|██████████▊                  | 439/1184 [16:26<27:44,  2.23s/it][A
    Iteration:  37%|██████████▊                  | 440/1184 [16:28<27:51,  2.25s/it][A
    Iteration:  37%|██████████▊                  | 441/1184 [16:30<27:42,  2.24s/it][A
    Iteration:  37%|██████████▊                  | 442/1184 [16:32<27:35,  2.23s/it][A
    Iteration:  37%|██████████▊                  | 443/1184 [16:35<27:34,  2.23s/it][A
    Iteration:  38%|██████████▉                  | 444/1184 [16:37<27:35,  2.24s/it][A
    Iteration:  38%|██████████▉                  | 445/1184 [16:39<27:39,  2.25s/it][A
    Iteration:  38%|██████████▉                  | 446/1184 [16:41<27:14,  2.22s/it][A
    Iteration:  38%|██████████▉                  | 447/1184 [16:43<26:56,  2.19s/it][A
    Iteration:  38%|██████████▉                  | 448/1184 [16:46<26:54,  2.19s/it][A
    Iteration:  38%|██████████▉                  | 449/1184 [16:48<26:47,  2.19s/it][A
    Iteration:  38%|███████████                  | 450/1184 [16:50<26:51,  2.20s/it][A
    Iteration:  38%|███████████                  | 451/1184 [16:52<26:53,  2.20s/it][A
    Iteration:  38%|███████████                  | 452/1184 [16:55<26:56,  2.21s/it][A
    Iteration:  38%|███████████                  | 453/1184 [16:57<27:05,  2.22s/it][A
    Iteration:  38%|███████████                  | 454/1184 [16:59<28:19,  2.33s/it][A
    Iteration:  38%|███████████▏                 | 455/1184 [17:02<27:51,  2.29s/it][A
    Iteration:  39%|███████████▏                 | 456/1184 [17:04<27:26,  2.26s/it][A
    Iteration:  39%|███████████▏                 | 457/1184 [17:06<27:19,  2.26s/it][A
    Iteration:  39%|███████████▏                 | 458/1184 [17:08<27:48,  2.30s/it][A
    Iteration:  39%|███████████▏                 | 459/1184 [17:11<27:32,  2.28s/it][A
    Iteration:  39%|███████████▎                 | 460/1184 [17:13<27:23,  2.27s/it][A
    Iteration:  39%|███████████▎                 | 461/1184 [17:15<27:18,  2.27s/it][A
    Iteration:  39%|███████████▎                 | 462/1184 [17:17<27:04,  2.25s/it][A
    Iteration:  39%|███████████▎                 | 463/1184 [17:20<26:56,  2.24s/it][A
    Iteration:  39%|███████████▎                 | 464/1184 [17:22<26:50,  2.24s/it][A
    Iteration:  39%|███████████▍                 | 465/1184 [17:24<26:40,  2.23s/it][A
    Iteration:  39%|███████████▍                 | 466/1184 [17:26<26:29,  2.21s/it][A
    Iteration:  39%|███████████▍                 | 467/1184 [17:28<26:15,  2.20s/it][A
    Iteration:  40%|███████████▍                 | 468/1184 [17:30<26:04,  2.19s/it][A
    Iteration:  40%|███████████▍                 | 469/1184 [17:33<25:57,  2.18s/it][A
    Iteration:  40%|███████████▌                 | 470/1184 [17:35<26:00,  2.19s/it][A
    Iteration:  40%|███████████▌                 | 471/1184 [17:37<26:27,  2.23s/it][A
    Iteration:  40%|███████████▌                 | 472/1184 [17:39<26:27,  2.23s/it][A
    Iteration:  40%|███████████▌                 | 473/1184 [17:42<26:21,  2.22s/it][A
    Iteration:  40%|███████████▌                 | 474/1184 [17:44<26:43,  2.26s/it][A
    Iteration:  40%|███████████▋                 | 475/1184 [17:46<26:27,  2.24s/it][A
    Iteration:  40%|███████████▋                 | 476/1184 [17:48<26:08,  2.22s/it][A
    Iteration:  40%|███████████▋                 | 477/1184 [17:50<25:56,  2.20s/it][A
    Iteration:  40%|███████████▋                 | 478/1184 [17:53<25:47,  2.19s/it][A
    Iteration:  40%|███████████▋                 | 479/1184 [17:55<25:54,  2.21s/it][A
    Iteration:  41%|███████████▊                 | 480/1184 [17:57<25:48,  2.20s/it][A
    Iteration:  41%|███████████▊                 | 481/1184 [17:59<25:51,  2.21s/it][A
    Iteration:  41%|███████████▊                 | 482/1184 [18:02<25:59,  2.22s/it][A
    Iteration:  41%|███████████▊                 | 483/1184 [18:04<26:14,  2.25s/it][A
    Iteration:  41%|███████████▊                 | 484/1184 [18:06<26:07,  2.24s/it][A
    Iteration:  41%|███████████▉                 | 485/1184 [18:08<25:55,  2.23s/it][A
    Iteration:  41%|███████████▉                 | 486/1184 [18:10<25:41,  2.21s/it][A
    Iteration:  41%|███████████▉                 | 487/1184 [18:13<25:32,  2.20s/it][A
    Iteration:  41%|███████████▉                 | 488/1184 [18:15<25:21,  2.19s/it][A
    Iteration:  41%|███████████▉                 | 489/1184 [18:17<25:35,  2.21s/it][A
    Iteration:  41%|████████████                 | 490/1184 [18:19<25:34,  2.21s/it][A
    Iteration:  41%|████████████                 | 491/1184 [18:21<25:25,  2.20s/it][A
    Iteration:  42%|████████████                 | 492/1184 [18:24<25:15,  2.19s/it][A
    Iteration:  42%|████████████                 | 493/1184 [18:26<25:09,  2.18s/it][A
    Iteration:  42%|████████████                 | 494/1184 [18:28<25:08,  2.19s/it][A
    Iteration:  42%|████████████                 | 495/1184 [18:30<25:02,  2.18s/it][A
    Iteration:  42%|████████████▏                | 496/1184 [18:32<25:03,  2.18s/it][A
    Iteration:  42%|████████████▏                | 497/1184 [18:34<24:58,  2.18s/it][A
    Iteration:  42%|████████████▏                | 498/1184 [18:37<24:51,  2.17s/it][A
    Iteration:  42%|████████████▏                | 499/1184 [18:39<24:47,  2.17s/it][A
    Iteration:  42%|████████████▏                | 500/1184 [18:41<24:44,  2.17s/it][A
    Iteration:  42%|████████████▎                | 501/1184 [18:43<24:35,  2.16s/it][A
    Iteration:  42%|████████████▎                | 502/1184 [18:45<24:46,  2.18s/it][A
    Iteration:  42%|████████████▎                | 503/1184 [18:48<24:49,  2.19s/it][A
    Iteration:  43%|████████████▎                | 504/1184 [18:50<25:00,  2.21s/it][A
    Iteration:  43%|████████████▎                | 505/1184 [18:52<25:27,  2.25s/it][A
    Iteration:  43%|████████████▍                | 506/1184 [18:54<25:20,  2.24s/it][A
    Iteration:  43%|████████████▍                | 507/1184 [18:57<25:26,  2.26s/it][A
    Iteration:  43%|████████████▍                | 508/1184 [18:59<25:04,  2.23s/it][A
    Iteration:  43%|████████████▍                | 509/1184 [19:01<25:01,  2.22s/it][A
    Iteration:  43%|████████████▍                | 510/1184 [19:03<24:57,  2.22s/it][A
    Iteration:  43%|████████████▌                | 511/1184 [19:06<25:06,  2.24s/it][A
    Iteration:  43%|████████████▌                | 512/1184 [19:08<24:53,  2.22s/it][A
    Iteration:  43%|████████████▌                | 513/1184 [19:10<24:47,  2.22s/it][A
    Iteration:  43%|████████████▌                | 514/1184 [19:12<24:32,  2.20s/it][A
    Iteration:  43%|████████████▌                | 515/1184 [19:14<24:38,  2.21s/it][A
    Iteration:  44%|████████████▋                | 516/1184 [19:17<24:40,  2.22s/it][A
    Iteration:  44%|████████████▋                | 517/1184 [19:19<24:58,  2.25s/it][A
    Iteration:  44%|████████████▋                | 518/1184 [19:21<25:08,  2.26s/it][A
    Iteration:  44%|████████████▋                | 519/1184 [19:23<25:18,  2.28s/it][A
    Iteration:  44%|████████████▋                | 520/1184 [19:26<25:42,  2.32s/it][A
    Iteration:  44%|████████████▊                | 521/1184 [19:28<25:17,  2.29s/it][A
    Iteration:  44%|████████████▊                | 522/1184 [19:30<24:59,  2.26s/it][A
    Iteration:  44%|████████████▊                | 523/1184 [19:32<24:32,  2.23s/it][A
    Iteration:  44%|████████████▊                | 524/1184 [19:35<24:29,  2.23s/it][A
    Iteration:  44%|████████████▊                | 525/1184 [19:37<24:17,  2.21s/it][A
    Iteration:  44%|████████████▉                | 526/1184 [19:39<24:18,  2.22s/it][A
    Iteration:  45%|████████████▉                | 527/1184 [19:41<24:20,  2.22s/it][A
    Iteration:  45%|████████████▉                | 528/1184 [19:44<24:19,  2.22s/it][A
    Iteration:  45%|████████████▉                | 529/1184 [19:46<24:35,  2.25s/it][A
    Iteration:  45%|████████████▉                | 530/1184 [19:48<24:21,  2.23s/it][A
    Iteration:  45%|█████████████                | 531/1184 [19:50<24:18,  2.23s/it][A
    Iteration:  45%|█████████████                | 532/1184 [19:53<24:30,  2.25s/it][A
    Iteration:  45%|█████████████                | 533/1184 [19:55<24:22,  2.25s/it][A
    Iteration:  45%|█████████████                | 534/1184 [19:57<24:29,  2.26s/it][A
    Iteration:  45%|█████████████                | 535/1184 [19:59<24:22,  2.25s/it][A
    Iteration:  45%|█████████████▏               | 536/1184 [20:02<24:17,  2.25s/it][A
    Iteration:  45%|█████████████▏               | 537/1184 [20:04<24:13,  2.25s/it][A
    Iteration:  45%|█████████████▏               | 538/1184 [20:06<24:08,  2.24s/it][A
    Iteration:  46%|█████████████▏               | 539/1184 [20:08<24:17,  2.26s/it][A
    Iteration:  46%|█████████████▏               | 540/1184 [20:11<24:18,  2.26s/it][A
    Iteration:  46%|█████████████▎               | 541/1184 [20:13<23:59,  2.24s/it][A
    Iteration:  46%|█████████████▎               | 542/1184 [20:15<23:58,  2.24s/it][A
    Iteration:  46%|█████████████▎               | 543/1184 [20:18<25:06,  2.35s/it][A
    Iteration:  46%|█████████████▎               | 544/1184 [20:20<24:51,  2.33s/it][A
    Iteration:  46%|█████████████▎               | 545/1184 [20:22<24:37,  2.31s/it][A
    Iteration:  46%|█████████████▎               | 546/1184 [20:25<24:47,  2.33s/it][A
    Iteration:  46%|█████████████▍               | 547/1184 [20:27<26:23,  2.49s/it][A
    Iteration:  46%|█████████████▍               | 548/1184 [20:30<25:27,  2.40s/it][A
    Iteration:  46%|█████████████▍               | 549/1184 [20:32<24:53,  2.35s/it][A
    Iteration:  46%|█████████████▍               | 550/1184 [20:34<24:28,  2.32s/it][A
    Iteration:  47%|█████████████▍               | 551/1184 [20:36<24:11,  2.29s/it][A
    Iteration:  47%|█████████████▌               | 552/1184 [20:39<23:56,  2.27s/it][A
    Iteration:  47%|█████████████▌               | 553/1184 [20:41<23:38,  2.25s/it][A
    Iteration:  47%|█████████████▌               | 554/1184 [20:43<23:23,  2.23s/it][A
    Iteration:  47%|█████████████▌               | 555/1184 [20:45<23:07,  2.21s/it][A
    Iteration:  47%|█████████████▌               | 556/1184 [20:47<22:56,  2.19s/it][A
    Iteration:  47%|█████████████▋               | 557/1184 [20:49<22:57,  2.20s/it][A
    Iteration:  47%|█████████████▋               | 558/1184 [20:52<22:50,  2.19s/it][A
    Iteration:  47%|█████████████▋               | 559/1184 [20:54<22:41,  2.18s/it][A
    Iteration:  47%|█████████████▋               | 560/1184 [20:56<22:37,  2.18s/it][A
    Iteration:  47%|█████████████▋               | 561/1184 [20:58<22:32,  2.17s/it][A
    Iteration:  47%|█████████████▊               | 562/1184 [21:00<22:46,  2.20s/it][A
    Iteration:  48%|█████████████▊               | 563/1184 [21:03<22:48,  2.20s/it][A
    Iteration:  48%|█████████████▊               | 564/1184 [21:05<22:50,  2.21s/it][A
    Iteration:  48%|█████████████▊               | 565/1184 [21:07<22:50,  2.21s/it][A
    Iteration:  48%|█████████████▊               | 566/1184 [21:09<22:48,  2.21s/it][A
    Iteration:  48%|█████████████▉               | 567/1184 [21:12<22:49,  2.22s/it][A
    Iteration:  48%|█████████████▉               | 568/1184 [21:14<22:49,  2.22s/it][A
    Iteration:  48%|█████████████▉               | 569/1184 [21:16<23:57,  2.34s/it][A
    Iteration:  48%|█████████████▉               | 570/1184 [21:19<23:33,  2.30s/it][A
    Iteration:  48%|█████████████▉               | 571/1184 [21:21<23:19,  2.28s/it][A
    Iteration:  48%|██████████████               | 572/1184 [21:23<23:06,  2.27s/it][A
    Iteration:  48%|██████████████               | 573/1184 [21:25<22:56,  2.25s/it][A
    Iteration:  48%|██████████████               | 574/1184 [21:28<22:53,  2.25s/it][A
    Iteration:  49%|██████████████               | 575/1184 [21:30<24:27,  2.41s/it][A
    Iteration:  49%|██████████████               | 576/1184 [21:32<23:35,  2.33s/it][A
    Iteration:  49%|██████████████▏              | 577/1184 [21:35<23:12,  2.29s/it][A
    Iteration:  49%|██████████████▏              | 578/1184 [21:37<22:47,  2.26s/it][A
    Iteration:  49%|██████████████▏              | 579/1184 [21:39<22:39,  2.25s/it][A
    Iteration:  49%|██████████████▏              | 580/1184 [21:41<22:24,  2.23s/it][A
    Iteration:  49%|██████████████▏              | 581/1184 [21:43<22:21,  2.22s/it][A
    Iteration:  49%|██████████████▎              | 582/1184 [21:46<22:26,  2.24s/it][A
    Iteration:  49%|██████████████▎              | 583/1184 [21:48<22:36,  2.26s/it][A
    Iteration:  49%|██████████████▎              | 584/1184 [21:50<22:34,  2.26s/it][A
    Iteration:  49%|██████████████▎              | 585/1184 [21:53<22:30,  2.25s/it][A
    Iteration:  49%|██████████████▎              | 586/1184 [21:55<22:13,  2.23s/it][A
    Iteration:  50%|██████████████▍              | 587/1184 [21:57<21:56,  2.21s/it][A
    Iteration:  50%|██████████████▍              | 588/1184 [21:59<21:53,  2.20s/it][A
    Iteration:  50%|██████████████▍              | 589/1184 [22:01<21:46,  2.20s/it][A
    Iteration:  50%|██████████████▍              | 590/1184 [22:03<21:40,  2.19s/it][A
    Iteration:  50%|██████████████▍              | 591/1184 [22:06<21:58,  2.22s/it][A
    Iteration:  50%|██████████████▌              | 592/1184 [22:08<21:50,  2.21s/it][A
    Iteration:  50%|██████████████▌              | 593/1184 [22:10<21:52,  2.22s/it][A
    Iteration:  50%|██████████████▌              | 594/1184 [22:12<21:57,  2.23s/it][A
    Iteration:  50%|██████████████▌              | 595/1184 [22:15<21:53,  2.23s/it][A
    Iteration:  50%|██████████████▌              | 596/1184 [22:17<21:44,  2.22s/it][A
    Iteration:  50%|██████████████▌              | 597/1184 [22:19<21:38,  2.21s/it][A
    Iteration:  51%|██████████████▋              | 598/1184 [22:21<21:34,  2.21s/it][A
    Iteration:  51%|██████████████▋              | 599/1184 [22:23<21:37,  2.22s/it][A
    Iteration:  51%|██████████████▋              | 600/1184 [22:26<21:38,  2.22s/it][A
    Iteration:  51%|██████████████▋              | 601/1184 [22:28<21:24,  2.20s/it][A
    Iteration:  51%|██████████████▋              | 602/1184 [22:30<21:28,  2.21s/it][A
    Iteration:  51%|██████████████▊              | 603/1184 [22:32<21:27,  2.22s/it][A
    Iteration:  51%|██████████████▊              | 604/1184 [22:34<21:21,  2.21s/it][A
    Iteration:  51%|██████████████▊              | 605/1184 [22:37<21:12,  2.20s/it][A
    Iteration:  51%|██████████████▊              | 606/1184 [22:39<21:05,  2.19s/it][A
    Iteration:  51%|██████████████▊              | 607/1184 [22:41<21:06,  2.19s/it][A
    Iteration:  51%|██████████████▉              | 608/1184 [22:43<21:02,  2.19s/it][A
    Iteration:  51%|██████████████▉              | 609/1184 [22:45<21:05,  2.20s/it][A
    Iteration:  52%|██████████████▉              | 610/1184 [22:48<20:55,  2.19s/it][A
    Iteration:  52%|██████████████▉              | 611/1184 [22:50<20:52,  2.19s/it][A
    Iteration:  52%|██████████████▉              | 612/1184 [22:52<20:40,  2.17s/it][A
    Iteration:  52%|███████████████              | 613/1184 [22:54<20:46,  2.18s/it][A
    Iteration:  52%|███████████████              | 614/1184 [22:56<20:49,  2.19s/it][A
    Iteration:  52%|███████████████              | 615/1184 [22:59<20:54,  2.20s/it][A
    Iteration:  52%|███████████████              | 616/1184 [23:01<20:59,  2.22s/it][A
    Iteration:  52%|███████████████              | 617/1184 [23:03<21:01,  2.23s/it][A
    Iteration:  52%|███████████████▏             | 618/1184 [23:05<20:58,  2.22s/it][A
    Iteration:  52%|███████████████▏             | 619/1184 [23:08<20:59,  2.23s/it][A
    Iteration:  52%|███████████████▏             | 620/1184 [23:10<20:59,  2.23s/it][A
    Iteration:  52%|███████████████▏             | 621/1184 [23:12<20:56,  2.23s/it][A
    Iteration:  53%|███████████████▏             | 622/1184 [23:14<20:53,  2.23s/it][A
    Iteration:  53%|███████████████▎             | 623/1184 [23:16<20:50,  2.23s/it][A
    Iteration:  53%|███████████████▎             | 624/1184 [23:19<20:50,  2.23s/it][A
    Iteration:  53%|███████████████▎             | 625/1184 [23:21<20:52,  2.24s/it][A
    Iteration:  53%|███████████████▎             | 626/1184 [23:23<20:53,  2.25s/it][A
    Iteration:  53%|███████████████▎             | 627/1184 [23:26<21:07,  2.28s/it][A
    Iteration:  53%|███████████████▍             | 628/1184 [23:28<20:52,  2.25s/it][A
    Iteration:  53%|███████████████▍             | 629/1184 [23:30<20:31,  2.22s/it][A
    Iteration:  53%|███████████████▍             | 630/1184 [23:32<20:33,  2.23s/it][A
    Iteration:  53%|███████████████▍             | 631/1184 [23:34<20:23,  2.21s/it][A
    Iteration:  53%|███████████████▍             | 632/1184 [23:36<20:11,  2.19s/it][A
    Iteration:  53%|███████████████▌             | 633/1184 [23:39<20:08,  2.19s/it][A
    Iteration:  54%|███████████████▌             | 634/1184 [23:41<20:06,  2.19s/it][A
    Iteration:  54%|███████████████▌             | 635/1184 [23:43<20:15,  2.21s/it][A
    Iteration:  54%|███████████████▌             | 636/1184 [23:45<20:09,  2.21s/it][A
    Iteration:  54%|███████████████▌             | 637/1184 [23:47<19:58,  2.19s/it][A
    Iteration:  54%|███████████████▋             | 638/1184 [23:50<20:16,  2.23s/it][A
    Iteration:  54%|███████████████▋             | 639/1184 [23:52<20:05,  2.21s/it][A
    Iteration:  54%|███████████████▋             | 640/1184 [23:54<20:07,  2.22s/it][A
    Iteration:  54%|███████████████▋             | 641/1184 [23:56<20:07,  2.22s/it][A
    Iteration:  54%|███████████████▋             | 642/1184 [23:59<20:06,  2.23s/it][A
    Iteration:  54%|███████████████▋             | 643/1184 [24:01<20:08,  2.23s/it][A
    Iteration:  54%|███████████████▊             | 644/1184 [24:03<20:21,  2.26s/it][A
    Iteration:  54%|███████████████▊             | 645/1184 [24:06<20:23,  2.27s/it][A
    Iteration:  55%|███████████████▊             | 646/1184 [24:08<20:16,  2.26s/it][A
    Iteration:  55%|███████████████▊             | 647/1184 [24:10<20:10,  2.25s/it][A
    Iteration:  55%|███████████████▊             | 648/1184 [24:12<19:59,  2.24s/it][A
    Iteration:  55%|███████████████▉             | 649/1184 [24:14<19:51,  2.23s/it][A
    Iteration:  55%|███████████████▉             | 650/1184 [24:17<19:56,  2.24s/it][A
    Iteration:  55%|███████████████▉             | 651/1184 [24:19<19:49,  2.23s/it][A
    Iteration:  55%|███████████████▉             | 652/1184 [24:21<19:52,  2.24s/it][A
    Iteration:  55%|███████████████▉             | 653/1184 [24:23<19:47,  2.24s/it][A
    Iteration:  55%|████████████████             | 654/1184 [24:26<19:43,  2.23s/it][A
    Iteration:  55%|████████████████             | 655/1184 [24:28<19:52,  2.25s/it][A
    Iteration:  55%|████████████████             | 656/1184 [24:30<19:46,  2.25s/it][A
    Iteration:  55%|████████████████             | 657/1184 [24:32<19:27,  2.22s/it][A
    Iteration:  56%|████████████████             | 658/1184 [24:34<19:27,  2.22s/it][A
    Iteration:  56%|████████████████▏            | 659/1184 [24:37<19:26,  2.22s/it][A
    Iteration:  56%|████████████████▏            | 660/1184 [24:39<19:46,  2.26s/it][A
    Iteration:  56%|████████████████▏            | 661/1184 [24:41<19:30,  2.24s/it][A
    Iteration:  56%|████████████████▏            | 662/1184 [24:43<19:17,  2.22s/it][A
    Iteration:  56%|████████████████▏            | 663/1184 [24:46<19:32,  2.25s/it][A
    Iteration:  56%|████████████████▎            | 664/1184 [24:48<19:46,  2.28s/it][A
    Iteration:  56%|████████████████▎            | 665/1184 [24:50<19:47,  2.29s/it][A
    Iteration:  56%|████████████████▎            | 666/1184 [24:53<19:53,  2.30s/it][A
    Iteration:  56%|████████████████▎            | 667/1184 [24:55<19:45,  2.29s/it][A
    Iteration:  56%|████████████████▎            | 668/1184 [24:57<19:38,  2.28s/it][A
    Iteration:  57%|████████████████▍            | 669/1184 [24:59<19:24,  2.26s/it][A
    Iteration:  57%|████████████████▍            | 670/1184 [25:02<19:08,  2.23s/it][A
    Iteration:  57%|████████████████▍            | 671/1184 [25:04<19:04,  2.23s/it][A
    Iteration:  57%|████████████████▍            | 672/1184 [25:06<18:59,  2.23s/it][A
    Iteration:  57%|████████████████▍            | 673/1184 [25:08<19:20,  2.27s/it][A
    Iteration:  57%|████████████████▌            | 674/1184 [25:11<19:13,  2.26s/it][A
    Iteration:  57%|████████████████▌            | 675/1184 [25:13<19:11,  2.26s/it][A
    Iteration:  57%|████████████████▌            | 676/1184 [25:15<19:09,  2.26s/it][A
    Iteration:  57%|████████████████▌            | 677/1184 [25:17<18:51,  2.23s/it][A
    Iteration:  57%|████████████████▌            | 678/1184 [25:20<18:45,  2.23s/it][A
    Iteration:  57%|████████████████▋            | 679/1184 [25:22<18:42,  2.22s/it][A
    Iteration:  57%|████████████████▋            | 680/1184 [25:24<18:36,  2.22s/it][A
    Iteration:  58%|████████████████▋            | 681/1184 [25:26<18:38,  2.22s/it][A
    Iteration:  58%|████████████████▋            | 682/1184 [25:29<18:43,  2.24s/it][A
    Iteration:  58%|████████████████▋            | 683/1184 [25:31<18:33,  2.22s/it][A
    Iteration:  58%|████████████████▊            | 684/1184 [25:33<18:29,  2.22s/it][A
    Iteration:  58%|████████████████▊            | 685/1184 [25:35<18:23,  2.21s/it][A
    Iteration:  58%|████████████████▊            | 686/1184 [25:37<18:23,  2.22s/it][A
    Iteration:  58%|████████████████▊            | 687/1184 [25:40<18:43,  2.26s/it][A
    Iteration:  58%|████████████████▊            | 688/1184 [25:42<18:43,  2.26s/it][A
    Iteration:  58%|████████████████▉            | 689/1184 [25:44<18:45,  2.27s/it][A
    Iteration:  58%|████████████████▉            | 690/1184 [25:47<18:40,  2.27s/it][A
    Iteration:  58%|████████████████▉            | 691/1184 [25:49<18:34,  2.26s/it][A
    Iteration:  58%|████████████████▉            | 692/1184 [25:51<18:29,  2.25s/it][A
    Iteration:  59%|████████████████▉            | 693/1184 [25:53<18:24,  2.25s/it][A
    Iteration:  59%|████████████████▉            | 694/1184 [25:56<18:20,  2.25s/it][A
    Iteration:  59%|█████████████████            | 695/1184 [25:58<18:16,  2.24s/it][A
    Iteration:  59%|█████████████████            | 696/1184 [26:00<18:03,  2.22s/it][A
    Iteration:  59%|█████████████████            | 697/1184 [26:02<18:06,  2.23s/it][A
    Iteration:  59%|█████████████████            | 698/1184 [26:04<18:05,  2.23s/it][A
    Iteration:  59%|█████████████████            | 699/1184 [26:07<18:06,  2.24s/it][A
    Iteration:  59%|█████████████████▏           | 700/1184 [26:09<18:11,  2.26s/it][A
    Iteration:  59%|█████████████████▏           | 701/1184 [26:11<18:24,  2.29s/it][A
    Iteration:  59%|█████████████████▏           | 702/1184 [26:14<18:12,  2.27s/it][A
    Iteration:  59%|█████████████████▏           | 703/1184 [26:16<18:06,  2.26s/it][A
    Iteration:  59%|█████████████████▏           | 704/1184 [26:18<18:01,  2.25s/it][A
    Iteration:  60%|█████████████████▎           | 705/1184 [26:20<17:58,  2.25s/it][A
    Iteration:  60%|█████████████████▎           | 706/1184 [26:23<17:55,  2.25s/it][A
    Iteration:  60%|█████████████████▎           | 707/1184 [26:25<17:53,  2.25s/it][A
    Iteration:  60%|█████████████████▎           | 708/1184 [26:27<17:51,  2.25s/it][A
    Iteration:  60%|█████████████████▎           | 709/1184 [26:29<17:49,  2.25s/it][A
    Iteration:  60%|█████████████████▍           | 710/1184 [26:32<17:47,  2.25s/it][A
    Iteration:  60%|█████████████████▍           | 711/1184 [26:34<17:43,  2.25s/it][A
    Iteration:  60%|█████████████████▍           | 712/1184 [26:36<17:49,  2.26s/it][A
    Iteration:  60%|█████████████████▍           | 713/1184 [26:38<17:44,  2.26s/it][A
    Iteration:  60%|█████████████████▍           | 714/1184 [26:41<17:37,  2.25s/it][A
    Iteration:  60%|█████████████████▌           | 715/1184 [26:43<17:31,  2.24s/it][A
    Iteration:  60%|█████████████████▌           | 716/1184 [26:45<17:29,  2.24s/it][A
    Iteration:  61%|█████████████████▌           | 717/1184 [26:47<17:24,  2.24s/it][A
    Iteration:  61%|█████████████████▌           | 718/1184 [26:49<17:14,  2.22s/it][A
    Iteration:  61%|█████████████████▌           | 719/1184 [26:52<17:04,  2.20s/it][A
    Iteration:  61%|█████████████████▋           | 720/1184 [26:54<17:02,  2.20s/it][A
    Iteration:  61%|█████████████████▋           | 721/1184 [26:56<17:13,  2.23s/it][A
    Iteration:  61%|█████████████████▋           | 722/1184 [26:58<17:20,  2.25s/it][A
    Iteration:  61%|█████████████████▋           | 723/1184 [27:01<17:17,  2.25s/it][A
    Iteration:  61%|█████████████████▋           | 724/1184 [27:03<17:12,  2.25s/it][A
    Iteration:  61%|█████████████████▊           | 725/1184 [27:05<17:07,  2.24s/it][A
    Iteration:  61%|█████████████████▊           | 726/1184 [27:07<17:04,  2.24s/it][A
    Iteration:  61%|█████████████████▊           | 727/1184 [27:10<17:03,  2.24s/it][A
    Iteration:  61%|█████████████████▊           | 728/1184 [27:12<17:00,  2.24s/it][A
    Iteration:  62%|█████████████████▊           | 729/1184 [27:14<16:59,  2.24s/it][A
    Iteration:  62%|█████████████████▉           | 730/1184 [27:16<16:57,  2.24s/it][A
    Iteration:  62%|█████████████████▉           | 731/1184 [27:19<16:55,  2.24s/it][A
    Iteration:  62%|█████████████████▉           | 732/1184 [27:21<16:52,  2.24s/it][A
    Iteration:  62%|█████████████████▉           | 733/1184 [27:23<16:51,  2.24s/it][A
    Iteration:  62%|█████████████████▉           | 734/1184 [27:25<16:40,  2.22s/it][A
    Iteration:  62%|██████████████████           | 735/1184 [27:27<16:29,  2.20s/it][A
    Iteration:  62%|██████████████████           | 736/1184 [27:29<16:21,  2.19s/it][A
    Iteration:  62%|██████████████████           | 737/1184 [27:32<16:24,  2.20s/it][A
    Iteration:  62%|██████████████████           | 738/1184 [27:34<16:30,  2.22s/it][A
    Iteration:  62%|██████████████████           | 739/1184 [27:36<16:29,  2.22s/it][A
    Iteration:  62%|██████████████████▏          | 740/1184 [27:39<16:40,  2.25s/it][A
    Iteration:  63%|██████████████████▏          | 741/1184 [27:41<16:43,  2.26s/it][A
    Iteration:  63%|██████████████████▏          | 742/1184 [27:43<17:07,  2.32s/it][A
    Iteration:  63%|██████████████████▏          | 743/1184 [27:45<16:44,  2.28s/it][A
    Iteration:  63%|██████████████████▏          | 744/1184 [27:48<16:29,  2.25s/it][A
    Iteration:  63%|██████████████████▏          | 745/1184 [27:50<16:29,  2.25s/it][A
    Iteration:  63%|██████████████████▎          | 746/1184 [27:52<16:27,  2.26s/it][A
    Iteration:  63%|██████████████████▎          | 747/1184 [27:54<16:20,  2.24s/it][A
    Iteration:  63%|██████████████████▎          | 748/1184 [27:57<16:13,  2.23s/it][A
    Iteration:  63%|██████████████████▎          | 749/1184 [27:59<16:07,  2.22s/it][A
    Iteration:  63%|██████████████████▎          | 750/1184 [28:01<16:07,  2.23s/it][A
    Iteration:  63%|██████████████████▍          | 751/1184 [28:03<16:13,  2.25s/it][A
    Iteration:  64%|██████████████████▍          | 752/1184 [28:06<16:04,  2.23s/it][A
    Iteration:  64%|██████████████████▍          | 753/1184 [28:08<15:54,  2.21s/it][A
    Iteration:  64%|██████████████████▍          | 754/1184 [28:10<15:48,  2.21s/it][A
    Iteration:  64%|██████████████████▍          | 755/1184 [28:12<15:50,  2.21s/it][A
    Iteration:  64%|██████████████████▌          | 756/1184 [28:14<15:49,  2.22s/it][A
    Iteration:  64%|██████████████████▌          | 757/1184 [28:17<15:48,  2.22s/it][A
    Iteration:  64%|██████████████████▌          | 758/1184 [28:19<15:49,  2.23s/it][A
    Iteration:  64%|██████████████████▌          | 759/1184 [28:21<15:47,  2.23s/it][A
    Iteration:  64%|██████████████████▌          | 760/1184 [28:23<15:49,  2.24s/it][A
    Iteration:  64%|██████████████████▋          | 761/1184 [28:26<15:46,  2.24s/it][A
    Iteration:  64%|██████████████████▋          | 762/1184 [28:28<15:45,  2.24s/it][A
    Iteration:  64%|██████████████████▋          | 763/1184 [28:30<15:42,  2.24s/it][A
    Iteration:  65%|██████████████████▋          | 764/1184 [28:32<15:40,  2.24s/it][A
    Iteration:  65%|██████████████████▋          | 765/1184 [28:34<15:35,  2.23s/it][A
    Iteration:  65%|██████████████████▊          | 766/1184 [28:37<15:44,  2.26s/it][A
    Iteration:  65%|██████████████████▊          | 767/1184 [28:39<15:41,  2.26s/it][A
    Iteration:  65%|██████████████████▊          | 768/1184 [28:41<15:37,  2.25s/it][A
    Iteration:  65%|██████████████████▊          | 769/1184 [28:44<15:35,  2.25s/it][A
    Iteration:  65%|██████████████████▊          | 770/1184 [28:46<15:40,  2.27s/it][A
    Iteration:  65%|██████████████████▉          | 771/1184 [28:48<15:33,  2.26s/it][A
    Iteration:  65%|██████████████████▉          | 772/1184 [28:50<15:25,  2.25s/it][A
    Iteration:  65%|██████████████████▉          | 773/1184 [28:53<15:25,  2.25s/it][A
    Iteration:  65%|██████████████████▉          | 774/1184 [28:55<15:28,  2.27s/it][A
    Iteration:  65%|██████████████████▉          | 775/1184 [28:57<15:24,  2.26s/it][A
    Iteration:  66%|███████████████████          | 776/1184 [28:59<15:22,  2.26s/it][A
    Iteration:  66%|███████████████████          | 777/1184 [29:02<15:12,  2.24s/it][A
    Iteration:  66%|███████████████████          | 778/1184 [29:04<15:10,  2.24s/it][A
    Iteration:  66%|███████████████████          | 779/1184 [29:06<15:08,  2.24s/it][A
    Iteration:  66%|███████████████████          | 780/1184 [29:08<15:08,  2.25s/it][A
    Iteration:  66%|███████████████████▏         | 781/1184 [29:11<15:06,  2.25s/it][A
    Iteration:  66%|███████████████████▏         | 782/1184 [29:13<15:06,  2.25s/it][A
    Iteration:  66%|███████████████████▏         | 783/1184 [29:15<15:03,  2.25s/it][A
    Iteration:  66%|███████████████████▏         | 784/1184 [29:17<14:59,  2.25s/it][A
    Iteration:  66%|███████████████████▏         | 785/1184 [29:20<14:57,  2.25s/it][A
    Iteration:  66%|███████████████████▎         | 786/1184 [29:22<14:54,  2.25s/it][A
    Iteration:  66%|███████████████████▎         | 787/1184 [29:24<14:52,  2.25s/it][A
    Iteration:  67%|███████████████████▎         | 788/1184 [29:26<14:51,  2.25s/it][A
    Iteration:  67%|███████████████████▎         | 789/1184 [29:29<14:48,  2.25s/it][A
    Iteration:  67%|███████████████████▎         | 790/1184 [29:31<14:47,  2.25s/it][A
    Iteration:  67%|███████████████████▎         | 791/1184 [29:33<14:38,  2.24s/it][A
    Iteration:  67%|███████████████████▍         | 792/1184 [29:35<14:36,  2.24s/it][A
    Iteration:  67%|███████████████████▍         | 793/1184 [29:38<14:34,  2.24s/it][A
    Iteration:  67%|███████████████████▍         | 794/1184 [29:40<14:32,  2.24s/it][A
    Iteration:  67%|███████████████████▍         | 795/1184 [29:42<14:24,  2.22s/it][A
    Iteration:  67%|███████████████████▍         | 796/1184 [29:44<14:24,  2.23s/it][A
    Iteration:  67%|███████████████████▌         | 797/1184 [29:46<14:24,  2.23s/it][A
    Iteration:  67%|███████████████████▌         | 798/1184 [29:49<14:31,  2.26s/it][A
    Iteration:  67%|███████████████████▌         | 799/1184 [29:51<14:45,  2.30s/it][A
    Iteration:  68%|███████████████████▌         | 800/1184 [29:53<14:40,  2.29s/it][A
    Iteration:  68%|███████████████████▌         | 801/1184 [29:56<14:36,  2.29s/it][A
    Iteration:  68%|███████████████████▋         | 802/1184 [29:58<14:49,  2.33s/it][A
    Iteration:  68%|███████████████████▋         | 803/1184 [30:00<14:48,  2.33s/it][A
    Iteration:  68%|███████████████████▋         | 804/1184 [30:03<14:36,  2.31s/it][A
    Iteration:  68%|███████████████████▋         | 805/1184 [30:05<14:28,  2.29s/it][A
    Iteration:  68%|███████████████████▋         | 806/1184 [30:07<14:19,  2.28s/it][A
    Iteration:  68%|███████████████████▊         | 807/1184 [30:09<14:13,  2.26s/it][A
    Iteration:  68%|███████████████████▊         | 808/1184 [30:12<14:07,  2.25s/it][A
    Iteration:  68%|███████████████████▊         | 809/1184 [30:14<14:03,  2.25s/it][A
    Iteration:  68%|███████████████████▊         | 810/1184 [30:16<13:59,  2.25s/it][A
    Iteration:  68%|███████████████████▊         | 811/1184 [30:18<13:57,  2.25s/it][A
    Iteration:  69%|███████████████████▉         | 812/1184 [30:21<14:09,  2.28s/it][A
    Iteration:  69%|███████████████████▉         | 813/1184 [30:23<13:55,  2.25s/it][A
    Iteration:  69%|███████████████████▉         | 814/1184 [30:25<13:47,  2.24s/it][A
    Iteration:  69%|███████████████████▉         | 815/1184 [30:27<13:39,  2.22s/it][A
    Iteration:  69%|███████████████████▉         | 816/1184 [30:30<13:46,  2.25s/it][A
    Iteration:  69%|████████████████████         | 817/1184 [30:32<13:38,  2.23s/it][A
    Iteration:  69%|████████████████████         | 818/1184 [30:34<13:30,  2.22s/it][A
    Iteration:  69%|████████████████████         | 819/1184 [30:36<13:32,  2.23s/it][A
    Iteration:  69%|████████████████████         | 820/1184 [30:39<13:50,  2.28s/it][A
    Iteration:  69%|████████████████████         | 821/1184 [30:41<13:37,  2.25s/it][A
    Iteration:  69%|████████████████████▏        | 822/1184 [30:43<13:38,  2.26s/it][A
    Iteration:  70%|████████████████████▏        | 823/1184 [30:45<13:27,  2.24s/it][A
    Iteration:  70%|████████████████████▏        | 824/1184 [30:48<13:27,  2.24s/it][A
    Iteration:  70%|████████████████████▏        | 825/1184 [30:50<13:21,  2.23s/it][A
    Iteration:  70%|████████████████████▏        | 826/1184 [30:52<13:20,  2.24s/it][A
    Iteration:  70%|████████████████████▎        | 827/1184 [30:54<13:29,  2.27s/it][A
    Iteration:  70%|████████████████████▎        | 828/1184 [30:57<13:28,  2.27s/it][A
    Iteration:  70%|████████████████████▎        | 829/1184 [30:59<13:22,  2.26s/it][A
    Iteration:  70%|████████████████████▎        | 830/1184 [31:01<13:14,  2.24s/it][A
    Iteration:  70%|████████████████████▎        | 831/1184 [31:03<13:05,  2.22s/it][A
    Iteration:  70%|████████████████████▍        | 832/1184 [31:06<13:02,  2.22s/it][A
    Iteration:  70%|████████████████████▍        | 833/1184 [31:08<12:52,  2.20s/it][A
    Iteration:  70%|████████████████████▍        | 834/1184 [31:10<12:51,  2.20s/it][A
    Iteration:  71%|████████████████████▍        | 835/1184 [31:12<12:43,  2.19s/it][A
    Iteration:  71%|████████████████████▍        | 836/1184 [31:14<12:39,  2.18s/it][A
    Iteration:  71%|████████████████████▌        | 837/1184 [31:16<12:46,  2.21s/it][A
    Iteration:  71%|████████████████████▌        | 838/1184 [31:19<12:56,  2.24s/it][A
    Iteration:  71%|████████████████████▌        | 839/1184 [31:21<12:52,  2.24s/it][A
    Iteration:  71%|████████████████████▌        | 840/1184 [31:23<12:59,  2.27s/it][A
    Iteration:  71%|████████████████████▌        | 841/1184 [31:26<12:55,  2.26s/it][A
    Iteration:  71%|████████████████████▌        | 842/1184 [31:28<12:51,  2.26s/it][A
    Iteration:  71%|████████████████████▋        | 843/1184 [31:30<12:47,  2.25s/it][A
    Iteration:  71%|████████████████████▋        | 844/1184 [31:32<12:44,  2.25s/it][A
    Iteration:  71%|████████████████████▋        | 845/1184 [31:35<12:50,  2.27s/it][A
    Iteration:  71%|████████████████████▋        | 846/1184 [31:37<12:45,  2.27s/it][A
    Iteration:  72%|████████████████████▋        | 847/1184 [31:39<12:41,  2.26s/it][A
    Iteration:  72%|████████████████████▊        | 848/1184 [31:41<12:39,  2.26s/it][A
    Iteration:  72%|████████████████████▊        | 849/1184 [31:44<12:39,  2.27s/it][A
    Iteration:  72%|████████████████████▊        | 850/1184 [31:46<12:37,  2.27s/it][A
    Iteration:  72%|████████████████████▊        | 851/1184 [31:48<12:31,  2.26s/it][A
    Iteration:  72%|████████████████████▊        | 852/1184 [31:50<12:34,  2.27s/it][A
    Iteration:  72%|████████████████████▉        | 853/1184 [31:53<12:39,  2.29s/it][A
    Iteration:  72%|████████████████████▉        | 854/1184 [31:55<12:40,  2.31s/it][A
    Iteration:  72%|████████████████████▉        | 855/1184 [31:57<12:30,  2.28s/it][A
    Iteration:  72%|████████████████████▉        | 856/1184 [32:00<12:23,  2.27s/it][A
    Iteration:  72%|████████████████████▉        | 857/1184 [32:02<12:27,  2.29s/it][A
    Iteration:  72%|█████████████████████        | 858/1184 [32:04<12:23,  2.28s/it][A
    Iteration:  73%|█████████████████████        | 859/1184 [32:06<12:10,  2.25s/it][A
    Iteration:  73%|█████████████████████        | 860/1184 [32:09<12:03,  2.23s/it][A
    Iteration:  73%|█████████████████████        | 861/1184 [32:11<11:58,  2.23s/it][A
    Iteration:  73%|█████████████████████        | 862/1184 [32:13<12:03,  2.25s/it][A
    Iteration:  73%|█████████████████████▏       | 863/1184 [32:15<12:11,  2.28s/it][A
    Iteration:  73%|█████████████████████▏       | 864/1184 [32:18<12:00,  2.25s/it][A
    Iteration:  73%|█████████████████████▏       | 865/1184 [32:20<12:02,  2.26s/it][A
    Iteration:  73%|█████████████████████▏       | 866/1184 [32:22<11:54,  2.25s/it][A
    Iteration:  73%|█████████████████████▏       | 867/1184 [32:24<11:46,  2.23s/it][A
    Iteration:  73%|█████████████████████▎       | 868/1184 [32:26<11:38,  2.21s/it][A
    Iteration:  73%|█████████████████████▎       | 869/1184 [32:29<11:40,  2.22s/it][A
    Iteration:  73%|█████████████████████▎       | 870/1184 [32:31<11:31,  2.20s/it][A
    Iteration:  74%|█████████████████████▎       | 871/1184 [32:33<11:33,  2.22s/it][A
    Iteration:  74%|█████████████████████▎       | 872/1184 [32:35<11:40,  2.25s/it][A
    Iteration:  74%|█████████████████████▍       | 873/1184 [32:38<11:36,  2.24s/it][A
    Iteration:  74%|█████████████████████▍       | 874/1184 [32:40<11:29,  2.23s/it][A
    Iteration:  74%|█████████████████████▍       | 875/1184 [32:42<11:23,  2.21s/it][A
    Iteration:  74%|█████████████████████▍       | 876/1184 [32:44<11:14,  2.19s/it][A
    Iteration:  74%|█████████████████████▍       | 877/1184 [32:46<11:16,  2.20s/it][A
    Iteration:  74%|█████████████████████▌       | 878/1184 [32:49<11:12,  2.20s/it][A
    Iteration:  74%|█████████████████████▌       | 879/1184 [32:51<11:06,  2.18s/it][A
    Iteration:  74%|█████████████████████▌       | 880/1184 [32:53<11:05,  2.19s/it][A
    Iteration:  74%|█████████████████████▌       | 881/1184 [32:55<11:02,  2.19s/it][A
    Iteration:  74%|█████████████████████▌       | 882/1184 [32:57<11:03,  2.20s/it][A
    Iteration:  75%|█████████████████████▋       | 883/1184 [33:00<11:13,  2.24s/it][A
    Iteration:  75%|█████████████████████▋       | 884/1184 [33:02<11:10,  2.24s/it][A
    Iteration:  75%|█████████████████████▋       | 885/1184 [33:05<12:02,  2.42s/it][A
    Iteration:  75%|█████████████████████▋       | 886/1184 [33:07<11:51,  2.39s/it][A
    Iteration:  75%|█████████████████████▋       | 887/1184 [33:09<11:38,  2.35s/it][A
    Iteration:  75%|█████████████████████▊       | 888/1184 [33:12<11:21,  2.30s/it][A
    Iteration:  75%|█████████████████████▊       | 889/1184 [33:14<11:06,  2.26s/it][A
    Iteration:  75%|█████████████████████▊       | 890/1184 [33:16<10:56,  2.23s/it][A
    Iteration:  75%|█████████████████████▊       | 891/1184 [33:18<10:47,  2.21s/it][A
    Iteration:  75%|█████████████████████▊       | 892/1184 [33:20<10:41,  2.20s/it][A
    Iteration:  75%|█████████████████████▊       | 893/1184 [33:22<10:41,  2.21s/it][A
    Iteration:  76%|█████████████████████▉       | 894/1184 [33:25<10:43,  2.22s/it][A
    Iteration:  76%|█████████████████████▉       | 895/1184 [33:27<10:42,  2.22s/it][A
    Iteration:  76%|█████████████████████▉       | 896/1184 [33:29<10:39,  2.22s/it][A
    Iteration:  76%|█████████████████████▉       | 897/1184 [33:31<10:38,  2.23s/it][A
    Iteration:  76%|█████████████████████▉       | 898/1184 [33:34<10:35,  2.22s/it][A
    Iteration:  76%|██████████████████████       | 899/1184 [33:36<10:29,  2.21s/it][A
    Iteration:  76%|██████████████████████       | 900/1184 [33:38<10:31,  2.22s/it][A
    Iteration:  76%|██████████████████████       | 901/1184 [33:40<10:31,  2.23s/it][A
    Iteration:  76%|██████████████████████       | 902/1184 [33:42<10:25,  2.22s/it][A
    Iteration:  76%|██████████████████████       | 903/1184 [33:45<10:17,  2.20s/it][A
    Iteration:  76%|██████████████████████▏      | 904/1184 [33:47<10:19,  2.21s/it][A
    Iteration:  76%|██████████████████████▏      | 905/1184 [33:49<10:19,  2.22s/it][A
    Iteration:  77%|██████████████████████▏      | 906/1184 [33:51<10:16,  2.22s/it][A
    Iteration:  77%|██████████████████████▏      | 907/1184 [33:53<10:08,  2.20s/it][A
    Iteration:  77%|██████████████████████▏      | 908/1184 [33:56<10:06,  2.20s/it][A
    Iteration:  77%|██████████████████████▎      | 909/1184 [33:58<10:12,  2.23s/it][A
    Iteration:  77%|██████████████████████▎      | 910/1184 [34:00<10:15,  2.25s/it][A
    Iteration:  77%|██████████████████████▎      | 911/1184 [34:02<10:13,  2.25s/it][A
    Iteration:  77%|██████████████████████▎      | 912/1184 [34:05<10:08,  2.24s/it][A
    Iteration:  77%|██████████████████████▎      | 913/1184 [34:07<10:11,  2.26s/it][A
    Iteration:  77%|██████████████████████▍      | 914/1184 [34:09<10:03,  2.24s/it][A
    Iteration:  77%|██████████████████████▍      | 915/1184 [34:11<10:00,  2.23s/it][A
    Iteration:  77%|██████████████████████▍      | 916/1184 [34:14<09:54,  2.22s/it][A
    Iteration:  77%|██████████████████████▍      | 917/1184 [34:16<09:47,  2.20s/it][A
    Iteration:  78%|██████████████████████▍      | 918/1184 [34:18<09:48,  2.21s/it][A
    Iteration:  78%|██████████████████████▌      | 919/1184 [34:20<09:47,  2.22s/it][A
    Iteration:  78%|██████████████████████▌      | 920/1184 [34:22<09:42,  2.21s/it][A
    Iteration:  78%|██████████████████████▌      | 921/1184 [34:25<09:37,  2.20s/it][A
    Iteration:  78%|██████████████████████▌      | 922/1184 [34:27<09:33,  2.19s/it][A
    Iteration:  78%|██████████████████████▌      | 923/1184 [34:29<09:33,  2.20s/it][A
    Iteration:  78%|██████████████████████▋      | 924/1184 [34:31<09:29,  2.19s/it][A
    Iteration:  78%|██████████████████████▋      | 925/1184 [34:33<09:26,  2.19s/it][A
    Iteration:  78%|██████████████████████▋      | 926/1184 [34:36<09:28,  2.20s/it][A
    Iteration:  78%|██████████████████████▋      | 927/1184 [34:38<09:27,  2.21s/it][A
    Iteration:  78%|██████████████████████▋      | 928/1184 [34:40<09:29,  2.22s/it][A
    Iteration:  78%|██████████████████████▊      | 929/1184 [34:42<09:26,  2.22s/it][A
    Iteration:  79%|██████████████████████▊      | 930/1184 [34:44<09:23,  2.22s/it][A
    Iteration:  79%|██████████████████████▊      | 931/1184 [34:47<09:25,  2.24s/it][A
    Iteration:  79%|██████████████████████▊      | 932/1184 [34:49<09:22,  2.23s/it][A
    Iteration:  79%|██████████████████████▊      | 933/1184 [34:51<09:25,  2.25s/it][A
    Iteration:  79%|██████████████████████▉      | 934/1184 [34:54<09:23,  2.25s/it][A
    Iteration:  79%|██████████████████████▉      | 935/1184 [34:56<09:20,  2.25s/it][A
    Iteration:  79%|██████████████████████▉      | 936/1184 [34:58<09:17,  2.25s/it][A
    Iteration:  79%|██████████████████████▉      | 937/1184 [35:00<09:12,  2.24s/it][A
    Iteration:  79%|██████████████████████▉      | 938/1184 [35:03<09:14,  2.25s/it][A
    Iteration:  79%|██████████████████████▉      | 939/1184 [35:05<09:12,  2.26s/it][A
    Iteration:  79%|███████████████████████      | 940/1184 [35:07<09:10,  2.25s/it][A
    Iteration:  79%|███████████████████████      | 941/1184 [35:09<09:06,  2.25s/it][A
    Iteration:  80%|███████████████████████      | 942/1184 [35:12<09:32,  2.36s/it][A
    Iteration:  80%|███████████████████████      | 943/1184 [35:14<09:17,  2.31s/it][A
    Iteration:  80%|███████████████████████      | 944/1184 [35:16<09:08,  2.28s/it][A
    Iteration:  80%|███████████████████████▏     | 945/1184 [35:18<08:59,  2.26s/it][A
    Iteration:  80%|███████████████████████▏     | 946/1184 [35:21<08:56,  2.25s/it][A
    Iteration:  80%|███████████████████████▏     | 947/1184 [35:23<08:57,  2.27s/it][A
    Iteration:  80%|███████████████████████▏     | 948/1184 [35:25<08:56,  2.27s/it][A
    Iteration:  80%|███████████████████████▏     | 949/1184 [35:28<08:47,  2.25s/it][A
    Iteration:  80%|███████████████████████▎     | 950/1184 [35:30<08:46,  2.25s/it][A
    Iteration:  80%|███████████████████████▎     | 951/1184 [35:32<08:39,  2.23s/it][A
    Iteration:  80%|███████████████████████▎     | 952/1184 [35:34<08:37,  2.23s/it][A
    Iteration:  80%|███████████████████████▎     | 953/1184 [35:36<08:35,  2.23s/it][A
    Iteration:  81%|███████████████████████▎     | 954/1184 [35:39<08:33,  2.23s/it][A
    Iteration:  81%|███████████████████████▍     | 955/1184 [35:41<08:31,  2.23s/it][A
    Iteration:  81%|███████████████████████▍     | 956/1184 [35:43<08:29,  2.23s/it][A
    Iteration:  81%|███████████████████████▍     | 957/1184 [35:45<08:31,  2.25s/it][A
    Iteration:  81%|███████████████████████▍     | 958/1184 [35:48<08:31,  2.26s/it][A
    Iteration:  81%|███████████████████████▍     | 959/1184 [35:50<08:22,  2.23s/it][A
    Iteration:  81%|███████████████████████▌     | 960/1184 [35:52<08:17,  2.22s/it][A
    Iteration:  81%|███████████████████████▌     | 961/1184 [35:54<08:11,  2.20s/it][A
    Iteration:  81%|███████████████████████▌     | 962/1184 [35:57<08:16,  2.24s/it][A
    Iteration:  81%|███████████████████████▌     | 963/1184 [35:59<08:10,  2.22s/it][A
    Iteration:  81%|███████████████████████▌     | 964/1184 [36:01<08:04,  2.20s/it][A
    Iteration:  82%|███████████████████████▋     | 965/1184 [36:03<07:59,  2.19s/it][A
    Iteration:  82%|███████████████████████▋     | 966/1184 [36:05<08:01,  2.21s/it][A
    Iteration:  82%|███████████████████████▋     | 967/1184 [36:07<07:57,  2.20s/it][A
    Iteration:  82%|███████████████████████▋     | 968/1184 [36:10<07:58,  2.21s/it][A
    Iteration:  82%|███████████████████████▋     | 969/1184 [36:12<07:55,  2.21s/it][A
    Iteration:  82%|███████████████████████▊     | 970/1184 [36:14<07:59,  2.24s/it][A
    Iteration:  82%|███████████████████████▊     | 971/1184 [36:16<07:58,  2.25s/it][A
    Iteration:  82%|███████████████████████▊     | 972/1184 [36:19<07:55,  2.24s/it][A
    Iteration:  82%|███████████████████████▊     | 973/1184 [36:21<07:49,  2.23s/it][A
    Iteration:  82%|███████████████████████▊     | 974/1184 [36:23<07:55,  2.26s/it][A
    Iteration:  82%|███████████████████████▉     | 975/1184 [36:25<07:48,  2.24s/it][A
    Iteration:  82%|███████████████████████▉     | 976/1184 [36:28<07:43,  2.23s/it][A
    Iteration:  83%|███████████████████████▉     | 977/1184 [36:30<07:41,  2.23s/it][A
    Iteration:  83%|███████████████████████▉     | 978/1184 [36:32<07:37,  2.22s/it][A
    Iteration:  83%|███████████████████████▉     | 979/1184 [36:34<07:33,  2.21s/it][A
    Iteration:  83%|████████████████████████     | 980/1184 [36:36<07:28,  2.20s/it][A
    Iteration:  83%|████████████████████████     | 981/1184 [36:39<07:23,  2.18s/it][A
    Iteration:  83%|████████████████████████     | 982/1184 [36:41<07:17,  2.17s/it][A
    Iteration:  83%|████████████████████████     | 983/1184 [36:43<07:14,  2.16s/it][A
    Iteration:  83%|████████████████████████     | 984/1184 [36:45<07:10,  2.15s/it][A
    Iteration:  83%|████████████████████████▏    | 985/1184 [36:47<07:08,  2.16s/it][A
    Iteration:  83%|████████████████████████▏    | 986/1184 [36:49<07:06,  2.16s/it][A
    Iteration:  83%|████████████████████████▏    | 987/1184 [36:51<07:03,  2.15s/it][A
    Iteration:  83%|████████████████████████▏    | 988/1184 [36:54<07:09,  2.19s/it][A
    Iteration:  84%|████████████████████████▏    | 989/1184 [36:56<07:17,  2.24s/it][A
    Iteration:  84%|████████████████████████▏    | 990/1184 [36:58<07:13,  2.23s/it][A
    Iteration:  84%|████████████████████████▎    | 991/1184 [37:00<07:08,  2.22s/it][A
    Iteration:  84%|████████████████████████▎    | 992/1184 [37:03<07:06,  2.22s/it][A
    Iteration:  84%|████████████████████████▎    | 993/1184 [37:05<07:01,  2.20s/it][A
    Iteration:  84%|████████████████████████▎    | 994/1184 [37:07<07:00,  2.21s/it][A
    Iteration:  84%|████████████████████████▎    | 995/1184 [37:09<06:59,  2.22s/it][A
    Iteration:  84%|████████████████████████▍    | 996/1184 [37:12<06:55,  2.21s/it][A
    Iteration:  84%|████████████████████████▍    | 997/1184 [37:14<06:59,  2.25s/it][A
    Iteration:  84%|████████████████████████▍    | 998/1184 [37:16<06:58,  2.25s/it][A
    Iteration:  84%|████████████████████████▍    | 999/1184 [37:18<07:01,  2.28s/it][A
    Iteration:  84%|███████████████████████▋    | 1000/1184 [37:21<06:55,  2.26s/it][A
    Iteration:  85%|███████████████████████▋    | 1001/1184 [37:23<06:48,  2.23s/it][A
    Iteration:  85%|███████████████████████▋    | 1002/1184 [37:25<06:45,  2.23s/it][A
    Iteration:  85%|███████████████████████▋    | 1003/1184 [37:27<06:40,  2.21s/it][A
    Iteration:  85%|███████████████████████▋    | 1004/1184 [37:30<06:40,  2.22s/it][A
    Iteration:  85%|███████████████████████▊    | 1005/1184 [37:32<06:36,  2.21s/it][A
    Iteration:  85%|███████████████████████▊    | 1006/1184 [37:34<06:35,  2.22s/it][A
    Iteration:  85%|███████████████████████▊    | 1007/1184 [37:36<06:31,  2.21s/it][A
    Iteration:  85%|███████████████████████▊    | 1008/1184 [37:38<06:24,  2.18s/it][A
    Iteration:  85%|███████████████████████▊    | 1009/1184 [37:41<06:28,  2.22s/it][A
    Iteration:  85%|███████████████████████▉    | 1010/1184 [37:43<06:26,  2.22s/it][A
    Iteration:  85%|███████████████████████▉    | 1011/1184 [37:45<06:24,  2.22s/it][A
    Iteration:  85%|███████████████████████▉    | 1012/1184 [37:47<06:22,  2.22s/it][A
    Iteration:  86%|███████████████████████▉    | 1013/1184 [37:49<06:20,  2.22s/it][A
    Iteration:  86%|███████████████████████▉    | 1014/1184 [37:52<06:18,  2.23s/it][A
    Iteration:  86%|████████████████████████    | 1015/1184 [37:54<06:16,  2.23s/it][A
    Iteration:  86%|████████████████████████    | 1016/1184 [37:56<06:14,  2.23s/it][A
    Iteration:  86%|████████████████████████    | 1017/1184 [37:58<06:12,  2.23s/it][A
    Iteration:  86%|████████████████████████    | 1018/1184 [38:01<06:10,  2.23s/it][A
    Iteration:  86%|████████████████████████    | 1019/1184 [38:03<06:09,  2.24s/it][A
    Iteration:  86%|████████████████████████    | 1020/1184 [38:05<06:09,  2.25s/it][A
    Iteration:  86%|████████████████████████▏   | 1021/1184 [38:07<06:06,  2.25s/it][A
    Iteration:  86%|████████████████████████▏   | 1022/1184 [38:10<06:08,  2.28s/it][A
    Iteration:  86%|████████████████████████▏   | 1023/1184 [38:12<06:06,  2.28s/it][A
    Iteration:  86%|████████████████████████▏   | 1024/1184 [38:14<06:03,  2.27s/it][A
    Iteration:  87%|████████████████████████▏   | 1025/1184 [38:16<05:55,  2.24s/it][A
    Iteration:  87%|████████████████████████▎   | 1026/1184 [38:19<05:55,  2.25s/it][A
    Iteration:  87%|████████████████████████▎   | 1027/1184 [38:21<05:57,  2.28s/it][A
    Iteration:  87%|████████████████████████▎   | 1028/1184 [38:23<05:58,  2.30s/it][A
    Iteration:  87%|████████████████████████▎   | 1029/1184 [38:26<05:55,  2.29s/it][A
    Iteration:  87%|████████████████████████▎   | 1030/1184 [38:28<05:56,  2.32s/it][A
    Iteration:  87%|████████████████████████▍   | 1031/1184 [38:30<05:51,  2.30s/it][A
    Iteration:  87%|████████████████████████▍   | 1032/1184 [38:33<05:53,  2.33s/it][A
    Iteration:  87%|████████████████████████▍   | 1033/1184 [38:35<05:48,  2.31s/it][A
    Iteration:  87%|████████████████████████▍   | 1034/1184 [38:37<05:46,  2.31s/it][A
    Iteration:  87%|████████████████████████▍   | 1035/1184 [38:40<05:43,  2.31s/it][A
    Iteration:  88%|████████████████████████▌   | 1036/1184 [38:42<05:38,  2.29s/it][A
    Iteration:  88%|████████████████████████▌   | 1037/1184 [38:44<05:34,  2.28s/it][A
    Iteration:  88%|████████████████████████▌   | 1038/1184 [38:46<05:30,  2.26s/it][A
    Iteration:  88%|████████████████████████▌   | 1039/1184 [38:49<05:31,  2.28s/it][A
    Iteration:  88%|████████████████████████▌   | 1040/1184 [38:51<05:26,  2.27s/it][A
    Iteration:  88%|████████████████████████▌   | 1041/1184 [38:53<05:23,  2.26s/it][A
    Iteration:  88%|████████████████████████▋   | 1042/1184 [38:55<05:20,  2.25s/it][A
    Iteration:  88%|████████████████████████▋   | 1043/1184 [38:58<05:19,  2.27s/it][A
    Iteration:  88%|████████████████████████▋   | 1044/1184 [39:00<05:19,  2.28s/it][A
    Iteration:  88%|████████████████████████▋   | 1045/1184 [39:02<05:15,  2.27s/it][A
    Iteration:  88%|████████████████████████▋   | 1046/1184 [39:04<05:12,  2.26s/it][A
    Iteration:  88%|████████████████████████▊   | 1047/1184 [39:07<05:08,  2.25s/it][A
    Iteration:  89%|████████████████████████▊   | 1048/1184 [39:09<05:06,  2.25s/it][A
    Iteration:  89%|████████████████████████▊   | 1049/1184 [39:11<05:03,  2.25s/it][A
    Iteration:  89%|████████████████████████▊   | 1050/1184 [39:13<05:00,  2.25s/it][A
    Iteration:  89%|████████████████████████▊   | 1051/1184 [39:16<04:59,  2.25s/it][A
    Iteration:  89%|████████████████████████▉   | 1052/1184 [39:18<04:56,  2.25s/it][A
    Iteration:  89%|████████████████████████▉   | 1053/1184 [39:20<04:53,  2.24s/it][A
    Iteration:  89%|████████████████████████▉   | 1054/1184 [39:22<04:54,  2.26s/it][A
    Iteration:  89%|████████████████████████▉   | 1055/1184 [39:25<04:54,  2.28s/it][A
    Iteration:  89%|████████████████████████▉   | 1056/1184 [39:27<05:01,  2.36s/it][A
    Iteration:  89%|████████████████████████▉   | 1057/1184 [39:30<05:00,  2.36s/it][A
    Iteration:  89%|█████████████████████████   | 1058/1184 [39:32<04:51,  2.32s/it][A
    Iteration:  89%|█████████████████████████   | 1059/1184 [39:34<04:42,  2.26s/it][A
    Iteration:  90%|█████████████████████████   | 1060/1184 [39:36<04:36,  2.23s/it][A
    Iteration:  90%|█████████████████████████   | 1061/1184 [39:38<04:31,  2.20s/it][A
    Iteration:  90%|█████████████████████████   | 1062/1184 [39:40<04:26,  2.19s/it][A
    Iteration:  90%|█████████████████████████▏  | 1063/1184 [39:43<04:26,  2.20s/it][A
    Iteration:  90%|█████████████████████████▏  | 1064/1184 [39:45<04:25,  2.21s/it][A
    Iteration:  90%|█████████████████████████▏  | 1065/1184 [39:47<04:22,  2.21s/it][A
    Iteration:  90%|█████████████████████████▏  | 1066/1184 [39:49<04:21,  2.22s/it][A
    Iteration:  90%|█████████████████████████▏  | 1067/1184 [39:52<04:20,  2.23s/it][A
    Iteration:  90%|█████████████████████████▎  | 1068/1184 [39:54<04:19,  2.24s/it][A
    Iteration:  90%|█████████████████████████▎  | 1069/1184 [39:56<04:17,  2.24s/it][A
    Iteration:  90%|█████████████████████████▎  | 1070/1184 [39:58<04:17,  2.26s/it][A
    Iteration:  90%|█████████████████████████▎  | 1071/1184 [40:01<04:14,  2.25s/it][A
    Iteration:  91%|█████████████████████████▎  | 1072/1184 [40:03<04:11,  2.24s/it][A
    Iteration:  91%|█████████████████████████▍  | 1073/1184 [40:05<04:07,  2.23s/it][A
    Iteration:  91%|█████████████████████████▍  | 1074/1184 [40:07<04:04,  2.22s/it][A
    Iteration:  91%|█████████████████████████▍  | 1075/1184 [40:10<04:04,  2.24s/it][A
    Iteration:  91%|█████████████████████████▍  | 1076/1184 [40:12<04:06,  2.28s/it][A
    Iteration:  91%|█████████████████████████▍  | 1077/1184 [40:14<04:02,  2.26s/it][A
    Iteration:  91%|█████████████████████████▍  | 1078/1184 [40:16<03:57,  2.24s/it][A
    Iteration:  91%|█████████████████████████▌  | 1079/1184 [40:19<03:53,  2.22s/it][A
    Iteration:  91%|█████████████████████████▌  | 1080/1184 [40:21<03:49,  2.21s/it][A
    Iteration:  91%|█████████████████████████▌  | 1081/1184 [40:23<03:46,  2.20s/it][A
    Iteration:  91%|█████████████████████████▌  | 1082/1184 [40:25<03:43,  2.19s/it][A
    Iteration:  91%|█████████████████████████▌  | 1083/1184 [40:27<03:40,  2.19s/it][A
    Iteration:  92%|█████████████████████████▋  | 1084/1184 [40:30<03:40,  2.21s/it][A
    Iteration:  92%|█████████████████████████▋  | 1085/1184 [40:32<03:41,  2.24s/it][A
    Iteration:  92%|█████████████████████████▋  | 1086/1184 [40:34<03:40,  2.25s/it][A
    Iteration:  92%|█████████████████████████▋  | 1087/1184 [40:36<03:36,  2.23s/it][A
    Iteration:  92%|█████████████████████████▋  | 1088/1184 [40:38<03:32,  2.21s/it][A
    Iteration:  92%|█████████████████████████▊  | 1089/1184 [40:41<03:30,  2.22s/it][A
    Iteration:  92%|█████████████████████████▊  | 1090/1184 [40:43<03:29,  2.23s/it][A
    Iteration:  92%|█████████████████████████▊  | 1091/1184 [40:45<03:25,  2.21s/it][A
    Iteration:  92%|█████████████████████████▊  | 1092/1184 [40:47<03:24,  2.22s/it][A
    Iteration:  92%|█████████████████████████▊  | 1093/1184 [40:50<03:25,  2.25s/it][A
    Iteration:  92%|█████████████████████████▊  | 1094/1184 [40:52<03:25,  2.28s/it][A
    Iteration:  92%|█████████████████████████▉  | 1095/1184 [40:54<03:22,  2.28s/it][A
    Iteration:  93%|█████████████████████████▉  | 1096/1184 [40:57<03:22,  2.30s/it][A
    Iteration:  93%|█████████████████████████▉  | 1097/1184 [40:59<03:17,  2.26s/it][A
    Iteration:  93%|█████████████████████████▉  | 1098/1184 [41:01<03:14,  2.26s/it][A
    Iteration:  93%|█████████████████████████▉  | 1099/1184 [41:03<03:09,  2.22s/it][A
    Iteration:  93%|██████████████████████████  | 1100/1184 [41:05<03:07,  2.23s/it][A
    Iteration:  93%|██████████████████████████  | 1101/1184 [41:08<03:03,  2.21s/it][A
    Iteration:  93%|██████████████████████████  | 1102/1184 [41:10<03:03,  2.24s/it][A
    Iteration:  93%|██████████████████████████  | 1103/1184 [41:12<03:00,  2.23s/it][A
    Iteration:  93%|██████████████████████████  | 1104/1184 [41:14<02:58,  2.23s/it][A
    Iteration:  93%|██████████████████████████▏ | 1105/1184 [41:17<02:55,  2.23s/it][A
    Iteration:  93%|██████████████████████████▏ | 1106/1184 [41:19<02:53,  2.23s/it][A
    Iteration:  93%|██████████████████████████▏ | 1107/1184 [41:21<02:52,  2.24s/it][A
    Iteration:  94%|██████████████████████████▏ | 1108/1184 [41:23<02:49,  2.23s/it][A
    Iteration:  94%|██████████████████████████▏ | 1109/1184 [41:26<02:46,  2.22s/it][A
    Iteration:  94%|██████████████████████████▎ | 1110/1184 [41:28<02:44,  2.22s/it][A
    Iteration:  94%|██████████████████████████▎ | 1111/1184 [41:30<02:41,  2.22s/it][A
    Iteration:  94%|██████████████████████████▎ | 1112/1184 [41:32<02:39,  2.22s/it][A
    Iteration:  94%|██████████████████████████▎ | 1113/1184 [41:35<02:44,  2.31s/it][A
    Iteration:  94%|██████████████████████████▎ | 1114/1184 [41:37<02:39,  2.28s/it][A
    Iteration:  94%|██████████████████████████▎ | 1115/1184 [41:39<02:35,  2.26s/it][A
    Iteration:  94%|██████████████████████████▍ | 1116/1184 [41:41<02:31,  2.23s/it][A
    Iteration:  94%|██████████████████████████▍ | 1117/1184 [41:43<02:28,  2.21s/it][A
    Iteration:  94%|██████████████████████████▍ | 1118/1184 [41:46<02:25,  2.20s/it][A
    Iteration:  95%|██████████████████████████▍ | 1119/1184 [41:48<02:22,  2.20s/it][A
    Iteration:  95%|██████████████████████████▍ | 1120/1184 [41:50<02:20,  2.19s/it][A
    Iteration:  95%|██████████████████████████▌ | 1121/1184 [41:52<02:17,  2.18s/it][A
    Iteration:  95%|██████████████████████████▌ | 1122/1184 [41:54<02:15,  2.18s/it][A
    Iteration:  95%|██████████████████████████▌ | 1123/1184 [41:56<02:12,  2.18s/it][A
    Iteration:  95%|██████████████████████████▌ | 1124/1184 [41:59<02:10,  2.17s/it][A
    Iteration:  95%|██████████████████████████▌ | 1125/1184 [42:01<02:07,  2.17s/it][A
    Iteration:  95%|██████████████████████████▋ | 1126/1184 [42:03<02:06,  2.19s/it][A
    Iteration:  95%|██████████████████████████▋ | 1127/1184 [42:05<02:05,  2.20s/it][A
    Iteration:  95%|██████████████████████████▋ | 1128/1184 [42:08<02:04,  2.23s/it][A
    Iteration:  95%|██████████████████████████▋ | 1129/1184 [42:10<02:04,  2.26s/it][A
    Iteration:  95%|██████████████████████████▋ | 1130/1184 [42:12<02:02,  2.27s/it][A
    Iteration:  96%|██████████████████████████▋ | 1131/1184 [42:14<01:59,  2.25s/it][A
    Iteration:  96%|██████████████████████████▊ | 1132/1184 [42:17<01:56,  2.25s/it][A
    Iteration:  96%|██████████████████████████▊ | 1133/1184 [42:19<01:54,  2.24s/it][A
    Iteration:  96%|██████████████████████████▊ | 1134/1184 [42:21<01:51,  2.24s/it][A
    Iteration:  96%|██████████████████████████▊ | 1135/1184 [42:23<01:49,  2.24s/it][A
    Iteration:  96%|██████████████████████████▊ | 1136/1184 [42:26<01:46,  2.22s/it][A
    Iteration:  96%|██████████████████████████▉ | 1137/1184 [42:28<01:44,  2.23s/it][A
    Iteration:  96%|██████████████████████████▉ | 1138/1184 [42:30<01:42,  2.23s/it][A
    Iteration:  96%|██████████████████████████▉ | 1139/1184 [42:32<01:40,  2.23s/it][A
    Iteration:  96%|██████████████████████████▉ | 1140/1184 [42:34<01:38,  2.24s/it][A
    Iteration:  96%|██████████████████████████▉ | 1141/1184 [42:37<01:40,  2.33s/it][A
    Iteration:  96%|███████████████████████████ | 1142/1184 [42:39<01:35,  2.29s/it][A
    Iteration:  97%|███████████████████████████ | 1143/1184 [42:41<01:33,  2.27s/it][A
    Iteration:  97%|███████████████████████████ | 1144/1184 [42:44<01:35,  2.40s/it][A
    Iteration:  97%|███████████████████████████ | 1145/1184 [42:46<01:30,  2.33s/it][A
    Iteration:  97%|███████████████████████████ | 1146/1184 [42:48<01:26,  2.29s/it][A
    Iteration:  97%|███████████████████████████▏| 1147/1184 [42:51<01:24,  2.28s/it][A
    Iteration:  97%|███████████████████████████▏| 1148/1184 [42:53<01:21,  2.25s/it][A
    Iteration:  97%|███████████████████████████▏| 1149/1184 [42:55<01:17,  2.22s/it][A
    Iteration:  97%|███████████████████████████▏| 1150/1184 [42:57<01:14,  2.20s/it][A
    Iteration:  97%|███████████████████████████▏| 1151/1184 [43:00<01:13,  2.24s/it][A
    Iteration:  97%|███████████████████████████▏| 1152/1184 [43:02<01:12,  2.26s/it][A
    Iteration:  97%|███████████████████████████▎| 1153/1184 [43:04<01:09,  2.26s/it][A
    Iteration:  97%|███████████████████████████▎| 1154/1184 [43:06<01:08,  2.27s/it][A
    Iteration:  98%|███████████████████████████▎| 1155/1184 [43:09<01:05,  2.26s/it][A
    Iteration:  98%|███████████████████████████▎| 1156/1184 [43:11<01:02,  2.25s/it][A
    Iteration:  98%|███████████████████████████▎| 1157/1184 [43:13<01:00,  2.24s/it][A
    Iteration:  98%|███████████████████████████▍| 1158/1184 [43:15<00:57,  2.22s/it][A
    Iteration:  98%|███████████████████████████▍| 1159/1184 [43:18<00:55,  2.23s/it][A
    Iteration:  98%|███████████████████████████▍| 1160/1184 [43:20<00:53,  2.23s/it][A
    Iteration:  98%|███████████████████████████▍| 1161/1184 [43:22<00:51,  2.23s/it][A
    Iteration:  98%|███████████████████████████▍| 1162/1184 [43:24<00:49,  2.26s/it][A
    Iteration:  98%|███████████████████████████▌| 1163/1184 [43:27<00:47,  2.25s/it][A
    Iteration:  98%|███████████████████████████▌| 1164/1184 [43:29<00:44,  2.24s/it][A
    Iteration:  98%|███████████████████████████▌| 1165/1184 [43:31<00:42,  2.24s/it][A
    Iteration:  98%|███████████████████████████▌| 1166/1184 [43:33<00:40,  2.24s/it][A
    Iteration:  99%|███████████████████████████▌| 1167/1184 [43:35<00:37,  2.22s/it][A
    Iteration:  99%|███████████████████████████▌| 1168/1184 [43:38<00:35,  2.23s/it][A
    Iteration:  99%|███████████████████████████▋| 1169/1184 [43:40<00:33,  2.25s/it][A
    Iteration:  99%|███████████████████████████▋| 1170/1184 [43:42<00:31,  2.26s/it][A
    Iteration:  99%|███████████████████████████▋| 1171/1184 [43:45<00:29,  2.26s/it][A
    Iteration:  99%|███████████████████████████▋| 1172/1184 [43:47<00:26,  2.25s/it][A
    Iteration:  99%|███████████████████████████▋| 1173/1184 [43:49<00:24,  2.23s/it][A
    Iteration:  99%|███████████████████████████▊| 1174/1184 [43:51<00:22,  2.21s/it][A
    Iteration:  99%|███████████████████████████▊| 1175/1184 [43:53<00:19,  2.21s/it][A
    Iteration:  99%|███████████████████████████▊| 1176/1184 [43:56<00:17,  2.21s/it][A
    Iteration:  99%|███████████████████████████▊| 1177/1184 [43:58<00:15,  2.20s/it][A
    Iteration:  99%|███████████████████████████▊| 1178/1184 [44:00<00:13,  2.24s/it][A
    Iteration: 100%|███████████████████████████▉| 1179/1184 [44:02<00:11,  2.25s/it][A
    Iteration: 100%|███████████████████████████▉| 1180/1184 [44:05<00:09,  2.28s/it][A
    Iteration: 100%|███████████████████████████▉| 1181/1184 [44:07<00:06,  2.29s/it][A
    Iteration: 100%|███████████████████████████▉| 1182/1184 [44:09<00:04,  2.28s/it][A
    Iteration: 100%|███████████████████████████▉| 1183/1184 [44:11<00:02,  2.26s/it][A
    Iteration: 100%|████████████████████████████| 1184/1184 [44:14<00:00,  2.24s/it][A
    Epoch:  50%|██████████████████                  | 1/2 [44:14<44:14, 2654.25s/it]
    Iteration:   0%|                                       | 0/1184 [00:00<?, ?it/s][A
    Iteration:   0%|                               | 1/1184 [00:02<44:23,  2.25s/it][A
    Iteration:   0%|                               | 2/1184 [00:04<44:22,  2.25s/it][A
    Iteration:   0%|                               | 3/1184 [00:06<44:21,  2.25s/it][A
    Iteration:   0%|                               | 4/1184 [00:09<44:24,  2.26s/it][A
    Iteration:   0%|▏                              | 5/1184 [00:11<44:18,  2.26s/it][A
    Iteration:   1%|▏                              | 6/1184 [00:13<44:38,  2.27s/it][A
    Iteration:   1%|▏                              | 7/1184 [00:15<44:29,  2.27s/it][A
    Iteration:   1%|▏                              | 8/1184 [00:18<44:51,  2.29s/it][A
    Iteration:   1%|▏                              | 9/1184 [00:20<44:24,  2.27s/it][A
    Iteration:   1%|▎                             | 10/1184 [00:22<43:55,  2.25s/it][A
    Iteration:   1%|▎                             | 11/1184 [00:24<43:51,  2.24s/it][A
    Iteration:   1%|▎                             | 12/1184 [00:27<43:43,  2.24s/it][A
    Iteration:   1%|▎                             | 13/1184 [00:29<44:17,  2.27s/it][A
    Iteration:   1%|▎                             | 14/1184 [00:32<46:41,  2.39s/it][A
    Iteration:   1%|▍                             | 15/1184 [00:34<45:26,  2.33s/it][A
    Iteration:   1%|▍                             | 16/1184 [00:36<46:39,  2.40s/it][A
    Iteration:   1%|▍                             | 17/1184 [00:39<45:30,  2.34s/it][A
    Iteration:   2%|▍                             | 18/1184 [00:41<44:33,  2.29s/it][A
    Iteration:   2%|▍                             | 19/1184 [00:43<43:43,  2.25s/it][A
    Iteration:   2%|▌                             | 20/1184 [00:45<43:09,  2.22s/it][A
    Iteration:   2%|▌                             | 21/1184 [00:47<42:44,  2.21s/it][A
    Iteration:   2%|▌                             | 22/1184 [00:49<42:43,  2.21s/it][A
    Iteration:   2%|▌                             | 23/1184 [00:52<42:47,  2.21s/it][A
    Iteration:   2%|▌                             | 24/1184 [00:54<43:26,  2.25s/it][A
    Iteration:   2%|▋                             | 25/1184 [00:56<42:59,  2.23s/it][A
    Iteration:   2%|▋                             | 26/1184 [00:58<43:03,  2.23s/it][A
    Iteration:   2%|▋                             | 27/1184 [01:01<42:38,  2.21s/it][A
    Iteration:   2%|▋                             | 28/1184 [01:03<42:21,  2.20s/it][A
    Iteration:   2%|▋                             | 29/1184 [01:05<42:12,  2.19s/it][A
    Iteration:   3%|▊                             | 30/1184 [01:07<43:03,  2.24s/it][A
    Iteration:   3%|▊                             | 31/1184 [01:10<43:35,  2.27s/it][A
    Iteration:   3%|▊                             | 32/1184 [01:12<43:38,  2.27s/it][A
    Iteration:   3%|▊                             | 33/1184 [01:14<43:45,  2.28s/it][A
    Iteration:   3%|▊                             | 34/1184 [01:16<43:49,  2.29s/it][A
    Iteration:   3%|▉                             | 35/1184 [01:19<43:50,  2.29s/it][A
    Iteration:   3%|▉                             | 36/1184 [01:21<43:44,  2.29s/it][A
    Iteration:   3%|▉                             | 37/1184 [01:23<43:57,  2.30s/it][A
    Iteration:   3%|▉                             | 38/1184 [01:26<43:39,  2.29s/it][A
    Iteration:   3%|▉                             | 39/1184 [01:28<43:20,  2.27s/it][A
    Iteration:   3%|█                             | 40/1184 [01:30<43:33,  2.28s/it][A
    Iteration:   3%|█                             | 41/1184 [01:32<43:25,  2.28s/it][A
    Iteration:   4%|█                             | 42/1184 [01:35<42:52,  2.25s/it][A
    Iteration:   4%|█                             | 43/1184 [01:37<42:20,  2.23s/it][A
    Iteration:   4%|█                             | 44/1184 [01:39<42:25,  2.23s/it][A
    Iteration:   4%|█▏                            | 45/1184 [01:41<42:08,  2.22s/it][A
    Iteration:   4%|█▏                            | 46/1184 [01:43<41:48,  2.20s/it][A
    Iteration:   4%|█▏                            | 47/1184 [01:46<41:33,  2.19s/it][A
    Iteration:   4%|█▏                            | 48/1184 [01:48<41:21,  2.18s/it][A
    Iteration:   4%|█▏                            | 49/1184 [01:50<41:08,  2.17s/it][A
    Iteration:   4%|█▎                            | 50/1184 [01:52<41:02,  2.17s/it][A
    Iteration:   4%|█▎                            | 51/1184 [01:54<41:15,  2.18s/it][A
    Iteration:   4%|█▎                            | 52/1184 [01:56<41:20,  2.19s/it][A
    Iteration:   4%|█▎                            | 53/1184 [01:59<41:24,  2.20s/it][A
    Iteration:   5%|█▎                            | 54/1184 [02:01<41:20,  2.19s/it][A
    Iteration:   5%|█▍                            | 55/1184 [02:03<41:33,  2.21s/it][A
    Iteration:   5%|█▍                            | 56/1184 [02:05<41:40,  2.22s/it][A
    Iteration:   5%|█▍                            | 57/1184 [02:08<41:46,  2.22s/it][A
    Iteration:   5%|█▍                            | 58/1184 [02:10<41:50,  2.23s/it][A
    Iteration:   5%|█▍                            | 59/1184 [02:12<41:51,  2.23s/it][A
    Iteration:   5%|█▌                            | 60/1184 [02:14<42:00,  2.24s/it][A
    Iteration:   5%|█▌                            | 61/1184 [02:17<42:13,  2.26s/it][A
    Iteration:   5%|█▌                            | 62/1184 [02:19<42:16,  2.26s/it][A
    Iteration:   5%|█▌                            | 63/1184 [02:21<42:16,  2.26s/it][A
    Iteration:   5%|█▌                            | 64/1184 [02:23<41:51,  2.24s/it][A
    Iteration:   5%|█▋                            | 65/1184 [02:26<41:48,  2.24s/it][A
    Iteration:   6%|█▋                            | 66/1184 [02:28<42:08,  2.26s/it][A
    Iteration:   6%|█▋                            | 67/1184 [02:30<41:55,  2.25s/it][A
    Iteration:   6%|█▋                            | 68/1184 [02:32<41:48,  2.25s/it][A
    Iteration:   6%|█▋                            | 69/1184 [02:35<41:42,  2.24s/it][A
    Iteration:   6%|█▊                            | 70/1184 [02:37<44:17,  2.39s/it][A
    Iteration:   6%|█▊                            | 71/1184 [02:39<43:06,  2.32s/it][A
    Iteration:   6%|█▊                            | 72/1184 [02:42<42:53,  2.31s/it][A
    Iteration:   6%|█▊                            | 73/1184 [02:44<43:34,  2.35s/it][A
    Iteration:   6%|█▉                            | 74/1184 [02:47<43:17,  2.34s/it][A
    Iteration:   6%|█▉                            | 75/1184 [02:49<42:17,  2.29s/it][A
    Iteration:   6%|█▉                            | 76/1184 [02:51<41:37,  2.25s/it][A
    Iteration:   7%|█▉                            | 77/1184 [02:53<41:30,  2.25s/it][A
    Iteration:   7%|█▉                            | 78/1184 [02:55<40:59,  2.22s/it][A
    Iteration:   7%|██                            | 79/1184 [02:57<40:43,  2.21s/it][A
    Iteration:   7%|██                            | 80/1184 [03:00<40:41,  2.21s/it][A
    Iteration:   7%|██                            | 81/1184 [03:02<41:10,  2.24s/it][A
    Iteration:   7%|██                            | 82/1184 [03:04<40:57,  2.23s/it][A
    Iteration:   7%|██                            | 83/1184 [03:06<40:45,  2.22s/it][A
    Iteration:   7%|██▏                           | 84/1184 [03:09<40:31,  2.21s/it][A
    Iteration:   7%|██▏                           | 85/1184 [03:11<40:21,  2.20s/it][A
    Iteration:   7%|██▏                           | 86/1184 [03:13<40:06,  2.19s/it][A
    Iteration:   7%|██▏                           | 87/1184 [03:15<39:56,  2.18s/it][A
    Iteration:   7%|██▏                           | 88/1184 [03:17<39:48,  2.18s/it][A
    Iteration:   8%|██▎                           | 89/1184 [03:19<40:06,  2.20s/it][A
    Iteration:   8%|██▎                           | 90/1184 [03:22<40:15,  2.21s/it][A
    Iteration:   8%|██▎                           | 91/1184 [03:24<40:07,  2.20s/it][A
    Iteration:   8%|██▎                           | 92/1184 [03:26<39:41,  2.18s/it][A
    Iteration:   8%|██▎                           | 93/1184 [03:28<39:26,  2.17s/it][A
    Iteration:   8%|██▍                           | 94/1184 [03:30<39:16,  2.16s/it][A
    Iteration:   8%|██▍                           | 95/1184 [03:33<39:16,  2.16s/it][A
    Iteration:   8%|██▍                           | 96/1184 [03:35<39:15,  2.16s/it][A
    Iteration:   8%|██▍                           | 97/1184 [03:37<39:06,  2.16s/it][A
    Iteration:   8%|██▍                           | 98/1184 [03:39<39:19,  2.17s/it][A
    Iteration:   8%|██▌                           | 99/1184 [03:41<39:36,  2.19s/it][A
    Iteration:   8%|██▍                          | 100/1184 [03:43<39:43,  2.20s/it][A
    Iteration:   9%|██▍                          | 101/1184 [03:46<39:55,  2.21s/it][A
    Iteration:   9%|██▍                          | 102/1184 [03:48<40:02,  2.22s/it][A
    Iteration:   9%|██▌                          | 103/1184 [03:50<40:26,  2.24s/it][A
    Iteration:   9%|██▌                          | 104/1184 [03:52<40:21,  2.24s/it][A
    Iteration:   9%|██▌                          | 105/1184 [03:55<39:55,  2.22s/it][A
    Iteration:   9%|██▌                          | 106/1184 [03:57<40:20,  2.25s/it][A
    Iteration:   9%|██▌                          | 107/1184 [03:59<40:32,  2.26s/it][A
    Iteration:   9%|██▋                          | 108/1184 [04:02<40:27,  2.26s/it][A
    Iteration:   9%|██▋                          | 109/1184 [04:04<40:04,  2.24s/it][A
    Iteration:   9%|██▋                          | 110/1184 [04:06<39:36,  2.21s/it][A
    Iteration:   9%|██▋                          | 111/1184 [04:08<39:10,  2.19s/it][A
    Iteration:   9%|██▋                          | 112/1184 [04:10<38:54,  2.18s/it][A
    Iteration:  10%|██▊                          | 113/1184 [04:12<38:42,  2.17s/it][A
    Iteration:  10%|██▊                          | 114/1184 [04:15<39:08,  2.19s/it][A
    Iteration:  10%|██▊                          | 115/1184 [04:17<39:02,  2.19s/it][A
    Iteration:  10%|██▊                          | 116/1184 [04:19<38:55,  2.19s/it][A
    Iteration:  10%|██▊                          | 117/1184 [04:21<39:14,  2.21s/it][A
    Iteration:  10%|██▉                          | 118/1184 [04:23<39:36,  2.23s/it][A
    Iteration:  10%|██▉                          | 119/1184 [04:26<39:24,  2.22s/it][A
    Iteration:  10%|██▉                          | 120/1184 [04:28<39:16,  2.22s/it][A
    Iteration:  10%|██▉                          | 121/1184 [04:30<39:31,  2.23s/it][A
    Iteration:  10%|██▉                          | 122/1184 [04:32<39:19,  2.22s/it][A
    Iteration:  10%|███                          | 123/1184 [04:35<39:23,  2.23s/it][A
    Iteration:  10%|███                          | 124/1184 [04:37<39:25,  2.23s/it][A
    Iteration:  11%|███                          | 125/1184 [04:39<39:27,  2.24s/it][A
    Iteration:  11%|███                          | 126/1184 [04:41<39:31,  2.24s/it][A
    Iteration:  11%|███                          | 127/1184 [04:44<41:26,  2.35s/it][A
    Iteration:  11%|███▏                         | 128/1184 [04:46<40:46,  2.32s/it][A
    Iteration:  11%|███▏                         | 129/1184 [04:48<40:13,  2.29s/it][A
    Iteration:  11%|███▏                         | 130/1184 [04:51<40:01,  2.28s/it][A
    Iteration:  11%|███▏                         | 131/1184 [04:53<39:33,  2.25s/it][A
    Iteration:  11%|███▏                         | 132/1184 [04:55<39:19,  2.24s/it][A
    Iteration:  11%|███▎                         | 133/1184 [04:57<38:52,  2.22s/it][A
    Iteration:  11%|███▎                         | 134/1184 [04:59<38:32,  2.20s/it][A
    Iteration:  11%|███▎                         | 135/1184 [05:02<38:42,  2.21s/it][A
    Iteration:  11%|███▎                         | 136/1184 [05:04<38:49,  2.22s/it][A
    Iteration:  12%|███▎                         | 137/1184 [05:06<38:56,  2.23s/it][A
    Iteration:  12%|███▍                         | 138/1184 [05:08<38:58,  2.24s/it][A
    Iteration:  12%|███▍                         | 139/1184 [05:11<38:58,  2.24s/it][A
    Iteration:  12%|███▍                         | 140/1184 [05:13<38:57,  2.24s/it][A
    Iteration:  12%|███▍                         | 141/1184 [05:15<38:53,  2.24s/it][A
    Iteration:  12%|███▍                         | 142/1184 [05:17<38:53,  2.24s/it][A
    Iteration:  12%|███▌                         | 143/1184 [05:20<38:57,  2.25s/it][A
    Iteration:  12%|███▌                         | 144/1184 [05:22<38:53,  2.24s/it][A
    Iteration:  12%|███▌                         | 145/1184 [05:24<38:52,  2.25s/it][A
    Iteration:  12%|███▌                         | 146/1184 [05:26<38:47,  2.24s/it][A
    Iteration:  12%|███▌                         | 147/1184 [05:29<38:44,  2.24s/it][A
    Iteration:  12%|███▋                         | 148/1184 [05:31<38:40,  2.24s/it][A
    Iteration:  13%|███▋                         | 149/1184 [05:33<38:35,  2.24s/it][A
    Iteration:  13%|███▋                         | 150/1184 [05:35<38:34,  2.24s/it][A
    Iteration:  13%|███▋                         | 151/1184 [05:37<38:33,  2.24s/it][A
    Iteration:  13%|███▋                         | 152/1184 [05:40<38:27,  2.24s/it][A
    Iteration:  13%|███▋                         | 153/1184 [05:42<38:24,  2.24s/it][A
    Iteration:  13%|███▊                         | 154/1184 [05:44<38:18,  2.23s/it][A
    Iteration:  13%|███▊                         | 155/1184 [05:46<38:26,  2.24s/it][A
    Iteration:  13%|███▊                         | 156/1184 [05:49<38:02,  2.22s/it][A
    Iteration:  13%|███▊                         | 157/1184 [05:51<38:09,  2.23s/it][A
    Iteration:  13%|███▊                         | 158/1184 [05:53<39:08,  2.29s/it][A
    Iteration:  13%|███▉                         | 159/1184 [05:56<38:53,  2.28s/it][A
    Iteration:  14%|███▉                         | 160/1184 [05:58<38:22,  2.25s/it][A
    Iteration:  14%|███▉                         | 161/1184 [06:00<38:33,  2.26s/it][A
    Iteration:  14%|███▉                         | 162/1184 [06:02<38:24,  2.25s/it][A
    Iteration:  14%|███▉                         | 163/1184 [06:05<38:39,  2.27s/it][A
    Iteration:  14%|████                         | 164/1184 [06:07<38:14,  2.25s/it][A
    Iteration:  14%|████                         | 165/1184 [06:09<37:38,  2.22s/it][A
    Iteration:  14%|████                         | 166/1184 [06:11<37:55,  2.24s/it][A
    Iteration:  14%|████                         | 167/1184 [06:13<37:52,  2.23s/it][A
    Iteration:  14%|████                         | 168/1184 [06:16<37:52,  2.24s/it][A
    Iteration:  14%|████▏                        | 169/1184 [06:18<37:55,  2.24s/it][A
    Iteration:  14%|████▏                        | 170/1184 [06:20<37:54,  2.24s/it][A
    Iteration:  14%|████▏                        | 171/1184 [06:22<37:57,  2.25s/it][A
    Iteration:  15%|████▏                        | 172/1184 [06:25<37:54,  2.25s/it][A
    Iteration:  15%|████▏                        | 173/1184 [06:27<37:35,  2.23s/it][A
    Iteration:  15%|████▎                        | 174/1184 [06:29<37:08,  2.21s/it][A
    Iteration:  15%|████▎                        | 175/1184 [06:31<36:49,  2.19s/it][A
    Iteration:  15%|████▎                        | 176/1184 [06:33<36:48,  2.19s/it][A
    Iteration:  15%|████▎                        | 177/1184 [06:36<36:41,  2.19s/it][A
    Iteration:  15%|████▎                        | 178/1184 [06:38<37:13,  2.22s/it][A
    Iteration:  15%|████▍                        | 179/1184 [06:40<37:09,  2.22s/it][A
    Iteration:  15%|████▍                        | 180/1184 [06:42<38:01,  2.27s/it][A
    Iteration:  15%|████▍                        | 181/1184 [06:45<38:17,  2.29s/it][A
    Iteration:  15%|████▍                        | 182/1184 [06:47<38:18,  2.29s/it][A
    Iteration:  15%|████▍                        | 183/1184 [06:49<37:20,  2.24s/it][A
    Iteration:  16%|████▌                        | 184/1184 [06:51<37:05,  2.23s/it][A
    Iteration:  16%|████▌                        | 185/1184 [06:54<36:49,  2.21s/it][A
    Iteration:  16%|████▌                        | 186/1184 [06:56<37:15,  2.24s/it][A
    Iteration:  16%|████▌                        | 187/1184 [06:58<37:04,  2.23s/it][A
    Iteration:  16%|████▌                        | 188/1184 [07:00<36:41,  2.21s/it][A
    Iteration:  16%|████▋                        | 189/1184 [07:02<36:35,  2.21s/it][A
    Iteration:  16%|████▋                        | 190/1184 [07:05<36:30,  2.20s/it][A
    Iteration:  16%|████▋                        | 191/1184 [07:07<36:09,  2.19s/it][A
    Iteration:  16%|████▋                        | 192/1184 [07:09<36:23,  2.20s/it][A
    Iteration:  16%|████▋                        | 193/1184 [07:11<36:15,  2.20s/it][A
    Iteration:  16%|████▊                        | 194/1184 [07:13<36:05,  2.19s/it][A
    Iteration:  16%|████▊                        | 195/1184 [07:16<36:27,  2.21s/it][A
    Iteration:  17%|████▊                        | 196/1184 [07:18<36:39,  2.23s/it][A
    Iteration:  17%|████▊                        | 197/1184 [07:20<36:54,  2.24s/it][A
    Iteration:  17%|████▊                        | 198/1184 [07:22<36:56,  2.25s/it][A
    Iteration:  17%|████▊                        | 199/1184 [07:25<36:48,  2.24s/it][A
    Iteration:  17%|████▉                        | 200/1184 [07:27<36:47,  2.24s/it][A
    Iteration:  17%|████▉                        | 201/1184 [07:29<37:02,  2.26s/it][A
    Iteration:  17%|████▉                        | 202/1184 [07:31<37:01,  2.26s/it][A
    Iteration:  17%|████▉                        | 203/1184 [07:34<37:01,  2.26s/it][A
    Iteration:  17%|████▉                        | 204/1184 [07:36<37:02,  2.27s/it][A
    Iteration:  17%|█████                        | 205/1184 [07:38<36:48,  2.26s/it][A
    Iteration:  17%|█████                        | 206/1184 [07:40<36:28,  2.24s/it][A
    Iteration:  17%|█████                        | 207/1184 [07:43<36:04,  2.21s/it][A
    Iteration:  18%|█████                        | 208/1184 [07:45<36:01,  2.21s/it][A
    Iteration:  18%|█████                        | 209/1184 [07:47<35:41,  2.20s/it][A
    Iteration:  18%|█████▏                       | 210/1184 [07:49<35:26,  2.18s/it][A
    Iteration:  18%|█████▏                       | 211/1184 [07:51<35:48,  2.21s/it][A
    Iteration:  18%|█████▏                       | 212/1184 [07:54<35:50,  2.21s/it][A
    Iteration:  18%|█████▏                       | 213/1184 [07:56<35:43,  2.21s/it][A
    Iteration:  18%|█████▏                       | 214/1184 [07:58<35:48,  2.21s/it][A
    Iteration:  18%|█████▎                       | 215/1184 [08:00<35:43,  2.21s/it][A
    Iteration:  18%|█████▎                       | 216/1184 [08:02<35:31,  2.20s/it][A
    Iteration:  18%|█████▎                       | 217/1184 [08:05<35:30,  2.20s/it][A
    Iteration:  18%|█████▎                       | 218/1184 [08:07<35:13,  2.19s/it][A
    Iteration:  18%|█████▎                       | 219/1184 [08:09<35:19,  2.20s/it][A
    Iteration:  19%|█████▍                       | 220/1184 [08:11<35:13,  2.19s/it][A
    Iteration:  19%|█████▍                       | 221/1184 [08:13<35:08,  2.19s/it][A
    Iteration:  19%|█████▍                       | 222/1184 [08:15<34:54,  2.18s/it][A
    Iteration:  19%|█████▍                       | 223/1184 [08:18<34:56,  2.18s/it][A
    Iteration:  19%|█████▍                       | 224/1184 [08:20<34:52,  2.18s/it][A
    Iteration:  19%|█████▌                       | 225/1184 [08:22<35:19,  2.21s/it][A
    Iteration:  19%|█████▌                       | 226/1184 [08:24<35:32,  2.23s/it][A
    Iteration:  19%|█████▌                       | 227/1184 [08:27<35:43,  2.24s/it][A
    Iteration:  19%|█████▌                       | 228/1184 [08:29<35:22,  2.22s/it][A
    Iteration:  19%|█████▌                       | 229/1184 [08:31<35:19,  2.22s/it][A
    Iteration:  19%|█████▋                       | 230/1184 [08:33<35:17,  2.22s/it][A
    Iteration:  20%|█████▋                       | 231/1184 [08:36<35:16,  2.22s/it][A
    Iteration:  20%|█████▋                       | 232/1184 [08:38<35:03,  2.21s/it][A
    Iteration:  20%|█████▋                       | 233/1184 [08:40<35:07,  2.22s/it][A
    Iteration:  20%|█████▋                       | 234/1184 [08:42<35:08,  2.22s/it][A
    Iteration:  20%|█████▊                       | 235/1184 [08:44<35:25,  2.24s/it][A
    Iteration:  20%|█████▊                       | 236/1184 [08:47<35:25,  2.24s/it][A
    Iteration:  20%|█████▊                       | 237/1184 [08:49<35:24,  2.24s/it][A
    Iteration:  20%|█████▊                       | 238/1184 [08:51<35:24,  2.25s/it][A
    Iteration:  20%|█████▊                       | 239/1184 [08:53<35:21,  2.24s/it][A
    Iteration:  20%|█████▉                       | 240/1184 [08:56<35:17,  2.24s/it][A
    Iteration:  20%|█████▉                       | 241/1184 [08:58<35:14,  2.24s/it][A
    Iteration:  20%|█████▉                       | 242/1184 [09:00<35:12,  2.24s/it][A
    Iteration:  21%|█████▉                       | 243/1184 [09:02<35:17,  2.25s/it][A
    Iteration:  21%|█████▉                       | 244/1184 [09:05<35:23,  2.26s/it][A
    Iteration:  21%|██████                       | 245/1184 [09:07<35:18,  2.26s/it][A
    Iteration:  21%|██████                       | 246/1184 [09:09<35:11,  2.25s/it][A
    Iteration:  21%|██████                       | 247/1184 [09:12<35:39,  2.28s/it][A
    Iteration:  21%|██████                       | 248/1184 [09:14<35:30,  2.28s/it][A
    Iteration:  21%|██████                       | 249/1184 [09:16<35:13,  2.26s/it][A
    Iteration:  21%|██████                       | 250/1184 [09:18<35:18,  2.27s/it][A
    Iteration:  21%|██████▏                      | 251/1184 [09:21<35:06,  2.26s/it][A
    Iteration:  21%|██████▏                      | 252/1184 [09:23<35:15,  2.27s/it][A
    Iteration:  21%|██████▏                      | 253/1184 [09:25<34:48,  2.24s/it][A
    Iteration:  21%|██████▏                      | 254/1184 [09:27<34:55,  2.25s/it][A
    Iteration:  22%|██████▏                      | 255/1184 [09:29<34:28,  2.23s/it][A
    Iteration:  22%|██████▎                      | 256/1184 [09:32<34:39,  2.24s/it][A
    Iteration:  22%|██████▎                      | 257/1184 [09:34<34:25,  2.23s/it][A
    Iteration:  22%|██████▎                      | 258/1184 [09:36<34:17,  2.22s/it][A
    Iteration:  22%|██████▎                      | 259/1184 [09:38<33:51,  2.20s/it][A
    Iteration:  22%|██████▎                      | 260/1184 [09:40<33:36,  2.18s/it][A
    Iteration:  22%|██████▍                      | 261/1184 [09:43<33:39,  2.19s/it][A
    Iteration:  22%|██████▍                      | 262/1184 [09:45<33:40,  2.19s/it][A
    Iteration:  22%|██████▍                      | 263/1184 [09:47<33:34,  2.19s/it][A
    Iteration:  22%|██████▍                      | 264/1184 [09:49<33:39,  2.19s/it][A
    Iteration:  22%|██████▍                      | 265/1184 [09:51<33:31,  2.19s/it][A
    Iteration:  22%|██████▌                      | 266/1184 [09:54<33:25,  2.18s/it][A
    Iteration:  23%|██████▌                      | 267/1184 [09:56<33:18,  2.18s/it][A
    Iteration:  23%|██████▌                      | 268/1184 [09:58<33:15,  2.18s/it][A
    Iteration:  23%|██████▌                      | 269/1184 [10:00<33:09,  2.17s/it][A
    Iteration:  23%|██████▌                      | 270/1184 [10:02<33:06,  2.17s/it][A
    Iteration:  23%|██████▋                      | 271/1184 [10:05<35:25,  2.33s/it][A
    Iteration:  23%|██████▋                      | 272/1184 [10:07<34:36,  2.28s/it][A
    Iteration:  23%|██████▋                      | 273/1184 [10:09<33:52,  2.23s/it][A
    Iteration:  23%|██████▋                      | 274/1184 [10:11<33:37,  2.22s/it][A
    Iteration:  23%|██████▋                      | 275/1184 [10:14<33:23,  2.20s/it][A
    Iteration:  23%|██████▊                      | 276/1184 [10:16<33:21,  2.20s/it][A
    Iteration:  23%|██████▊                      | 277/1184 [10:18<33:21,  2.21s/it][A
    Iteration:  23%|██████▊                      | 278/1184 [10:20<33:11,  2.20s/it][A
    Iteration:  24%|██████▊                      | 279/1184 [10:22<33:00,  2.19s/it][A
    Iteration:  24%|██████▊                      | 280/1184 [10:25<33:08,  2.20s/it][A
    Iteration:  24%|██████▉                      | 281/1184 [10:27<33:08,  2.20s/it][A
    Iteration:  24%|██████▉                      | 282/1184 [10:29<32:58,  2.19s/it][A
    Iteration:  24%|██████▉                      | 283/1184 [10:31<32:47,  2.18s/it][A
    Iteration:  24%|██████▉                      | 284/1184 [10:33<32:45,  2.18s/it][A
    Iteration:  24%|██████▉                      | 285/1184 [10:35<32:48,  2.19s/it][A
    Iteration:  24%|███████                      | 286/1184 [10:38<32:28,  2.17s/it][A
    Iteration:  24%|███████                      | 287/1184 [10:40<32:19,  2.16s/it][A
    Iteration:  24%|███████                      | 288/1184 [10:42<32:26,  2.17s/it][A
    Iteration:  24%|███████                      | 289/1184 [10:44<32:17,  2.17s/it][A
    Iteration:  24%|███████                      | 290/1184 [10:46<32:11,  2.16s/it][A
    Iteration:  25%|███████▏                     | 291/1184 [10:48<32:07,  2.16s/it][A
    Iteration:  25%|███████▏                     | 292/1184 [10:51<31:59,  2.15s/it][A
    Iteration:  25%|███████▏                     | 293/1184 [10:53<31:51,  2.15s/it][A
    Iteration:  25%|███████▏                     | 294/1184 [10:55<31:46,  2.14s/it][A
    Iteration:  25%|███████▏                     | 295/1184 [10:57<31:43,  2.14s/it][A
    Iteration:  25%|███████▎                     | 296/1184 [10:59<31:45,  2.15s/it][A
    Iteration:  25%|███████▎                     | 297/1184 [11:01<31:47,  2.15s/it][A
    Iteration:  25%|███████▎                     | 298/1184 [11:03<31:48,  2.15s/it][A
    Iteration:  25%|███████▎                     | 299/1184 [11:06<32:22,  2.20s/it][A
    Iteration:  25%|███████▎                     | 300/1184 [11:08<32:49,  2.23s/it][A
    Iteration:  25%|███████▎                     | 301/1184 [11:10<32:46,  2.23s/it][A
    Iteration:  26%|███████▍                     | 302/1184 [11:12<32:49,  2.23s/it][A
    Iteration:  26%|███████▍                     | 303/1184 [11:15<33:09,  2.26s/it][A
    Iteration:  26%|███████▍                     | 304/1184 [11:17<33:12,  2.26s/it][A
    Iteration:  26%|███████▍                     | 305/1184 [11:19<33:35,  2.29s/it][A
    Iteration:  26%|███████▍                     | 306/1184 [11:22<33:20,  2.28s/it][A
    Iteration:  26%|███████▌                     | 307/1184 [11:24<32:55,  2.25s/it][A
    Iteration:  26%|███████▌                     | 308/1184 [11:26<32:48,  2.25s/it][A
    Iteration:  26%|███████▌                     | 309/1184 [11:28<32:44,  2.25s/it][A
    Iteration:  26%|███████▌                     | 310/1184 [11:31<32:41,  2.24s/it][A
    Iteration:  26%|███████▌                     | 311/1184 [11:33<32:34,  2.24s/it][A
    Iteration:  26%|███████▋                     | 312/1184 [11:35<32:25,  2.23s/it][A
    Iteration:  26%|███████▋                     | 313/1184 [11:37<32:02,  2.21s/it][A
    Iteration:  27%|███████▋                     | 314/1184 [11:39<32:08,  2.22s/it][A
    Iteration:  27%|███████▋                     | 315/1184 [11:42<32:15,  2.23s/it][A
    Iteration:  27%|███████▋                     | 316/1184 [11:44<32:36,  2.25s/it][A
    Iteration:  27%|███████▊                     | 317/1184 [11:46<32:11,  2.23s/it][A
    Iteration:  27%|███████▊                     | 318/1184 [11:48<32:13,  2.23s/it][A
    Iteration:  27%|███████▊                     | 319/1184 [11:51<31:46,  2.20s/it][A
    Iteration:  27%|███████▊                     | 320/1184 [11:53<32:02,  2.22s/it][A
    Iteration:  27%|███████▊                     | 321/1184 [11:55<31:46,  2.21s/it][A
    Iteration:  27%|███████▉                     | 322/1184 [11:57<31:41,  2.21s/it][A
    Iteration:  27%|███████▉                     | 323/1184 [11:59<31:39,  2.21s/it][A
    Iteration:  27%|███████▉                     | 324/1184 [12:02<31:51,  2.22s/it][A
    Iteration:  27%|███████▉                     | 325/1184 [12:03<29:41,  2.07s/it][A
    Iteration:  28%|███████▉                     | 326/1184 [12:06<29:56,  2.09s/it][A
    Iteration:  28%|████████                     | 327/1184 [12:08<30:11,  2.11s/it][A
    Iteration:  28%|████████                     | 328/1184 [12:10<30:22,  2.13s/it][A
    Iteration:  28%|████████                     | 329/1184 [12:12<30:59,  2.18s/it][A
    Iteration:  28%|████████                     | 330/1184 [12:14<31:10,  2.19s/it][A
    Iteration:  28%|████████                     | 331/1184 [12:17<31:26,  2.21s/it][A
    Iteration:  28%|████████▏                    | 332/1184 [12:19<31:19,  2.21s/it][A
    Iteration:  28%|████████▏                    | 333/1184 [12:21<31:00,  2.19s/it][A
    Iteration:  28%|████████▏                    | 334/1184 [12:23<31:03,  2.19s/it][A
    Iteration:  28%|████████▏                    | 335/1184 [12:25<31:01,  2.19s/it][A
    Iteration:  28%|████████▏                    | 336/1184 [12:28<30:53,  2.19s/it][A
    Iteration:  28%|████████▎                    | 337/1184 [12:30<30:48,  2.18s/it][A
    Iteration:  29%|████████▎                    | 338/1184 [12:32<30:41,  2.18s/it][A
    Iteration:  29%|████████▎                    | 339/1184 [12:34<30:33,  2.17s/it][A
    Iteration:  29%|████████▎                    | 340/1184 [12:36<30:49,  2.19s/it][A
    Iteration:  29%|████████▎                    | 341/1184 [12:39<31:08,  2.22s/it][A
    Iteration:  29%|████████▍                    | 342/1184 [12:41<31:21,  2.24s/it][A
    Iteration:  29%|████████▍                    | 343/1184 [12:43<31:24,  2.24s/it][A
    Iteration:  29%|████████▍                    | 344/1184 [12:45<31:28,  2.25s/it][A
    Iteration:  29%|████████▍                    | 345/1184 [12:48<31:34,  2.26s/it][A
    Iteration:  29%|████████▍                    | 346/1184 [12:50<31:49,  2.28s/it][A
    Iteration:  29%|████████▍                    | 347/1184 [12:52<31:35,  2.26s/it][A
    Iteration:  29%|████████▌                    | 348/1184 [12:54<31:48,  2.28s/it][A
    Iteration:  29%|████████▌                    | 349/1184 [12:57<31:37,  2.27s/it][A
    Iteration:  30%|████████▌                    | 350/1184 [12:59<31:27,  2.26s/it][A
    Iteration:  30%|████████▌                    | 351/1184 [13:01<31:17,  2.25s/it][A
    Iteration:  30%|████████▌                    | 352/1184 [13:03<31:15,  2.25s/it][A
    Iteration:  30%|████████▋                    | 353/1184 [13:06<30:59,  2.24s/it][A
    Iteration:  30%|████████▋                    | 354/1184 [13:08<31:03,  2.25s/it][A
    Iteration:  30%|████████▋                    | 355/1184 [13:10<31:05,  2.25s/it][A
    Iteration:  30%|████████▋                    | 356/1184 [13:12<31:04,  2.25s/it][A
    Iteration:  30%|████████▋                    | 357/1184 [13:15<31:33,  2.29s/it][A
    Iteration:  30%|████████▊                    | 358/1184 [13:17<31:25,  2.28s/it][A
    Iteration:  30%|████████▊                    | 359/1184 [13:19<31:04,  2.26s/it][A
    Iteration:  30%|████████▊                    | 360/1184 [13:22<31:33,  2.30s/it][A
    Iteration:  30%|████████▊                    | 361/1184 [13:24<31:12,  2.27s/it][A
    Iteration:  31%|████████▊                    | 362/1184 [13:26<30:43,  2.24s/it][A
    Iteration:  31%|████████▉                    | 363/1184 [13:28<30:40,  2.24s/it][A
    Iteration:  31%|████████▉                    | 364/1184 [13:31<30:43,  2.25s/it][A
    Iteration:  31%|████████▉                    | 365/1184 [13:33<30:29,  2.23s/it][A
    Iteration:  31%|████████▉                    | 366/1184 [13:35<30:25,  2.23s/it][A
    Iteration:  31%|████████▉                    | 367/1184 [13:37<30:25,  2.23s/it][A
    Iteration:  31%|█████████                    | 368/1184 [13:39<30:26,  2.24s/it][A
    Iteration:  31%|█████████                    | 369/1184 [13:42<30:24,  2.24s/it][A
    Iteration:  31%|█████████                    | 370/1184 [13:44<30:25,  2.24s/it][A
    Iteration:  31%|█████████                    | 371/1184 [13:46<30:15,  2.23s/it][A
    Iteration:  31%|█████████                    | 372/1184 [13:48<29:56,  2.21s/it][A
    Iteration:  32%|█████████▏                   | 373/1184 [13:51<29:40,  2.20s/it][A
    Iteration:  32%|█████████▏                   | 374/1184 [13:53<29:26,  2.18s/it][A
    Iteration:  32%|█████████▏                   | 375/1184 [13:55<29:49,  2.21s/it][A
    Iteration:  32%|█████████▏                   | 376/1184 [13:57<29:39,  2.20s/it][A
    Iteration:  32%|█████████▏                   | 377/1184 [13:59<29:29,  2.19s/it][A
    Iteration:  32%|█████████▎                   | 378/1184 [14:02<29:42,  2.21s/it][A
    Iteration:  32%|█████████▎                   | 379/1184 [14:04<29:45,  2.22s/it][A
    Iteration:  32%|█████████▎                   | 380/1184 [14:06<29:50,  2.23s/it][A
    Iteration:  32%|█████████▎                   | 381/1184 [14:08<29:42,  2.22s/it][A
    Iteration:  32%|█████████▎                   | 382/1184 [14:10<29:29,  2.21s/it][A
    Iteration:  32%|█████████▍                   | 383/1184 [14:13<29:25,  2.20s/it][A
    Iteration:  32%|█████████▍                   | 384/1184 [14:15<29:15,  2.19s/it][A
    Iteration:  33%|█████████▍                   | 385/1184 [14:17<29:22,  2.21s/it][A
    Iteration:  33%|█████████▍                   | 386/1184 [14:20<30:34,  2.30s/it][A
    Iteration:  33%|█████████▍                   | 387/1184 [14:22<30:05,  2.27s/it][A
    Iteration:  33%|█████████▌                   | 388/1184 [14:25<32:27,  2.45s/it][A
    Iteration:  33%|█████████▌                   | 389/1184 [14:27<31:33,  2.38s/it][A
    Iteration:  33%|█████████▌                   | 390/1184 [14:29<30:39,  2.32s/it][A
    Iteration:  33%|█████████▌                   | 391/1184 [14:31<30:00,  2.27s/it][A
    Iteration:  33%|█████████▌                   | 392/1184 [14:33<29:32,  2.24s/it][A
    Iteration:  33%|█████████▋                   | 393/1184 [14:36<29:36,  2.25s/it][A
    Iteration:  33%|█████████▋                   | 394/1184 [14:38<29:36,  2.25s/it][A
    Iteration:  33%|█████████▋                   | 395/1184 [14:40<29:58,  2.28s/it][A
    Iteration:  33%|█████████▋                   | 396/1184 [14:42<29:50,  2.27s/it][A
    Iteration:  34%|█████████▋                   | 397/1184 [14:45<29:18,  2.23s/it][A
    Iteration:  34%|█████████▋                   | 398/1184 [14:47<28:57,  2.21s/it][A
    Iteration:  34%|█████████▊                   | 399/1184 [14:49<29:05,  2.22s/it][A
    Iteration:  34%|█████████▊                   | 400/1184 [14:51<28:54,  2.21s/it][A
    Iteration:  34%|█████████▊                   | 401/1184 [14:53<28:44,  2.20s/it][A
    Iteration:  34%|█████████▊                   | 402/1184 [14:56<28:37,  2.20s/it][A
    Iteration:  34%|█████████▊                   | 403/1184 [14:58<28:32,  2.19s/it][A
    Iteration:  34%|█████████▉                   | 404/1184 [15:00<28:53,  2.22s/it][A
    Iteration:  34%|█████████▉                   | 405/1184 [15:02<28:43,  2.21s/it][A
    Iteration:  34%|█████████▉                   | 406/1184 [15:04<28:26,  2.19s/it][A
    Iteration:  34%|█████████▉                   | 407/1184 [15:07<28:15,  2.18s/it][A
    Iteration:  34%|█████████▉                   | 408/1184 [15:09<28:00,  2.17s/it][A
    Iteration:  35%|██████████                   | 409/1184 [15:11<27:58,  2.17s/it][A
    Iteration:  35%|██████████                   | 410/1184 [15:13<27:59,  2.17s/it][A
    Iteration:  35%|██████████                   | 411/1184 [15:15<28:15,  2.19s/it][A
    Iteration:  35%|██████████                   | 412/1184 [15:17<28:29,  2.21s/it][A
    Iteration:  35%|██████████                   | 413/1184 [15:20<28:28,  2.22s/it][A
    Iteration:  35%|██████████▏                  | 414/1184 [15:22<28:25,  2.22s/it][A
    Iteration:  35%|██████████▏                  | 415/1184 [15:24<28:18,  2.21s/it][A
    Iteration:  35%|██████████▏                  | 416/1184 [15:26<28:10,  2.20s/it][A
    Iteration:  35%|██████████▏                  | 417/1184 [15:29<28:24,  2.22s/it][A
    Iteration:  35%|██████████▏                  | 418/1184 [15:31<28:13,  2.21s/it][A
    Iteration:  35%|██████████▎                  | 419/1184 [15:33<28:05,  2.20s/it][A
    Iteration:  35%|██████████▎                  | 420/1184 [15:35<28:03,  2.20s/it][A
    Iteration:  36%|██████████▎                  | 421/1184 [15:37<28:12,  2.22s/it][A
    Iteration:  36%|██████████▎                  | 422/1184 [15:40<28:30,  2.25s/it][A
    Iteration:  36%|██████████▎                  | 423/1184 [15:42<28:49,  2.27s/it][A
    Iteration:  36%|██████████▍                  | 424/1184 [15:44<28:53,  2.28s/it][A
    Iteration:  36%|██████████▍                  | 425/1184 [15:47<28:48,  2.28s/it][A
    Iteration:  36%|██████████▍                  | 426/1184 [15:49<28:28,  2.25s/it][A
    Iteration:  36%|██████████▍                  | 427/1184 [15:51<28:24,  2.25s/it][A
    Iteration:  36%|██████████▍                  | 428/1184 [15:53<28:28,  2.26s/it][A
    Iteration:  36%|██████████▌                  | 429/1184 [15:56<28:42,  2.28s/it][A
    Iteration:  36%|██████████▌                  | 430/1184 [15:58<28:58,  2.31s/it][A
    Iteration:  36%|██████████▌                  | 431/1184 [16:00<28:35,  2.28s/it][A
    Iteration:  36%|██████████▌                  | 432/1184 [16:02<28:22,  2.26s/it][A
    Iteration:  37%|██████████▌                  | 433/1184 [16:05<27:56,  2.23s/it][A
    Iteration:  37%|██████████▋                  | 434/1184 [16:07<27:36,  2.21s/it][A
    Iteration:  37%|██████████▋                  | 435/1184 [16:09<27:31,  2.20s/it][A
    Iteration:  37%|██████████▋                  | 436/1184 [16:11<27:20,  2.19s/it][A
    Iteration:  37%|██████████▋                  | 437/1184 [16:13<27:13,  2.19s/it][A
    Iteration:  37%|██████████▋                  | 438/1184 [16:15<26:58,  2.17s/it][A
    Iteration:  37%|██████████▊                  | 439/1184 [16:18<26:50,  2.16s/it][A
    Iteration:  37%|██████████▊                  | 440/1184 [16:20<26:52,  2.17s/it][A
    Iteration:  37%|██████████▊                  | 441/1184 [16:22<26:51,  2.17s/it][A
    Iteration:  37%|██████████▊                  | 442/1184 [16:24<26:55,  2.18s/it][A
    Iteration:  37%|██████████▊                  | 443/1184 [16:26<26:47,  2.17s/it][A
    Iteration:  38%|██████████▉                  | 444/1184 [16:28<26:36,  2.16s/it][A
    Iteration:  38%|██████████▉                  | 445/1184 [16:31<26:35,  2.16s/it][A
    Iteration:  38%|██████████▉                  | 446/1184 [16:33<26:47,  2.18s/it][A
    Iteration:  38%|██████████▉                  | 447/1184 [16:35<27:08,  2.21s/it][A
    Iteration:  38%|██████████▉                  | 448/1184 [16:37<27:07,  2.21s/it][A
    Iteration:  38%|██████████▉                  | 449/1184 [16:40<27:05,  2.21s/it][A
    Iteration:  38%|███████████                  | 450/1184 [16:42<27:05,  2.21s/it][A
    Iteration:  38%|███████████                  | 451/1184 [16:44<27:01,  2.21s/it][A
    Iteration:  38%|███████████                  | 452/1184 [16:46<26:53,  2.20s/it][A
    Iteration:  38%|███████████                  | 453/1184 [16:48<26:40,  2.19s/it][A
    Iteration:  38%|███████████                  | 454/1184 [16:50<26:33,  2.18s/it][A
    Iteration:  38%|███████████▏                 | 455/1184 [16:53<26:28,  2.18s/it][A
    Iteration:  39%|███████████▏                 | 456/1184 [16:55<26:53,  2.22s/it][A
    Iteration:  39%|███████████▏                 | 457/1184 [16:57<26:44,  2.21s/it][A
    Iteration:  39%|███████████▏                 | 458/1184 [16:59<26:56,  2.23s/it][A
    Iteration:  39%|███████████▏                 | 459/1184 [17:02<27:24,  2.27s/it][A
    Iteration:  39%|███████████▎                 | 460/1184 [17:04<27:11,  2.25s/it][A
    Iteration:  39%|███████████▎                 | 461/1184 [17:06<27:29,  2.28s/it][A
    Iteration:  39%|███████████▎                 | 462/1184 [17:09<27:40,  2.30s/it][A
    Iteration:  39%|███████████▎                 | 463/1184 [17:11<27:16,  2.27s/it][A
    Iteration:  39%|███████████▎                 | 464/1184 [17:13<27:10,  2.26s/it][A
    Iteration:  39%|███████████▍                 | 465/1184 [17:15<27:03,  2.26s/it][A
    Iteration:  39%|███████████▍                 | 466/1184 [17:18<26:48,  2.24s/it][A
    Iteration:  39%|███████████▍                 | 467/1184 [17:20<26:31,  2.22s/it][A
    Iteration:  40%|███████████▍                 | 468/1184 [17:22<26:51,  2.25s/it][A
    Iteration:  40%|███████████▍                 | 469/1184 [17:24<27:04,  2.27s/it][A
    Iteration:  40%|███████████▌                 | 470/1184 [17:27<26:57,  2.27s/it][A
    Iteration:  40%|███████████▌                 | 471/1184 [17:29<26:56,  2.27s/it][A
    Iteration:  40%|███████████▌                 | 472/1184 [17:31<26:31,  2.23s/it][A
    Iteration:  40%|███████████▌                 | 473/1184 [17:33<26:15,  2.22s/it][A
    Iteration:  40%|███████████▌                 | 474/1184 [17:35<26:08,  2.21s/it][A
    Iteration:  40%|███████████▋                 | 475/1184 [17:38<26:03,  2.20s/it][A
    Iteration:  40%|███████████▋                 | 476/1184 [17:40<26:05,  2.21s/it][A
    Iteration:  40%|███████████▋                 | 477/1184 [17:42<25:56,  2.20s/it][A
    Iteration:  40%|███████████▋                 | 478/1184 [17:44<25:48,  2.19s/it][A
    Iteration:  40%|███████████▋                 | 479/1184 [17:46<26:01,  2.21s/it][A
    Iteration:  41%|███████████▊                 | 480/1184 [17:49<26:04,  2.22s/it][A
    Iteration:  41%|███████████▊                 | 481/1184 [17:51<25:53,  2.21s/it][A
    Iteration:  41%|███████████▊                 | 482/1184 [17:53<25:39,  2.19s/it][A
    Iteration:  41%|███████████▊                 | 483/1184 [17:55<25:39,  2.20s/it][A
    Iteration:  41%|███████████▊                 | 484/1184 [17:57<25:34,  2.19s/it][A
    Iteration:  41%|███████████▉                 | 485/1184 [18:00<25:25,  2.18s/it][A
    Iteration:  41%|███████████▉                 | 486/1184 [18:02<25:25,  2.19s/it][A
    Iteration:  41%|███████████▉                 | 487/1184 [18:04<25:22,  2.18s/it][A
    Iteration:  41%|███████████▉                 | 488/1184 [18:06<25:32,  2.20s/it][A
    Iteration:  41%|███████████▉                 | 489/1184 [18:08<25:39,  2.21s/it][A
    Iteration:  41%|████████████                 | 490/1184 [18:11<25:46,  2.23s/it][A
    Iteration:  41%|████████████                 | 491/1184 [18:13<25:47,  2.23s/it][A
    Iteration:  42%|████████████                 | 492/1184 [18:15<25:46,  2.23s/it][A
    Iteration:  42%|████████████                 | 493/1184 [18:17<25:32,  2.22s/it][A
    Iteration:  42%|████████████                 | 494/1184 [18:20<25:34,  2.22s/it][A
    Iteration:  42%|████████████                 | 495/1184 [18:22<25:24,  2.21s/it][A
    Iteration:  42%|████████████▏                | 496/1184 [18:24<25:06,  2.19s/it][A
    Iteration:  42%|████████████▏                | 497/1184 [18:26<24:57,  2.18s/it][A
    Iteration:  42%|████████████▏                | 498/1184 [18:28<25:00,  2.19s/it][A
    Iteration:  42%|████████████▏                | 499/1184 [18:30<24:57,  2.19s/it][A
    Iteration:  42%|████████████▏                | 500/1184 [18:33<24:50,  2.18s/it][A
    Iteration:  42%|████████████▎                | 501/1184 [18:35<24:45,  2.18s/it][A
    Iteration:  42%|████████████▎                | 502/1184 [18:37<24:39,  2.17s/it][A
    Iteration:  42%|████████████▎                | 503/1184 [18:39<25:09,  2.22s/it][A
    Iteration:  43%|████████████▎                | 504/1184 [18:42<25:17,  2.23s/it][A
    Iteration:  43%|████████████▎                | 505/1184 [18:44<25:01,  2.21s/it][A
    Iteration:  43%|████████████▍                | 506/1184 [18:46<24:50,  2.20s/it][A
    Iteration:  43%|████████████▍                | 507/1184 [18:48<24:42,  2.19s/it][A
    Iteration:  43%|████████████▍                | 508/1184 [18:50<24:36,  2.18s/it][A
    Iteration:  43%|████████████▍                | 509/1184 [18:53<25:00,  2.22s/it][A
    Iteration:  43%|████████████▍                | 510/1184 [18:55<25:18,  2.25s/it][A
    Iteration:  43%|████████████▌                | 511/1184 [18:57<25:08,  2.24s/it][A
    Iteration:  43%|████████████▌                | 512/1184 [18:59<24:57,  2.23s/it][A
    Iteration:  43%|████████████▌                | 513/1184 [19:02<24:57,  2.23s/it][A
    Iteration:  43%|████████████▌                | 514/1184 [19:04<25:11,  2.26s/it][A
    Iteration:  43%|████████████▌                | 515/1184 [19:06<25:22,  2.28s/it][A
    Iteration:  44%|████████████▋                | 516/1184 [19:08<25:12,  2.26s/it][A
    Iteration:  44%|████████████▋                | 517/1184 [19:11<24:56,  2.24s/it][A
    Iteration:  44%|████████████▋                | 518/1184 [19:13<24:52,  2.24s/it][A
    Iteration:  44%|████████████▋                | 519/1184 [19:15<24:48,  2.24s/it][A
    Iteration:  44%|████████████▋                | 520/1184 [19:17<24:45,  2.24s/it][A
    Iteration:  44%|████████████▊                | 521/1184 [19:19<24:32,  2.22s/it][A
    Iteration:  44%|████████████▊                | 522/1184 [19:22<24:21,  2.21s/it][A
    Iteration:  44%|████████████▊                | 523/1184 [19:24<24:23,  2.21s/it][A
    Iteration:  44%|████████████▊                | 524/1184 [19:26<24:24,  2.22s/it][A
    Iteration:  44%|████████████▊                | 525/1184 [19:28<24:31,  2.23s/it][A
    Iteration:  44%|████████████▉                | 526/1184 [19:31<24:35,  2.24s/it][A
    Iteration:  45%|████████████▉                | 527/1184 [19:33<25:00,  2.28s/it][A
    Iteration:  45%|████████████▉                | 528/1184 [19:35<24:43,  2.26s/it][A
    Iteration:  45%|████████████▉                | 529/1184 [19:37<24:25,  2.24s/it][A
    Iteration:  45%|████████████▉                | 530/1184 [19:40<24:06,  2.21s/it][A
    Iteration:  45%|█████████████                | 531/1184 [19:42<23:54,  2.20s/it][A
    Iteration:  45%|█████████████                | 532/1184 [19:44<24:11,  2.23s/it][A
    Iteration:  45%|█████████████                | 533/1184 [19:46<23:57,  2.21s/it][A
    Iteration:  45%|█████████████                | 534/1184 [19:48<24:02,  2.22s/it][A
    Iteration:  45%|█████████████                | 535/1184 [19:51<24:24,  2.26s/it][A
    Iteration:  45%|█████████████▏               | 536/1184 [19:53<24:34,  2.28s/it][A
    Iteration:  45%|█████████████▏               | 537/1184 [19:56<25:00,  2.32s/it][A
    Iteration:  45%|█████████████▏               | 538/1184 [19:58<24:53,  2.31s/it][A
    Iteration:  46%|█████████████▏               | 539/1184 [20:00<24:45,  2.30s/it][A
    Iteration:  46%|█████████████▏               | 540/1184 [20:02<24:18,  2.27s/it][A
    Iteration:  46%|█████████████▎               | 541/1184 [20:04<23:58,  2.24s/it][A
    Iteration:  46%|█████████████▎               | 542/1184 [20:07<24:13,  2.26s/it][A
    Iteration:  46%|█████████████▎               | 543/1184 [20:09<24:08,  2.26s/it][A
    Iteration:  46%|█████████████▎               | 544/1184 [20:11<24:00,  2.25s/it][A
    Iteration:  46%|█████████████▎               | 545/1184 [20:13<23:54,  2.24s/it][A
    Iteration:  46%|█████████████▎               | 546/1184 [20:16<23:49,  2.24s/it][A
    Iteration:  46%|█████████████▍               | 547/1184 [20:18<23:46,  2.24s/it][A
    Iteration:  46%|█████████████▍               | 548/1184 [20:20<23:42,  2.24s/it][A
    Iteration:  46%|█████████████▍               | 549/1184 [20:22<23:43,  2.24s/it][A
    Iteration:  46%|█████████████▍               | 550/1184 [20:25<23:41,  2.24s/it][A
    Iteration:  47%|█████████████▍               | 551/1184 [20:27<23:37,  2.24s/it][A
    Iteration:  47%|█████████████▌               | 552/1184 [20:29<23:33,  2.24s/it][A
    Iteration:  47%|█████████████▌               | 553/1184 [20:31<23:27,  2.23s/it][A
    Iteration:  47%|█████████████▌               | 554/1184 [20:34<23:24,  2.23s/it][A
    Iteration:  47%|█████████████▌               | 555/1184 [20:36<23:20,  2.23s/it][A
    Iteration:  47%|█████████████▌               | 556/1184 [20:38<23:15,  2.22s/it][A
    Iteration:  47%|█████████████▋               | 557/1184 [20:40<23:05,  2.21s/it][A
    Iteration:  47%|█████████████▋               | 558/1184 [20:42<23:04,  2.21s/it][A
    Iteration:  47%|█████████████▋               | 559/1184 [20:45<23:03,  2.21s/it][A
    Iteration:  47%|█████████████▋               | 560/1184 [20:47<23:15,  2.24s/it][A
    Iteration:  47%|█████████████▋               | 561/1184 [20:49<23:05,  2.22s/it][A
    Iteration:  47%|█████████████▊               | 562/1184 [20:51<23:05,  2.23s/it][A
    Iteration:  48%|█████████████▊               | 563/1184 [20:54<23:07,  2.23s/it][A
    Iteration:  48%|█████████████▊               | 564/1184 [20:56<22:57,  2.22s/it][A
    Iteration:  48%|█████████████▊               | 565/1184 [20:58<22:49,  2.21s/it][A
    Iteration:  48%|█████████████▊               | 566/1184 [21:00<22:55,  2.23s/it][A
    Iteration:  48%|█████████████▉               | 567/1184 [21:02<22:59,  2.24s/it][A
    Iteration:  48%|█████████████▉               | 568/1184 [21:05<22:58,  2.24s/it][A
    Iteration:  48%|█████████████▉               | 569/1184 [21:07<22:44,  2.22s/it][A
    Iteration:  48%|█████████████▉               | 570/1184 [21:09<22:48,  2.23s/it][A
    Iteration:  48%|█████████████▉               | 571/1184 [21:11<22:46,  2.23s/it][A
    Iteration:  48%|██████████████               | 572/1184 [21:14<22:41,  2.23s/it][A
    Iteration:  48%|██████████████               | 573/1184 [21:16<22:37,  2.22s/it][A
    Iteration:  48%|██████████████               | 574/1184 [21:18<22:34,  2.22s/it][A
    Iteration:  49%|██████████████               | 575/1184 [21:20<22:34,  2.22s/it][A
    Iteration:  49%|██████████████               | 576/1184 [21:22<22:32,  2.22s/it][A
    Iteration:  49%|██████████████▏              | 577/1184 [21:25<22:32,  2.23s/it][A
    Iteration:  49%|██████████████▏              | 578/1184 [21:27<22:44,  2.25s/it][A
    Iteration:  49%|██████████████▏              | 579/1184 [21:29<22:38,  2.25s/it][A
    Iteration:  49%|██████████████▏              | 580/1184 [21:32<22:35,  2.24s/it][A
    Iteration:  49%|██████████████▏              | 581/1184 [21:34<22:30,  2.24s/it][A
    Iteration:  49%|██████████████▎              | 582/1184 [21:36<22:30,  2.24s/it][A
    Iteration:  49%|██████████████▎              | 583/1184 [21:38<22:14,  2.22s/it][A
    Iteration:  49%|██████████████▎              | 584/1184 [21:40<22:14,  2.22s/it][A
    Iteration:  49%|██████████████▎              | 585/1184 [21:43<23:32,  2.36s/it][A
    Iteration:  49%|██████████████▎              | 586/1184 [21:45<23:15,  2.33s/it][A
    Iteration:  50%|██████████████▍              | 587/1184 [21:48<22:44,  2.28s/it][A
    Iteration:  50%|██████████████▍              | 588/1184 [21:50<22:31,  2.27s/it][A
    Iteration:  50%|██████████████▍              | 589/1184 [21:52<22:26,  2.26s/it][A
    Iteration:  50%|██████████████▍              | 590/1184 [21:54<22:06,  2.23s/it][A
    Iteration:  50%|██████████████▍              | 591/1184 [21:56<21:54,  2.22s/it][A
    Iteration:  50%|██████████████▌              | 592/1184 [21:58<21:44,  2.20s/it][A
    Iteration:  50%|██████████████▌              | 593/1184 [22:01<21:35,  2.19s/it][A
    Iteration:  50%|██████████████▌              | 594/1184 [22:03<21:38,  2.20s/it][A
    Iteration:  50%|██████████████▌              | 595/1184 [22:05<21:44,  2.21s/it][A
    Iteration:  50%|██████████████▌              | 596/1184 [22:07<21:42,  2.22s/it][A
    Iteration:  50%|██████████████▌              | 597/1184 [22:10<21:49,  2.23s/it][A
    Iteration:  51%|██████████████▋              | 598/1184 [22:12<21:34,  2.21s/it][A
    Iteration:  51%|██████████████▋              | 599/1184 [22:14<21:29,  2.20s/it][A
    Iteration:  51%|██████████████▋              | 600/1184 [22:16<21:21,  2.19s/it][A
    Iteration:  51%|██████████████▋              | 601/1184 [22:18<21:22,  2.20s/it][A
    Iteration:  51%|██████████████▋              | 602/1184 [22:21<21:26,  2.21s/it][A
    Iteration:  51%|██████████████▊              | 603/1184 [22:23<21:40,  2.24s/it][A
    Iteration:  51%|██████████████▊              | 604/1184 [22:25<22:07,  2.29s/it][A
    Iteration:  51%|██████████████▊              | 605/1184 [22:28<21:51,  2.27s/it][A
    Iteration:  51%|██████████████▊              | 606/1184 [22:30<21:34,  2.24s/it][A
    Iteration:  51%|██████████████▊              | 607/1184 [22:32<21:34,  2.24s/it][A
    Iteration:  51%|██████████████▉              | 608/1184 [22:34<21:36,  2.25s/it][A
    Iteration:  51%|██████████████▉              | 609/1184 [22:36<21:36,  2.25s/it][A
    Iteration:  52%|██████████████▉              | 610/1184 [22:39<21:31,  2.25s/it][A
    Iteration:  52%|██████████████▉              | 611/1184 [22:41<21:23,  2.24s/it][A
    Iteration:  52%|██████████████▉              | 612/1184 [22:43<21:21,  2.24s/it][A
    Iteration:  52%|███████████████              | 613/1184 [22:45<21:18,  2.24s/it][A
    Iteration:  52%|███████████████              | 614/1184 [22:48<21:22,  2.25s/it][A
    Iteration:  52%|███████████████              | 615/1184 [22:50<21:12,  2.24s/it][A
    Iteration:  52%|███████████████              | 616/1184 [22:52<21:11,  2.24s/it][A
    Iteration:  52%|███████████████              | 617/1184 [22:55<21:32,  2.28s/it][A
    Iteration:  52%|███████████████▏             | 618/1184 [22:57<21:32,  2.28s/it][A
    Iteration:  52%|███████████████▏             | 619/1184 [22:59<21:19,  2.26s/it][A
    Iteration:  52%|███████████████▏             | 620/1184 [23:01<21:01,  2.24s/it][A
    Iteration:  52%|███████████████▏             | 621/1184 [23:03<20:52,  2.22s/it][A
    Iteration:  53%|███████████████▏             | 622/1184 [23:06<20:55,  2.23s/it][A
    Iteration:  53%|███████████████▎             | 623/1184 [23:08<21:08,  2.26s/it][A
    Iteration:  53%|███████████████▎             | 624/1184 [23:10<21:18,  2.28s/it][A
    Iteration:  53%|███████████████▎             | 625/1184 [23:13<21:23,  2.30s/it][A
    Iteration:  53%|███████████████▎             | 626/1184 [23:15<21:17,  2.29s/it][A
    Iteration:  53%|███████████████▎             | 627/1184 [23:17<20:57,  2.26s/it][A
    Iteration:  53%|███████████████▍             | 628/1184 [23:19<20:46,  2.24s/it][A
    Iteration:  53%|███████████████▍             | 629/1184 [23:22<20:40,  2.24s/it][A
    Iteration:  53%|███████████████▍             | 630/1184 [23:24<20:27,  2.22s/it][A
    Iteration:  53%|███████████████▍             | 631/1184 [23:26<20:25,  2.22s/it][A
    Iteration:  53%|███████████████▍             | 632/1184 [23:28<20:26,  2.22s/it][A
    Iteration:  53%|███████████████▌             | 633/1184 [23:30<20:23,  2.22s/it][A
    Iteration:  54%|███████████████▌             | 634/1184 [23:33<20:24,  2.23s/it][A
    Iteration:  54%|███████████████▌             | 635/1184 [23:35<20:22,  2.23s/it][A
    Iteration:  54%|███████████████▌             | 636/1184 [23:37<20:37,  2.26s/it][A
    Iteration:  54%|███████████████▌             | 637/1184 [23:39<20:29,  2.25s/it][A
    Iteration:  54%|███████████████▋             | 638/1184 [23:42<20:16,  2.23s/it][A
    Iteration:  54%|███████████████▋             | 639/1184 [23:44<20:03,  2.21s/it][A
    Iteration:  54%|███████████████▋             | 640/1184 [23:46<19:55,  2.20s/it][A
    Iteration:  54%|███████████████▋             | 641/1184 [23:48<20:00,  2.21s/it][A
    Iteration:  54%|███████████████▋             | 642/1184 [23:50<19:57,  2.21s/it][A
    Iteration:  54%|███████████████▋             | 643/1184 [23:53<19:54,  2.21s/it][A
    Iteration:  54%|███████████████▊             | 644/1184 [23:55<20:07,  2.24s/it][A
    Iteration:  54%|███████████████▊             | 645/1184 [23:57<20:04,  2.24s/it][A
    Iteration:  55%|███████████████▊             | 646/1184 [23:59<19:50,  2.21s/it][A
    Iteration:  55%|███████████████▊             | 647/1184 [24:01<19:36,  2.19s/it][A
    Iteration:  55%|███████████████▊             | 648/1184 [24:04<19:25,  2.18s/it][A
    Iteration:  55%|███████████████▉             | 649/1184 [24:06<19:15,  2.16s/it][A
    Iteration:  55%|███████████████▉             | 650/1184 [24:08<19:11,  2.16s/it][A
    Iteration:  55%|███████████████▉             | 651/1184 [24:10<19:23,  2.18s/it][A
    Iteration:  55%|███████████████▉             | 652/1184 [24:12<19:25,  2.19s/it][A
    Iteration:  55%|███████████████▉             | 653/1184 [24:14<19:29,  2.20s/it][A
    Iteration:  55%|████████████████             | 654/1184 [24:17<19:33,  2.21s/it][A
    Iteration:  55%|████████████████             | 655/1184 [24:19<19:45,  2.24s/it][A
    Iteration:  55%|████████████████             | 656/1184 [24:21<19:32,  2.22s/it][A
    Iteration:  55%|████████████████             | 657/1184 [24:23<19:29,  2.22s/it][A
    Iteration:  56%|████████████████             | 658/1184 [24:26<19:19,  2.21s/it][A
    Iteration:  56%|████████████████▏            | 659/1184 [24:28<19:30,  2.23s/it][A
    Iteration:  56%|████████████████▏            | 660/1184 [24:30<19:33,  2.24s/it][A
    Iteration:  56%|████████████████▏            | 661/1184 [24:32<19:30,  2.24s/it][A
    Iteration:  56%|████████████████▏            | 662/1184 [24:35<19:32,  2.25s/it][A
    Iteration:  56%|████████████████▏            | 663/1184 [24:37<19:34,  2.26s/it][A
    Iteration:  56%|████████████████▎            | 664/1184 [24:39<19:29,  2.25s/it][A
    Iteration:  56%|████████████████▎            | 665/1184 [24:41<19:29,  2.25s/it][A
    Iteration:  56%|████████████████▎            | 666/1184 [24:44<19:26,  2.25s/it][A
    Iteration:  56%|████████████████▎            | 667/1184 [24:46<19:30,  2.26s/it][A
    Iteration:  56%|████████████████▎            | 668/1184 [24:48<19:30,  2.27s/it][A
    Iteration:  57%|████████████████▍            | 669/1184 [24:50<19:29,  2.27s/it][A
    Iteration:  57%|████████████████▍            | 670/1184 [24:53<19:24,  2.26s/it][A
    Iteration:  57%|████████████████▍            | 671/1184 [24:55<19:04,  2.23s/it][A
    Iteration:  57%|████████████████▍            | 672/1184 [24:57<19:03,  2.23s/it][A
    Iteration:  57%|████████████████▍            | 673/1184 [24:59<19:00,  2.23s/it][A
    Iteration:  57%|████████████████▌            | 674/1184 [25:02<19:02,  2.24s/it][A
    Iteration:  57%|████████████████▌            | 675/1184 [25:04<19:17,  2.27s/it][A
    Iteration:  57%|████████████████▌            | 676/1184 [25:06<19:10,  2.27s/it][A
    Iteration:  57%|████████████████▌            | 677/1184 [25:08<19:05,  2.26s/it][A
    Iteration:  57%|████████████████▌            | 678/1184 [25:11<19:00,  2.25s/it][A
    Iteration:  57%|████████████████▋            | 679/1184 [25:13<18:48,  2.23s/it][A
    Iteration:  57%|████████████████▋            | 680/1184 [25:15<18:47,  2.24s/it][A
    Iteration:  58%|████████████████▋            | 681/1184 [25:17<18:46,  2.24s/it][A
    Iteration:  58%|████████████████▋            | 682/1184 [25:20<18:59,  2.27s/it][A
    Iteration:  58%|████████████████▋            | 683/1184 [25:22<19:23,  2.32s/it][A
    Iteration:  58%|████████████████▊            | 684/1184 [25:24<18:59,  2.28s/it][A
    Iteration:  58%|████████████████▊            | 685/1184 [25:26<18:36,  2.24s/it][A
    Iteration:  58%|████████████████▊            | 686/1184 [25:29<18:18,  2.21s/it][A
    Iteration:  58%|████████████████▊            | 687/1184 [25:31<18:04,  2.18s/it][A
    Iteration:  58%|████████████████▊            | 688/1184 [25:33<17:53,  2.16s/it][A
    Iteration:  58%|████████████████▉            | 689/1184 [25:35<17:53,  2.17s/it][A
    Iteration:  58%|████████████████▉            | 690/1184 [25:37<17:51,  2.17s/it][A
    Iteration:  58%|████████████████▉            | 691/1184 [25:39<17:58,  2.19s/it][A
    Iteration:  58%|████████████████▉            | 692/1184 [25:42<18:01,  2.20s/it][A
    Iteration:  59%|████████████████▉            | 693/1184 [25:44<18:03,  2.21s/it][A
    Iteration:  59%|████████████████▉            | 694/1184 [25:46<18:03,  2.21s/it][A
    Iteration:  59%|█████████████████            | 695/1184 [25:48<18:05,  2.22s/it][A
    Iteration:  59%|█████████████████            | 696/1184 [25:51<18:00,  2.21s/it][A
    Iteration:  59%|█████████████████            | 697/1184 [25:53<18:01,  2.22s/it][A
    Iteration:  59%|█████████████████            | 698/1184 [25:55<17:48,  2.20s/it][A
    Iteration:  59%|█████████████████            | 699/1184 [25:57<17:39,  2.19s/it][A
    Iteration:  59%|█████████████████▏           | 700/1184 [25:59<17:41,  2.19s/it][A
    Iteration:  59%|█████████████████▏           | 701/1184 [26:02<17:46,  2.21s/it][A
    Iteration:  59%|█████████████████▏           | 702/1184 [26:04<17:59,  2.24s/it][A
    Iteration:  59%|█████████████████▏           | 703/1184 [26:07<19:49,  2.47s/it][A
    Iteration:  59%|█████████████████▏           | 704/1184 [26:09<19:04,  2.38s/it][A
    Iteration:  60%|█████████████████▎           | 705/1184 [26:11<18:44,  2.35s/it][A
    Iteration:  60%|█████████████████▎           | 706/1184 [26:13<18:12,  2.29s/it][A
    Iteration:  60%|█████████████████▎           | 707/1184 [26:16<18:02,  2.27s/it][A
    Iteration:  60%|█████████████████▎           | 708/1184 [26:18<17:43,  2.23s/it][A
    Iteration:  60%|█████████████████▎           | 709/1184 [26:20<17:42,  2.24s/it][A
    Iteration:  60%|█████████████████▍           | 710/1184 [26:22<17:39,  2.24s/it][A
    Iteration:  60%|█████████████████▍           | 711/1184 [26:25<17:38,  2.24s/it][A
    Iteration:  60%|█████████████████▍           | 712/1184 [26:27<17:38,  2.24s/it][A
    Iteration:  60%|█████████████████▍           | 713/1184 [26:29<17:41,  2.25s/it][A
    Iteration:  60%|█████████████████▍           | 714/1184 [26:31<17:42,  2.26s/it][A
    Iteration:  60%|█████████████████▌           | 715/1184 [26:34<17:39,  2.26s/it][A
    Iteration:  60%|█████████████████▌           | 716/1184 [26:36<17:35,  2.26s/it][A
    Iteration:  61%|█████████████████▌           | 717/1184 [26:38<17:32,  2.25s/it][A
    Iteration:  61%|█████████████████▌           | 718/1184 [26:40<17:29,  2.25s/it][A
    Iteration:  61%|█████████████████▌           | 719/1184 [26:43<17:37,  2.27s/it][A
    Iteration:  61%|█████████████████▋           | 720/1184 [26:45<17:30,  2.26s/it][A
    Iteration:  61%|█████████████████▋           | 721/1184 [26:47<17:25,  2.26s/it][A
    Iteration:  61%|█████████████████▋           | 722/1184 [26:49<17:24,  2.26s/it][A
    Iteration:  61%|█████████████████▋           | 723/1184 [26:52<17:20,  2.26s/it][A
    Iteration:  61%|█████████████████▋           | 724/1184 [26:54<17:10,  2.24s/it][A
    Iteration:  61%|█████████████████▊           | 725/1184 [26:56<17:08,  2.24s/it][A
    Iteration:  61%|█████████████████▊           | 726/1184 [26:58<17:07,  2.24s/it][A
    Iteration:  61%|█████████████████▊           | 727/1184 [27:01<16:51,  2.21s/it][A
    Iteration:  61%|█████████████████▊           | 728/1184 [27:03<16:41,  2.20s/it][A
    Iteration:  62%|█████████████████▊           | 729/1184 [27:05<16:42,  2.20s/it][A
    Iteration:  62%|█████████████████▉           | 730/1184 [27:07<16:49,  2.22s/it][A
    Iteration:  62%|█████████████████▉           | 731/1184 [27:10<17:02,  2.26s/it][A
    Iteration:  62%|█████████████████▉           | 732/1184 [27:12<17:01,  2.26s/it][A
    Iteration:  62%|█████████████████▉           | 733/1184 [27:14<16:48,  2.24s/it][A
    Iteration:  62%|█████████████████▉           | 734/1184 [27:16<16:36,  2.21s/it][A
    Iteration:  62%|██████████████████           | 735/1184 [27:18<16:27,  2.20s/it][A
    Iteration:  62%|██████████████████           | 736/1184 [27:21<16:29,  2.21s/it][A
    Iteration:  62%|██████████████████           | 737/1184 [27:23<16:34,  2.22s/it][A
    Iteration:  62%|██████████████████           | 738/1184 [27:25<16:32,  2.23s/it][A
    Iteration:  62%|██████████████████           | 739/1184 [27:27<16:32,  2.23s/it][A
    Iteration:  62%|██████████████████▏          | 740/1184 [27:30<16:31,  2.23s/it][A
    Iteration:  63%|██████████████████▏          | 741/1184 [27:32<16:37,  2.25s/it][A
    Iteration:  63%|██████████████████▏          | 742/1184 [27:34<16:26,  2.23s/it][A
    Iteration:  63%|██████████████████▏          | 743/1184 [27:36<16:25,  2.24s/it][A
    Iteration:  63%|██████████████████▏          | 744/1184 [27:38<16:15,  2.22s/it][A
    Iteration:  63%|██████████████████▏          | 745/1184 [27:41<16:08,  2.21s/it][A
    Iteration:  63%|██████████████████▎          | 746/1184 [27:43<16:02,  2.20s/it][A
    Iteration:  63%|██████████████████▎          | 747/1184 [27:45<15:56,  2.19s/it][A
    Iteration:  63%|██████████████████▎          | 748/1184 [27:47<16:00,  2.20s/it][A
    Iteration:  63%|██████████████████▎          | 749/1184 [27:49<15:53,  2.19s/it][A
    Iteration:  63%|██████████████████▎          | 750/1184 [27:51<15:47,  2.18s/it][A
    Iteration:  63%|██████████████████▍          | 751/1184 [27:54<15:44,  2.18s/it][A
    Iteration:  64%|██████████████████▍          | 752/1184 [27:56<15:39,  2.18s/it][A
    Iteration:  64%|██████████████████▍          | 753/1184 [27:58<15:34,  2.17s/it][A
    Iteration:  64%|██████████████████▍          | 754/1184 [28:00<15:30,  2.16s/it][A
    Iteration:  64%|██████████████████▍          | 755/1184 [28:02<15:45,  2.20s/it][A
    Iteration:  64%|██████████████████▌          | 756/1184 [28:05<15:54,  2.23s/it][A
    Iteration:  64%|██████████████████▌          | 757/1184 [28:07<15:56,  2.24s/it][A
    Iteration:  64%|██████████████████▌          | 758/1184 [28:09<15:54,  2.24s/it][A
    Iteration:  64%|██████████████████▌          | 759/1184 [28:11<15:48,  2.23s/it][A
    Iteration:  64%|██████████████████▌          | 760/1184 [28:14<15:42,  2.22s/it][A
    Iteration:  64%|██████████████████▋          | 761/1184 [28:16<15:34,  2.21s/it][A
    Iteration:  64%|██████████████████▋          | 762/1184 [28:18<15:28,  2.20s/it][A
    Iteration:  64%|██████████████████▋          | 763/1184 [28:20<15:31,  2.21s/it][A
    Iteration:  65%|██████████████████▋          | 764/1184 [28:22<15:32,  2.22s/it][A
    Iteration:  65%|██████████████████▋          | 765/1184 [28:25<15:33,  2.23s/it][A
    Iteration:  65%|██████████████████▊          | 766/1184 [28:27<15:32,  2.23s/it][A
    Iteration:  65%|██████████████████▊          | 767/1184 [28:29<15:32,  2.24s/it][A
    Iteration:  65%|██████████████████▊          | 768/1184 [28:31<15:31,  2.24s/it][A
    Iteration:  65%|██████████████████▊          | 769/1184 [28:34<15:29,  2.24s/it][A
    Iteration:  65%|██████████████████▊          | 770/1184 [28:36<15:39,  2.27s/it][A
    Iteration:  65%|██████████████████▉          | 771/1184 [28:38<15:35,  2.27s/it][A
    Iteration:  65%|██████████████████▉          | 772/1184 [28:40<15:26,  2.25s/it][A
    Iteration:  65%|██████████████████▉          | 773/1184 [28:43<15:13,  2.22s/it][A
    Iteration:  65%|██████████████████▉          | 774/1184 [28:45<15:13,  2.23s/it][A
    Iteration:  65%|██████████████████▉          | 775/1184 [28:47<15:21,  2.25s/it][A
    Iteration:  66%|███████████████████          | 776/1184 [28:49<15:18,  2.25s/it][A
    Iteration:  66%|███████████████████          | 777/1184 [28:52<15:26,  2.28s/it][A
    Iteration:  66%|███████████████████          | 778/1184 [28:54<15:27,  2.28s/it][A
    Iteration:  66%|███████████████████          | 779/1184 [28:56<15:21,  2.28s/it][A
    Iteration:  66%|███████████████████          | 780/1184 [28:59<15:14,  2.26s/it][A
    Iteration:  66%|███████████████████▏         | 781/1184 [29:01<14:59,  2.23s/it][A
    Iteration:  66%|███████████████████▏         | 782/1184 [29:03<15:10,  2.27s/it][A
    Iteration:  66%|███████████████████▏         | 783/1184 [29:05<15:03,  2.25s/it][A
    Iteration:  66%|███████████████████▏         | 784/1184 [29:08<14:55,  2.24s/it][A
    Iteration:  66%|███████████████████▏         | 785/1184 [29:10<14:53,  2.24s/it][A
    Iteration:  66%|███████████████████▎         | 786/1184 [29:12<14:43,  2.22s/it][A
    Iteration:  66%|███████████████████▎         | 787/1184 [29:14<14:42,  2.22s/it][A
    Iteration:  67%|███████████████████▎         | 788/1184 [29:16<14:50,  2.25s/it][A
    Iteration:  67%|███████████████████▎         | 789/1184 [29:19<14:52,  2.26s/it][A
    Iteration:  67%|███████████████████▎         | 790/1184 [29:21<14:49,  2.26s/it][A
    Iteration:  67%|███████████████████▎         | 791/1184 [29:23<14:40,  2.24s/it][A
    Iteration:  67%|███████████████████▍         | 792/1184 [29:25<14:29,  2.22s/it][A
    Iteration:  67%|███████████████████▍         | 793/1184 [29:28<14:22,  2.21s/it][A
    Iteration:  67%|███████████████████▍         | 794/1184 [29:30<14:15,  2.19s/it][A
    Iteration:  67%|███████████████████▍         | 795/1184 [29:32<14:11,  2.19s/it][A
    Iteration:  67%|███████████████████▍         | 796/1184 [29:34<14:12,  2.20s/it][A
    Iteration:  67%|███████████████████▌         | 797/1184 [29:36<14:18,  2.22s/it][A
    Iteration:  67%|███████████████████▌         | 798/1184 [29:39<14:15,  2.22s/it][A
    Iteration:  67%|███████████████████▌         | 799/1184 [29:41<14:29,  2.26s/it][A
    Iteration:  68%|███████████████████▌         | 800/1184 [29:43<14:16,  2.23s/it][A
    Iteration:  68%|███████████████████▌         | 801/1184 [29:45<14:03,  2.20s/it][A
    Iteration:  68%|███████████████████▋         | 802/1184 [29:47<13:55,  2.19s/it][A
    Iteration:  68%|███████████████████▋         | 803/1184 [29:50<14:07,  2.22s/it][A
    Iteration:  68%|███████████████████▋         | 804/1184 [29:52<14:17,  2.26s/it][A
    Iteration:  68%|███████████████████▋         | 805/1184 [29:54<14:14,  2.25s/it][A
    Iteration:  68%|███████████████████▋         | 806/1184 [29:56<14:03,  2.23s/it][A
    Iteration:  68%|███████████████████▊         | 807/1184 [29:59<13:55,  2.22s/it][A
    Iteration:  68%|███████████████████▊         | 808/1184 [30:01<13:55,  2.22s/it][A
    Iteration:  68%|███████████████████▊         | 809/1184 [30:03<13:57,  2.23s/it][A
    Iteration:  68%|███████████████████▊         | 810/1184 [30:05<13:53,  2.23s/it][A
    Iteration:  68%|███████████████████▊         | 811/1184 [30:08<13:52,  2.23s/it][A
    Iteration:  69%|███████████████████▉         | 812/1184 [30:10<13:47,  2.22s/it][A
    Iteration:  69%|███████████████████▉         | 813/1184 [30:12<13:51,  2.24s/it][A
    Iteration:  69%|███████████████████▉         | 814/1184 [30:14<13:44,  2.23s/it][A
    Iteration:  69%|███████████████████▉         | 815/1184 [30:16<13:34,  2.21s/it][A
    Iteration:  69%|███████████████████▉         | 816/1184 [30:19<13:30,  2.20s/it][A
    Iteration:  69%|████████████████████         | 817/1184 [30:21<13:38,  2.23s/it][A
    Iteration:  69%|████████████████████         | 818/1184 [30:23<13:31,  2.22s/it][A
    Iteration:  69%|████████████████████         | 819/1184 [30:25<13:27,  2.21s/it][A
    Iteration:  69%|████████████████████         | 820/1184 [30:28<13:21,  2.20s/it][A
    Iteration:  69%|████████████████████         | 821/1184 [30:30<13:16,  2.19s/it][A
    Iteration:  69%|████████████████████▏        | 822/1184 [30:32<13:22,  2.22s/it][A
    Iteration:  70%|████████████████████▏        | 823/1184 [30:34<13:16,  2.21s/it][A
    Iteration:  70%|████████████████████▏        | 824/1184 [30:36<13:11,  2.20s/it][A
    Iteration:  70%|████████████████████▏        | 825/1184 [30:38<13:04,  2.18s/it][A
    Iteration:  70%|████████████████████▏        | 826/1184 [30:41<13:00,  2.18s/it][A
    Iteration:  70%|████████████████████▎        | 827/1184 [30:43<13:06,  2.20s/it][A
    Iteration:  70%|████████████████████▎        | 828/1184 [30:45<13:43,  2.31s/it][A
    Iteration:  70%|████████████████████▎        | 829/1184 [30:48<13:30,  2.28s/it][A
    Iteration:  70%|████████████████████▎        | 830/1184 [30:50<13:19,  2.26s/it][A
    Iteration:  70%|████████████████████▎        | 831/1184 [30:52<13:22,  2.27s/it][A
    Iteration:  70%|████████████████████▍        | 832/1184 [30:54<13:11,  2.25s/it][A
    Iteration:  70%|████████████████████▍        | 833/1184 [30:57<13:07,  2.24s/it][A
    Iteration:  70%|████████████████████▍        | 834/1184 [30:59<12:57,  2.22s/it][A
    Iteration:  71%|████████████████████▍        | 835/1184 [31:01<12:52,  2.21s/it][A
    Iteration:  71%|████████████████████▍        | 836/1184 [31:03<12:45,  2.20s/it][A
    Iteration:  71%|████████████████████▌        | 837/1184 [31:05<12:37,  2.18s/it][A
    Iteration:  71%|████████████████████▌        | 838/1184 [31:08<12:40,  2.20s/it][A
    Iteration:  71%|████████████████████▌        | 839/1184 [31:10<12:33,  2.18s/it][A
    Iteration:  71%|████████████████████▌        | 840/1184 [31:12<12:29,  2.18s/it][A
    Iteration:  71%|████████████████████▌        | 841/1184 [31:14<12:34,  2.20s/it][A
    Iteration:  71%|████████████████████▌        | 842/1184 [31:16<12:25,  2.18s/it][A
    Iteration:  71%|████████████████████▋        | 843/1184 [31:18<12:24,  2.18s/it][A
    Iteration:  71%|████████████████████▋        | 844/1184 [31:21<12:22,  2.18s/it][A
    Iteration:  71%|████████████████████▋        | 845/1184 [31:23<12:19,  2.18s/it][A
    Iteration:  71%|████████████████████▋        | 846/1184 [31:25<12:16,  2.18s/it][A
    Iteration:  72%|████████████████████▋        | 847/1184 [31:27<12:15,  2.18s/it][A
    Iteration:  72%|████████████████████▊        | 848/1184 [31:29<12:20,  2.20s/it][A
    Iteration:  72%|████████████████████▊        | 849/1184 [31:32<12:15,  2.20s/it][A
    Iteration:  72%|████████████████████▊        | 850/1184 [31:34<12:20,  2.22s/it][A
    Iteration:  72%|████████████████████▊        | 851/1184 [31:36<12:14,  2.21s/it][A
    Iteration:  72%|████████████████████▊        | 852/1184 [31:38<12:12,  2.21s/it][A
    Iteration:  72%|████████████████████▉        | 853/1184 [31:40<12:07,  2.20s/it][A
    Iteration:  72%|████████████████████▉        | 854/1184 [31:43<12:01,  2.19s/it][A
    Iteration:  72%|████████████████████▉        | 855/1184 [31:45<12:03,  2.20s/it][A
    Iteration:  72%|████████████████████▉        | 856/1184 [31:47<12:05,  2.21s/it][A
    Iteration:  72%|████████████████████▉        | 857/1184 [31:49<12:15,  2.25s/it][A
    Iteration:  72%|█████████████████████        | 858/1184 [31:52<12:05,  2.23s/it][A
    Iteration:  73%|█████████████████████        | 859/1184 [31:54<12:00,  2.22s/it][A
    Iteration:  73%|█████████████████████        | 860/1184 [31:56<11:56,  2.21s/it][A
    Iteration:  73%|█████████████████████        | 861/1184 [31:58<11:52,  2.21s/it][A
    Iteration:  73%|█████████████████████        | 862/1184 [32:00<11:49,  2.20s/it][A
    Iteration:  73%|█████████████████████▏       | 863/1184 [32:02<11:42,  2.19s/it][A
    Iteration:  73%|█████████████████████▏       | 864/1184 [32:05<11:39,  2.18s/it][A
    Iteration:  73%|█████████████████████▏       | 865/1184 [32:07<11:38,  2.19s/it][A
    Iteration:  73%|█████████████████████▏       | 866/1184 [32:09<11:38,  2.20s/it][A
    Iteration:  73%|█████████████████████▏       | 867/1184 [32:11<11:39,  2.21s/it][A
    Iteration:  73%|█████████████████████▎       | 868/1184 [32:14<11:48,  2.24s/it][A
    Iteration:  73%|█████████████████████▎       | 869/1184 [32:16<11:51,  2.26s/it][A
    Iteration:  73%|█████████████████████▎       | 870/1184 [32:18<11:41,  2.24s/it][A
    Iteration:  74%|█████████████████████▎       | 871/1184 [32:20<11:35,  2.22s/it][A
    Iteration:  74%|█████████████████████▎       | 872/1184 [32:23<11:31,  2.22s/it][A
    Iteration:  74%|█████████████████████▍       | 873/1184 [32:25<11:25,  2.20s/it][A
    Iteration:  74%|█████████████████████▍       | 874/1184 [32:27<11:20,  2.20s/it][A
    Iteration:  74%|█████████████████████▍       | 875/1184 [32:29<11:16,  2.19s/it][A
    Iteration:  74%|█████████████████████▍       | 876/1184 [32:31<11:16,  2.20s/it][A
    Iteration:  74%|█████████████████████▍       | 877/1184 [32:33<11:19,  2.21s/it][A
    Iteration:  74%|█████████████████████▌       | 878/1184 [32:36<11:19,  2.22s/it][A
    Iteration:  74%|█████████████████████▌       | 879/1184 [32:38<11:17,  2.22s/it][A
    Iteration:  74%|█████████████████████▌       | 880/1184 [32:40<11:16,  2.22s/it][A
    Iteration:  74%|█████████████████████▌       | 881/1184 [32:42<11:10,  2.21s/it][A
    Iteration:  74%|█████████████████████▌       | 882/1184 [32:45<11:08,  2.22s/it][A
    Iteration:  75%|█████████████████████▋       | 883/1184 [32:47<11:06,  2.21s/it][A
    Iteration:  75%|█████████████████████▋       | 884/1184 [32:49<11:05,  2.22s/it][A
    Iteration:  75%|█████████████████████▋       | 885/1184 [32:51<11:06,  2.23s/it][A
    Iteration:  75%|█████████████████████▋       | 886/1184 [32:53<11:01,  2.22s/it][A
    Iteration:  75%|█████████████████████▋       | 887/1184 [32:56<10:56,  2.21s/it][A
    Iteration:  75%|█████████████████████▊       | 888/1184 [32:58<10:50,  2.20s/it][A
    Iteration:  75%|█████████████████████▊       | 889/1184 [33:00<10:43,  2.18s/it][A
    Iteration:  75%|█████████████████████▊       | 890/1184 [33:02<10:42,  2.18s/it][A
    Iteration:  75%|█████████████████████▊       | 891/1184 [33:04<10:38,  2.18s/it][A
    Iteration:  75%|█████████████████████▊       | 892/1184 [33:07<10:35,  2.18s/it][A
    Iteration:  75%|█████████████████████▊       | 893/1184 [33:09<10:29,  2.16s/it][A
    Iteration:  76%|█████████████████████▉       | 894/1184 [33:11<10:28,  2.17s/it][A
    Iteration:  76%|█████████████████████▉       | 895/1184 [33:13<10:28,  2.18s/it][A
    Iteration:  76%|█████████████████████▉       | 896/1184 [33:15<10:29,  2.19s/it][A
    Iteration:  76%|█████████████████████▉       | 897/1184 [33:18<10:36,  2.22s/it][A
    Iteration:  76%|█████████████████████▉       | 898/1184 [33:20<10:31,  2.21s/it][A
    Iteration:  76%|██████████████████████       | 899/1184 [33:22<10:25,  2.20s/it][A
    Iteration:  76%|██████████████████████       | 900/1184 [33:24<10:24,  2.20s/it][A
    Iteration:  76%|██████████████████████       | 901/1184 [33:26<10:26,  2.21s/it][A
    Iteration:  76%|██████████████████████       | 902/1184 [33:29<10:30,  2.23s/it][A
    Iteration:  76%|██████████████████████       | 903/1184 [33:31<10:24,  2.22s/it][A
    Iteration:  76%|██████████████████████▏      | 904/1184 [33:33<10:20,  2.22s/it][A
    Iteration:  76%|██████████████████████▏      | 905/1184 [33:35<10:22,  2.23s/it][A
    Iteration:  77%|██████████████████████▏      | 906/1184 [33:37<10:16,  2.22s/it][A
    Iteration:  77%|██████████████████████▏      | 907/1184 [33:40<10:09,  2.20s/it][A
    Iteration:  77%|██████████████████████▏      | 908/1184 [33:42<10:13,  2.22s/it][A
    Iteration:  77%|██████████████████████▎      | 909/1184 [33:44<10:06,  2.21s/it][A
    Iteration:  77%|██████████████████████▎      | 910/1184 [33:46<10:08,  2.22s/it][A
    Iteration:  77%|██████████████████████▎      | 911/1184 [33:49<10:08,  2.23s/it][A
    Iteration:  77%|██████████████████████▎      | 912/1184 [33:51<10:09,  2.24s/it][A
    Iteration:  77%|██████████████████████▎      | 913/1184 [33:53<10:07,  2.24s/it][A
    Iteration:  77%|██████████████████████▍      | 914/1184 [33:55<10:05,  2.24s/it][A
    Iteration:  77%|██████████████████████▍      | 915/1184 [33:58<10:06,  2.25s/it][A
    Iteration:  77%|██████████████████████▍      | 916/1184 [34:00<10:04,  2.26s/it][A
    Iteration:  77%|██████████████████████▍      | 917/1184 [34:02<10:01,  2.25s/it][A
    Iteration:  78%|██████████████████████▍      | 918/1184 [34:04<09:54,  2.24s/it][A
    Iteration:  78%|██████████████████████▌      | 919/1184 [34:07<09:49,  2.23s/it][A
    Iteration:  78%|██████████████████████▌      | 920/1184 [34:09<09:45,  2.22s/it][A
    Iteration:  78%|██████████████████████▌      | 921/1184 [34:11<09:39,  2.20s/it][A
    Iteration:  78%|██████████████████████▌      | 922/1184 [34:13<09:34,  2.19s/it][A
    Iteration:  78%|██████████████████████▌      | 923/1184 [34:15<09:36,  2.21s/it][A
    Iteration:  78%|██████████████████████▋      | 924/1184 [34:17<09:33,  2.20s/it][A
    Iteration:  78%|██████████████████████▋      | 925/1184 [34:20<09:27,  2.19s/it][A
    Iteration:  78%|██████████████████████▋      | 926/1184 [34:22<09:31,  2.22s/it][A
    Iteration:  78%|██████████████████████▋      | 927/1184 [34:24<09:53,  2.31s/it][A
    Iteration:  78%|██████████████████████▋      | 928/1184 [34:27<09:46,  2.29s/it][A
    Iteration:  78%|██████████████████████▊      | 929/1184 [34:29<09:50,  2.32s/it][A
    Iteration:  79%|██████████████████████▊      | 930/1184 [34:31<09:51,  2.33s/it][A
    Iteration:  79%|██████████████████████▊      | 931/1184 [34:34<09:49,  2.33s/it][A
    Iteration:  79%|██████████████████████▊      | 932/1184 [34:36<09:37,  2.29s/it][A
    Iteration:  79%|██████████████████████▊      | 933/1184 [34:38<09:27,  2.26s/it][A
    Iteration:  79%|██████████████████████▉      | 934/1184 [34:40<09:19,  2.24s/it][A
    Iteration:  79%|██████████████████████▉      | 935/1184 [34:43<09:13,  2.22s/it][A
    Iteration:  79%|██████████████████████▉      | 936/1184 [34:45<09:14,  2.24s/it][A
    Iteration:  79%|██████████████████████▉      | 937/1184 [34:47<09:09,  2.22s/it][A
    Iteration:  79%|██████████████████████▉      | 938/1184 [34:49<09:05,  2.22s/it][A
    Iteration:  79%|██████████████████████▉      | 939/1184 [34:51<09:06,  2.23s/it][A
    Iteration:  79%|███████████████████████      | 940/1184 [34:54<09:00,  2.22s/it][A
    Iteration:  79%|███████████████████████      | 941/1184 [34:56<08:56,  2.21s/it][A
    Iteration:  80%|███████████████████████      | 942/1184 [34:58<08:53,  2.20s/it][A
    Iteration:  80%|███████████████████████      | 943/1184 [35:00<08:57,  2.23s/it][A
    Iteration:  80%|███████████████████████      | 944/1184 [35:02<08:48,  2.20s/it][A
    Iteration:  80%|███████████████████████▏     | 945/1184 [35:05<08:39,  2.17s/it][A
    Iteration:  80%|███████████████████████▏     | 946/1184 [35:07<08:34,  2.16s/it][A
    Iteration:  80%|███████████████████████▏     | 947/1184 [35:09<08:35,  2.18s/it][A
    Iteration:  80%|███████████████████████▏     | 948/1184 [35:11<08:35,  2.18s/it][A
    Iteration:  80%|███████████████████████▏     | 949/1184 [35:13<08:33,  2.19s/it][A
    Iteration:  80%|███████████████████████▎     | 950/1184 [35:16<08:39,  2.22s/it][A
    Iteration:  80%|███████████████████████▎     | 951/1184 [35:18<08:35,  2.21s/it][A
    Iteration:  80%|███████████████████████▎     | 952/1184 [35:20<08:32,  2.21s/it][A
    Iteration:  80%|███████████████████████▎     | 953/1184 [35:22<08:29,  2.21s/it][A
    Iteration:  81%|███████████████████████▎     | 954/1184 [35:24<08:27,  2.21s/it][A
    Iteration:  81%|███████████████████████▍     | 955/1184 [35:27<08:21,  2.19s/it][A
    Iteration:  81%|███████████████████████▍     | 956/1184 [35:29<08:18,  2.19s/it][A
    Iteration:  81%|███████████████████████▍     | 957/1184 [35:31<08:12,  2.17s/it][A
    Iteration:  81%|███████████████████████▍     | 958/1184 [35:33<08:08,  2.16s/it][A
    Iteration:  81%|███████████████████████▍     | 959/1184 [35:35<08:06,  2.16s/it][A
    Iteration:  81%|███████████████████████▌     | 960/1184 [35:37<08:11,  2.20s/it][A
    Iteration:  81%|███████████████████████▌     | 961/1184 [35:40<08:09,  2.20s/it][A
    Iteration:  81%|███████████████████████▌     | 962/1184 [35:42<08:10,  2.21s/it][A
    Iteration:  81%|███████████████████████▌     | 963/1184 [35:44<08:05,  2.20s/it][A
    Iteration:  81%|███████████████████████▌     | 964/1184 [35:46<08:04,  2.20s/it][A
    Iteration:  82%|███████████████████████▋     | 965/1184 [35:48<08:03,  2.21s/it][A
    Iteration:  82%|███████████████████████▋     | 966/1184 [35:51<08:02,  2.21s/it][A
    Iteration:  82%|███████████████████████▋     | 967/1184 [35:53<08:00,  2.21s/it][A
    Iteration:  82%|███████████████████████▋     | 968/1184 [35:55<07:59,  2.22s/it][A
    Iteration:  82%|███████████████████████▋     | 969/1184 [35:57<07:58,  2.22s/it][A
    Iteration:  82%|███████████████████████▊     | 970/1184 [36:00<07:54,  2.22s/it][A
    Iteration:  82%|███████████████████████▊     | 971/1184 [36:02<07:54,  2.23s/it][A
    Iteration:  82%|███████████████████████▊     | 972/1184 [36:04<07:55,  2.24s/it][A
    Iteration:  82%|███████████████████████▊     | 973/1184 [36:06<07:54,  2.25s/it][A
    Iteration:  82%|███████████████████████▊     | 974/1184 [36:09<07:52,  2.25s/it][A
    Iteration:  82%|███████████████████████▉     | 975/1184 [36:11<07:50,  2.25s/it][A
    Iteration:  82%|███████████████████████▉     | 976/1184 [36:13<07:44,  2.23s/it][A
    Iteration:  83%|███████████████████████▉     | 977/1184 [36:15<07:43,  2.24s/it][A
    Iteration:  83%|███████████████████████▉     | 978/1184 [36:18<07:38,  2.23s/it][A
    Iteration:  83%|███████████████████████▉     | 979/1184 [36:20<07:32,  2.21s/it][A
    Iteration:  83%|████████████████████████     | 980/1184 [36:22<07:34,  2.23s/it][A
    Iteration:  83%|████████████████████████     | 981/1184 [36:24<07:28,  2.21s/it][A
    Iteration:  83%|████████████████████████     | 982/1184 [36:26<07:24,  2.20s/it][A
    Iteration:  83%|████████████████████████     | 983/1184 [36:29<07:25,  2.22s/it][A
    Iteration:  83%|████████████████████████     | 984/1184 [36:31<07:28,  2.24s/it][A
    Iteration:  83%|████████████████████████▏    | 985/1184 [36:33<07:26,  2.24s/it][A
    Iteration:  83%|████████████████████████▏    | 986/1184 [36:35<07:20,  2.22s/it][A
    Iteration:  83%|████████████████████████▏    | 987/1184 [36:38<07:21,  2.24s/it][A
    Iteration:  83%|████████████████████████▏    | 988/1184 [36:40<07:20,  2.25s/it][A
    Iteration:  84%|████████████████████████▏    | 989/1184 [36:42<07:24,  2.28s/it][A
    Iteration:  84%|████████████████████████▏    | 990/1184 [36:44<07:18,  2.26s/it][A
    Iteration:  84%|████████████████████████▎    | 991/1184 [36:47<07:16,  2.26s/it][A
    Iteration:  84%|████████████████████████▎    | 992/1184 [36:49<07:14,  2.26s/it][A
    Iteration:  84%|████████████████████████▎    | 993/1184 [36:51<07:11,  2.26s/it][A
    Iteration:  84%|████████████████████████▎    | 994/1184 [36:53<07:08,  2.25s/it][A
    Iteration:  84%|████████████████████████▎    | 995/1184 [36:56<07:07,  2.26s/it][A
    Iteration:  84%|████████████████████████▍    | 996/1184 [36:58<06:59,  2.23s/it][A
    Iteration:  84%|████████████████████████▍    | 997/1184 [37:00<06:56,  2.23s/it][A
    Iteration:  84%|████████████████████████▍    | 998/1184 [37:02<06:57,  2.24s/it][A
    Iteration:  84%|████████████████████████▍    | 999/1184 [37:05<06:52,  2.23s/it][A
    Iteration:  84%|███████████████████████▋    | 1000/1184 [37:06<06:32,  2.13s/it][A
    Iteration:  85%|███████████████████████▋    | 1001/1184 [37:09<06:40,  2.19s/it][A
    Iteration:  85%|███████████████████████▋    | 1002/1184 [37:11<06:39,  2.19s/it][A
    Iteration:  85%|███████████████████████▋    | 1003/1184 [37:13<06:40,  2.21s/it][A
    Iteration:  85%|███████████████████████▋    | 1004/1184 [37:15<06:34,  2.19s/it][A
    Iteration:  85%|███████████████████████▊    | 1005/1184 [37:18<06:31,  2.19s/it][A
    Iteration:  85%|███████████████████████▊    | 1006/1184 [37:20<06:31,  2.20s/it][A
    Iteration:  85%|███████████████████████▊    | 1007/1184 [37:22<06:30,  2.20s/it][A
    Iteration:  85%|███████████████████████▊    | 1008/1184 [37:24<06:29,  2.21s/it][A
    Iteration:  85%|███████████████████████▊    | 1009/1184 [37:27<06:29,  2.22s/it][A
    Iteration:  85%|███████████████████████▉    | 1010/1184 [37:29<06:28,  2.23s/it][A
    Iteration:  85%|███████████████████████▉    | 1011/1184 [37:31<06:27,  2.24s/it][A
    Iteration:  85%|███████████████████████▉    | 1012/1184 [37:33<06:30,  2.27s/it][A
    Iteration:  86%|███████████████████████▉    | 1013/1184 [37:36<06:25,  2.26s/it][A
    Iteration:  86%|███████████████████████▉    | 1014/1184 [37:38<06:20,  2.24s/it][A
    Iteration:  86%|████████████████████████    | 1015/1184 [37:40<06:15,  2.22s/it][A
    Iteration:  86%|████████████████████████    | 1016/1184 [37:42<06:14,  2.23s/it][A
    Iteration:  86%|████████████████████████    | 1017/1184 [37:45<06:17,  2.26s/it][A
    Iteration:  86%|████████████████████████    | 1018/1184 [37:47<06:14,  2.26s/it][A
    Iteration:  86%|████████████████████████    | 1019/1184 [37:49<06:08,  2.23s/it][A
    Iteration:  86%|████████████████████████    | 1020/1184 [37:51<06:06,  2.24s/it][A
    Iteration:  86%|████████████████████████▏   | 1021/1184 [37:53<06:03,  2.23s/it][A
    Iteration:  86%|████████████████████████▏   | 1022/1184 [37:56<06:00,  2.22s/it][A
    Iteration:  86%|████████████████████████▏   | 1023/1184 [37:58<05:58,  2.23s/it][A
    Iteration:  86%|████████████████████████▏   | 1024/1184 [38:00<05:53,  2.21s/it][A
    Iteration:  87%|████████████████████████▏   | 1025/1184 [38:02<05:49,  2.20s/it][A
    Iteration:  87%|████████████████████████▎   | 1026/1184 [38:04<05:44,  2.18s/it][A
    Iteration:  87%|████████████████████████▎   | 1027/1184 [38:07<05:41,  2.18s/it][A
    Iteration:  87%|████████████████████████▎   | 1028/1184 [38:09<05:39,  2.18s/it][A
    Iteration:  87%|████████████████████████▎   | 1029/1184 [38:11<05:37,  2.18s/it][A
    Iteration:  87%|████████████████████████▎   | 1030/1184 [38:13<05:37,  2.19s/it][A
    Iteration:  87%|████████████████████████▍   | 1031/1184 [38:15<05:35,  2.20s/it][A
    Iteration:  87%|████████████████████████▍   | 1032/1184 [38:17<05:31,  2.18s/it][A
    Iteration:  87%|████████████████████████▍   | 1033/1184 [38:20<05:30,  2.19s/it][A
    Iteration:  87%|████████████████████████▍   | 1034/1184 [38:22<05:27,  2.18s/it][A
    Iteration:  87%|████████████████████████▍   | 1035/1184 [38:24<05:30,  2.22s/it][A
    Iteration:  88%|████████████████████████▌   | 1036/1184 [38:26<05:28,  2.22s/it][A
    Iteration:  88%|████████████████████████▌   | 1037/1184 [38:28<05:22,  2.20s/it][A
    Iteration:  88%|████████████████████████▌   | 1038/1184 [38:31<05:20,  2.19s/it][A
    Iteration:  88%|████████████████████████▌   | 1039/1184 [38:33<05:17,  2.19s/it][A
    Iteration:  88%|████████████████████████▌   | 1040/1184 [38:35<05:14,  2.18s/it][A
    Iteration:  88%|████████████████████████▌   | 1041/1184 [38:37<05:12,  2.19s/it][A
    Iteration:  88%|████████████████████████▋   | 1042/1184 [38:39<05:12,  2.20s/it][A
    Iteration:  88%|████████████████████████▋   | 1043/1184 [38:42<05:08,  2.19s/it][A
    Iteration:  88%|████████████████████████▋   | 1044/1184 [38:44<05:16,  2.26s/it][A
    Iteration:  88%|████████████████████████▋   | 1045/1184 [38:46<05:17,  2.28s/it][A
    Iteration:  88%|████████████████████████▋   | 1046/1184 [38:49<05:15,  2.28s/it][A
    Iteration:  88%|████████████████████████▊   | 1047/1184 [38:51<05:07,  2.25s/it][A
    Iteration:  89%|████████████████████████▊   | 1048/1184 [38:53<05:05,  2.25s/it][A
    Iteration:  89%|████████████████████████▊   | 1049/1184 [38:55<05:02,  2.24s/it][A
    Iteration:  89%|████████████████████████▊   | 1050/1184 [38:57<04:57,  2.22s/it][A
    Iteration:  89%|████████████████████████▊   | 1051/1184 [39:00<04:53,  2.21s/it][A
    Iteration:  89%|████████████████████████▉   | 1052/1184 [39:02<04:50,  2.20s/it][A
    Iteration:  89%|████████████████████████▉   | 1053/1184 [39:04<04:48,  2.21s/it][A
    Iteration:  89%|████████████████████████▉   | 1054/1184 [39:06<04:47,  2.21s/it][A
    Iteration:  89%|████████████████████████▉   | 1055/1184 [39:09<04:47,  2.23s/it][A
    Iteration:  89%|████████████████████████▉   | 1056/1184 [39:11<04:45,  2.23s/it][A
    Iteration:  89%|████████████████████████▉   | 1057/1184 [39:13<04:42,  2.22s/it][A
    Iteration:  89%|█████████████████████████   | 1058/1184 [39:15<04:40,  2.23s/it][A
    Iteration:  89%|█████████████████████████   | 1059/1184 [39:18<04:44,  2.27s/it][A
    Iteration:  90%|█████████████████████████   | 1060/1184 [39:20<04:39,  2.25s/it][A
    Iteration:  90%|█████████████████████████   | 1061/1184 [39:22<04:35,  2.24s/it][A
    Iteration:  90%|█████████████████████████   | 1062/1184 [39:24<04:31,  2.23s/it][A
    Iteration:  90%|█████████████████████████▏  | 1063/1184 [39:26<04:29,  2.22s/it][A
    Iteration:  90%|█████████████████████████▏  | 1064/1184 [39:29<04:31,  2.26s/it][A
    Iteration:  90%|█████████████████████████▏  | 1065/1184 [39:31<04:27,  2.25s/it][A
    Iteration:  90%|█████████████████████████▏  | 1066/1184 [39:33<04:30,  2.29s/it][A
    Iteration:  90%|█████████████████████████▏  | 1067/1184 [39:36<04:26,  2.28s/it][A
    Iteration:  90%|█████████████████████████▎  | 1068/1184 [39:38<04:20,  2.25s/it][A
    Iteration:  90%|█████████████████████████▎  | 1069/1184 [39:40<04:29,  2.34s/it][A
    Iteration:  90%|█████████████████████████▎  | 1070/1184 [39:43<04:28,  2.36s/it][A
    Iteration:  90%|█████████████████████████▎  | 1071/1184 [39:45<04:20,  2.31s/it][A
    Iteration:  91%|█████████████████████████▎  | 1072/1184 [39:47<04:15,  2.28s/it][A
    Iteration:  91%|█████████████████████████▍  | 1073/1184 [39:49<04:11,  2.27s/it][A
    Iteration:  91%|█████████████████████████▍  | 1074/1184 [39:52<04:09,  2.27s/it][A
    Iteration:  91%|█████████████████████████▍  | 1075/1184 [39:54<04:05,  2.25s/it][A
    Iteration:  91%|█████████████████████████▍  | 1076/1184 [39:56<04:06,  2.28s/it][A
    Iteration:  91%|█████████████████████████▍  | 1077/1184 [39:58<04:02,  2.27s/it][A
    Iteration:  91%|█████████████████████████▍  | 1078/1184 [40:01<03:59,  2.26s/it][A
    Iteration:  91%|█████████████████████████▌  | 1079/1184 [40:03<03:56,  2.25s/it][A
    Iteration:  91%|█████████████████████████▌  | 1080/1184 [40:05<03:52,  2.23s/it][A
    Iteration:  91%|█████████████████████████▌  | 1081/1184 [40:07<03:47,  2.21s/it][A
    Iteration:  91%|█████████████████████████▌  | 1082/1184 [40:10<03:45,  2.21s/it][A
    Iteration:  91%|█████████████████████████▌  | 1083/1184 [40:12<03:42,  2.20s/it][A
    Iteration:  92%|█████████████████████████▋  | 1084/1184 [40:14<03:40,  2.20s/it][A
    Iteration:  92%|█████████████████████████▋  | 1085/1184 [40:16<03:36,  2.19s/it][A
    Iteration:  92%|█████████████████████████▋  | 1086/1184 [40:18<03:36,  2.21s/it][A
    Iteration:  92%|█████████████████████████▋  | 1087/1184 [40:21<03:36,  2.24s/it][A
    Iteration:  92%|█████████████████████████▋  | 1088/1184 [40:23<03:35,  2.25s/it][A
    Iteration:  92%|█████████████████████████▊  | 1089/1184 [40:25<03:35,  2.27s/it][A
    Iteration:  92%|█████████████████████████▊  | 1090/1184 [40:28<03:35,  2.29s/it][A
    Iteration:  92%|█████████████████████████▊  | 1091/1184 [40:30<03:30,  2.27s/it][A
    Iteration:  92%|█████████████████████████▊  | 1092/1184 [40:32<03:25,  2.23s/it][A
    Iteration:  92%|█████████████████████████▊  | 1093/1184 [40:34<03:23,  2.23s/it][A
    Iteration:  92%|█████████████████████████▊  | 1094/1184 [40:36<03:18,  2.21s/it][A
    Iteration:  92%|█████████████████████████▉  | 1095/1184 [40:38<03:15,  2.20s/it][A
    Iteration:  93%|█████████████████████████▉  | 1096/1184 [40:41<03:12,  2.19s/it][A
    Iteration:  93%|█████████████████████████▉  | 1097/1184 [40:43<03:16,  2.26s/it][A
    Iteration:  93%|█████████████████████████▉  | 1098/1184 [40:45<03:13,  2.25s/it][A
    Iteration:  93%|█████████████████████████▉  | 1099/1184 [40:47<03:07,  2.21s/it][A
    Iteration:  93%|██████████████████████████  | 1100/1184 [40:50<03:04,  2.20s/it][A
    Iteration:  93%|██████████████████████████  | 1101/1184 [40:52<03:01,  2.19s/it][A
    Iteration:  93%|██████████████████████████  | 1102/1184 [40:54<02:59,  2.19s/it][A
    Iteration:  93%|██████████████████████████  | 1103/1184 [40:56<02:58,  2.21s/it][A
    Iteration:  93%|██████████████████████████  | 1104/1184 [40:58<02:57,  2.22s/it][A
    Iteration:  93%|██████████████████████████▏ | 1105/1184 [41:01<02:55,  2.22s/it][A
    Iteration:  93%|██████████████████████████▏ | 1106/1184 [41:03<02:53,  2.23s/it][A
    Iteration:  93%|██████████████████████████▏ | 1107/1184 [41:05<02:51,  2.23s/it][A
    Iteration:  94%|██████████████████████████▏ | 1108/1184 [41:07<02:50,  2.24s/it][A
    Iteration:  94%|██████████████████████████▏ | 1109/1184 [41:10<02:46,  2.21s/it][A
    Iteration:  94%|██████████████████████████▎ | 1110/1184 [41:12<02:44,  2.22s/it][A
    Iteration:  94%|██████████████████████████▎ | 1111/1184 [41:14<02:42,  2.22s/it][A
    Iteration:  94%|██████████████████████████▎ | 1112/1184 [41:16<02:38,  2.21s/it][A
    Iteration:  94%|██████████████████████████▎ | 1113/1184 [41:18<02:35,  2.19s/it][A
    Iteration:  94%|██████████████████████████▎ | 1114/1184 [41:21<02:32,  2.18s/it][A
    Iteration:  94%|██████████████████████████▎ | 1115/1184 [41:23<02:30,  2.18s/it][A
    Iteration:  94%|██████████████████████████▍ | 1116/1184 [41:25<02:27,  2.17s/it][A
    Iteration:  94%|██████████████████████████▍ | 1117/1184 [41:27<02:27,  2.20s/it][A
    Iteration:  94%|██████████████████████████▍ | 1118/1184 [41:29<02:24,  2.19s/it][A
    Iteration:  95%|██████████████████████████▍ | 1119/1184 [41:31<02:21,  2.18s/it][A
    Iteration:  95%|██████████████████████████▍ | 1120/1184 [41:34<02:18,  2.16s/it][A
    Iteration:  95%|██████████████████████████▌ | 1121/1184 [41:36<02:16,  2.17s/it][A
    Iteration:  95%|██████████████████████████▌ | 1122/1184 [41:38<02:15,  2.19s/it][A
    Iteration:  95%|██████████████████████████▌ | 1123/1184 [41:40<02:14,  2.21s/it][A
    Iteration:  95%|██████████████████████████▌ | 1124/1184 [41:42<02:13,  2.22s/it][A
    Iteration:  95%|██████████████████████████▌ | 1125/1184 [41:45<02:11,  2.23s/it][A
    Iteration:  95%|██████████████████████████▋ | 1126/1184 [41:47<02:08,  2.21s/it][A
    Iteration:  95%|██████████████████████████▋ | 1127/1184 [41:49<02:05,  2.21s/it][A
    Iteration:  95%|██████████████████████████▋ | 1128/1184 [41:51<02:04,  2.22s/it][A
    Iteration:  95%|██████████████████████████▋ | 1129/1184 [41:54<02:03,  2.25s/it][A
    Iteration:  95%|██████████████████████████▋ | 1130/1184 [41:56<02:02,  2.26s/it][A
    Iteration:  96%|██████████████████████████▋ | 1131/1184 [41:58<02:00,  2.27s/it][A
    Iteration:  96%|██████████████████████████▊ | 1132/1184 [42:00<01:56,  2.25s/it][A
    Iteration:  96%|██████████████████████████▊ | 1133/1184 [42:03<01:53,  2.22s/it][A
    Iteration:  96%|██████████████████████████▊ | 1134/1184 [42:05<01:50,  2.20s/it][A
    Iteration:  96%|██████████████████████████▊ | 1135/1184 [42:07<01:47,  2.19s/it][A
    Iteration:  96%|██████████████████████████▊ | 1136/1184 [42:09<01:44,  2.19s/it][A
    Iteration:  96%|██████████████████████████▉ | 1137/1184 [42:11<01:42,  2.19s/it][A
    Iteration:  96%|██████████████████████████▉ | 1138/1184 [42:13<01:40,  2.19s/it][A
    Iteration:  96%|██████████████████████████▉ | 1139/1184 [42:16<01:40,  2.23s/it][A
    Iteration:  96%|██████████████████████████▉ | 1140/1184 [42:18<01:38,  2.24s/it][A
    Iteration:  96%|██████████████████████████▉ | 1141/1184 [42:20<01:36,  2.24s/it][A
    Iteration:  96%|███████████████████████████ | 1142/1184 [42:23<01:33,  2.23s/it][A
    Iteration:  97%|███████████████████████████ | 1143/1184 [42:25<01:33,  2.28s/it][A
    Iteration:  97%|███████████████████████████ | 1144/1184 [42:27<01:30,  2.26s/it][A
    Iteration:  97%|███████████████████████████ | 1145/1184 [42:29<01:28,  2.28s/it][A
    Iteration:  97%|███████████████████████████ | 1146/1184 [42:32<01:29,  2.37s/it][A
    Iteration:  97%|███████████████████████████▏| 1147/1184 [42:34<01:25,  2.31s/it][A
    Iteration:  97%|███████████████████████████▏| 1148/1184 [42:36<01:21,  2.27s/it][A
    Iteration:  97%|███████████████████████████▏| 1149/1184 [42:39<01:19,  2.26s/it][A
    Iteration:  97%|███████████████████████████▏| 1150/1184 [42:41<01:16,  2.24s/it][A
    Iteration:  97%|███████████████████████████▏| 1151/1184 [42:43<01:13,  2.23s/it][A
    Iteration:  97%|███████████████████████████▏| 1152/1184 [42:45<01:10,  2.21s/it][A
    Iteration:  97%|███████████████████████████▎| 1153/1184 [42:47<01:08,  2.20s/it][A
    Iteration:  97%|███████████████████████████▎| 1154/1184 [42:50<01:05,  2.19s/it][A
    Iteration:  98%|███████████████████████████▎| 1155/1184 [42:52<01:03,  2.19s/it][A
    Iteration:  98%|███████████████████████████▎| 1156/1184 [42:54<01:01,  2.19s/it][A
    Iteration:  98%|███████████████████████████▎| 1157/1184 [42:56<00:59,  2.20s/it][A
    Iteration:  98%|███████████████████████████▍| 1158/1184 [42:58<00:57,  2.22s/it][A
    Iteration:  98%|███████████████████████████▍| 1159/1184 [43:01<01:01,  2.45s/it][A
    Iteration:  98%|███████████████████████████▍| 1160/1184 [43:03<00:56,  2.35s/it][A
    Iteration:  98%|███████████████████████████▍| 1161/1184 [43:06<00:52,  2.29s/it][A
    Iteration:  98%|███████████████████████████▍| 1162/1184 [43:08<00:50,  2.27s/it][A
    Iteration:  98%|███████████████████████████▌| 1163/1184 [43:10<00:47,  2.26s/it][A
    Iteration:  98%|███████████████████████████▌| 1164/1184 [43:12<00:45,  2.26s/it][A
    Iteration:  98%|███████████████████████████▌| 1165/1184 [43:15<00:42,  2.23s/it][A
    Iteration:  98%|███████████████████████████▌| 1166/1184 [43:17<00:39,  2.21s/it][A
    Iteration:  99%|███████████████████████████▌| 1167/1184 [43:19<00:37,  2.19s/it][A
    Iteration:  99%|███████████████████████████▌| 1168/1184 [43:21<00:34,  2.18s/it][A
    Iteration:  99%|███████████████████████████▋| 1169/1184 [43:23<00:32,  2.18s/it][A
    Iteration:  99%|███████████████████████████▋| 1170/1184 [43:25<00:30,  2.17s/it][A
    Iteration:  99%|███████████████████████████▋| 1171/1184 [43:27<00:28,  2.16s/it][A
    Iteration:  99%|███████████████████████████▋| 1172/1184 [43:30<00:26,  2.17s/it][A
    Iteration:  99%|███████████████████████████▋| 1173/1184 [43:32<00:23,  2.17s/it][A
    Iteration:  99%|███████████████████████████▊| 1174/1184 [43:34<00:21,  2.19s/it][A
    Iteration:  99%|███████████████████████████▊| 1175/1184 [43:36<00:20,  2.23s/it][A
    Iteration:  99%|███████████████████████████▊| 1176/1184 [43:39<00:17,  2.23s/it][A
    Iteration:  99%|███████████████████████████▊| 1177/1184 [43:41<00:15,  2.23s/it][A
    Iteration:  99%|███████████████████████████▊| 1178/1184 [43:43<00:13,  2.22s/it][A
    Iteration: 100%|███████████████████████████▉| 1179/1184 [43:45<00:11,  2.21s/it][A
    Iteration: 100%|███████████████████████████▉| 1180/1184 [43:48<00:08,  2.25s/it][A
    Iteration: 100%|███████████████████████████▉| 1181/1184 [43:50<00:06,  2.28s/it][A
    Iteration: 100%|███████████████████████████▉| 1182/1184 [43:52<00:04,  2.27s/it][A
    Iteration: 100%|███████████████████████████▉| 1183/1184 [43:54<00:02,  2.27s/it][A
    Iteration: 100%|████████████████████████████| 1184/1184 [43:57<00:00,  2.23s/it][A
    Epoch: 100%|██████████████████████████████████| 2/2 [1:28:11<00:00, 2645.71s/it]
    09/08/2022 15:50:51 - INFO - __main__ -    global_step = 2368, average loss = 2.2925813325454256
    09/08/2022 15:50:51 - INFO - __main__ -   Saving model checkpoint to output
    09/08/2022 15:50:56 - INFO - __main__ -   Evaluate the following checkpoints: ['output']
    09/08/2022 15:51:00 - INFO - __main__ -   Loading features from cached file input_data/gpt2-medium_cached_lm_200_val_harry.txt
    09/08/2022 15:51:00 - INFO - __main__ -   ***** Running evaluation  *****
    09/08/2022 15:51:00 - INFO - __main__ -     Num examples = 418
    09/08/2022 15:51:00 - INFO - __main__ -     Batch size = 4
    Evaluating: 100%|█████████████████████████████| 105/105 [02:40<00:00,  1.53s/it]
    09/08/2022 15:53:40 - INFO - __main__ -   ***** Eval results  *****
    09/08/2022 15:53:40 - INFO - __main__ -     perplexity = tensor(12.7392)



```python
!python run_generation.py --model_type gpt2 --model_name_or_path output --length 300 --prompt "Malfoy hadn’t noticed anything."
```

    /home/user/juputer_env/lib/python3.9/site-packages/torch/cuda/__init__.py:83: UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 10010). Please update your GPU driver by downloading and installing a new version from the URL: http://www.nvidia.com/Download/index.aspx Alternatively, go to: https://pytorch.org to install a PyTorch version that has been compiled with your version of the CUDA driver. (Triggered internally at  ../c10/cuda/CUDAFunctions.cpp:109.)
      return torch._C._cuda_getDeviceCount() > 0
    09/08/2022 16:02:47 - INFO - __main__ -   Namespace(model_type='gpt2', prompt='Malfoy hadn’t noticed anything.', padding_text='', xlm_lang='', length=300, num_samples=1, temperature=1.0, repetition_penalty=1.0, top_k=0, top_p=0.9, no_cuda=False, seed=42, stop_token=None, model_name_or_path='output', device=device(type='cpu'), n_gpu=0)
    100%|█████████████████████████████████████████| 300/300 [01:59<00:00,  2.51it/s]
    ”
    
    “Could you believe he’s still lying?” said Aunt
    
    Petunia.
    
    “Sister, what are we going to do?”
    
     
    
    “We’re going to speak to Harry about this,” said Aunt
    
    Petunia grimly. “He’s sick. He couldn’t take Snape’s
    
    spell....”
    
    “Better talk to him in private, won’t you, dear?” said
    
    Dudley impatiently.
    
    “Dudley, the people around here want to see a bit
    
    more of Harry,” said Aunt Petunia angrily. “It’s the only
    
    way they’ll find out what he’s up to....”
    
    “I think Harry needs to see you,” said Aunt Petunia gently
    
    as Harry slid into bed. He had managed to remember
    
    what she’d said about using Leg-Locker before he had
    
    entered the room and she seemed to think it had been so funny
    
    that he’d forgotten all about it.
    
    There was a knock on the door and Dudley dashed in, but
    
    it wasn’t Harry. It was Madam Malkin, the little librarian
    
    who lived



```python
!pwd
```

    /home/user/ki/Deep-Learning/GPT2-HarryPotter-Training/examples



```python

```
