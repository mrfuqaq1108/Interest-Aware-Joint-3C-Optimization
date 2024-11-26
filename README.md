# Related Paper
- Baojie Fu, Tong Tang, Dapeng Wu, and Ruyan Wang, Interest-Aware Joint Caching, Computing, and Communication Optimization for Mobile VR Delivery in MEC Networks (2024). arXiv: 2403.05851.
- B. Fu, T. Tang, D. Wu et al., Interest-aware joint caching, computing, and communication optimization for mobile VR delivery in MEC networks, Digital Communications and Networks, doi: https://doi.org/10.1016/j.dcan.2024.10.018.

- Paper Available Website: https://www.sciencedirect.com/science/article/pii/S2352864824001469

***Any use of the codes should explicitly cite the aforementioned paper.***

# Dataset
- IMDB dataset, which is available online at https://ai.stanford.edu/~amaas/data/sentiment/
- Amazon Review Data (2018) dataset, which is available online at https://nijianmo.github.io/amazon/index.html

# Documentation

## sentiment_analysis_DCN
**Solving the request probability problem for each user.**
 
Four common methods for analyzing the text are used and compared:

|  Method  | Pytorch code | Our training model | Our numerical result | training accuracy | validation accuracy | testing accuracy |
|:--------:|:------------:|:------------------:|:--------------------:|:-----------------:|:-------------------:|:----------------:|
| TextCNN  |    CNN.py    |    CNN-model.pt    |    CNN_result.py     |    **94.17%**     |       87.26%        |      85.89%      |
| TextRNN  |    RNN.py    |    RNN-model.pt    |    RNN_result.py     |      86.66%       |       88.79%        |      87.85%      |
| FastText | Fasttext.py  | Fasttext-model.pt  |  Fasttext_result.py  |      86.87%       |       85.66%        |      85.45%      |
|   Bert   |   Bert.py    |   Bert-model.pt    |    Bert_result.py    |      93.07%       |     **91.23%**      |    **91.87%**    |

Hence, we choose **Bert model**  to acquire the users’ request probability matrix.

Without loss of generality, we choose 10 contents from the Amazon Review Data (2018) dataset. 
We define the "popularity" as the normalized number of comments per content.
For each chosen content, we choose 5 comments.

**prob_acquire.py**: Pytorch code for sentiment analysis from the dataset using Bert model.

**prob_matrix.py**: Numerical result (without normalized).

For our training model, you can download at: 
https://drive.google.com/drive/folders/19BlMoVZ8_dLkbxxcrZJdvPTtjBGlwNR6?usp=drive_link

You should download bert-base-uncased yourself, which contains:
- config.json
- pytorch_model.bin
- vocab.txt

You should download glove.6B.zip in the .vector_cache, and imdb dataset in the .data.

## manuscript_code

**para.py** contains four different distribution.
- **random_prob**: Random distribution.
- **real_prob**: Our Bert model, which is normalized from **prob_matrix.py**.
- **avg_prob**: Uniform distribution.
- **zif_prob**: Zipf distribution (according to the popularity from **prob_matrix.py**).

**function_algorithm1.py**: Corresponding to **Algorithm 1 Caching and Computing Policy Design** in our work.
Through this code, we can obtain the optimal caching and computing policy under the given bandwidth distribution, thus problem (P3) can be solved.

**function_cost2a.py**: Corresponding to the first 6 lines of **Algorithm 2 Bandwidth Allocation Policy Design** in our work.
Through this code, we can obtain a specific bandwidth for each user under the given caching and computing policy.

**function_algorithm2.py**: Corresponding to the remaining of **Algorithm 2 Bandwidth Allocation Policy Design** in our work.
Through this code, we can obtain a optimal bandwidth distribution under the given caching and computing policy, thus problem (P4) can be solved.

**function_algorithm3.py**: Corresponding to **Algorithm 3 Joint 3C Optimization Policy Design** in our work.
Through this code, we can solve problem (P1) by iterating **Algorithm 1** and **Algorithm 2** alternatively.

**T_E.py**: Since our optimization goal, cost, is depicted as the weighted sum of overall VR delivery delay and energy consumption of local device, 
we can use this code to separately calculate the delay T, energy consumption E, and cost of VR services.

### Analysis on the cache scheme (Fig. 3)
- **compare_rand.py**: Cost of random distribution.
- **compare_real.py**: Cost of our sentiment analysis.
- **compare_avg.py**: Cost of uniform distribution.
- **compare_zif.py**: Cost of zipf distribution.
---
- **different_prob_Cu**:  Numerical result under different cache schemes with different local cache capability.

### Convergence of Algorithm 3 (Fig. 2)
- **convergence.py**: Record the convergence of the main loop of Algorithm 3 using **compare_real.py**.

### Analysis on user fairness (Fig. 4)
- **fairness.py**: Output "Max Cost" and "Min Cost".
- **compare_real.py**: Output "Fairness (Proposed)".
---
- **fairness_fc.py**: Numerical results under different computing capability.
- **fairness_Cu.py**: Numerical results under different local cache capability.
- **fairness_multiuser.py**: Numerical results under different number of users.
- **fairness_B.py**: Numerical results under different available bandwidth.

### Analysis on joint 3C optimization scheme
#### Different Scheme (Fig. 5)
- **scheme1_greedy_edge.py**: Code for scheme 1 (Greedy edge computing).
- **scheme2_greedy_loc_without.py**: Code for scheme 2 (Greedy local computing without caching).
- **scheme3_cooperative_without.py**: Code for scheme 3 (Joint 3C policy without caching).
- **compare_real.py**: Our proposed Joint 3C policy.
---
- **scheme_fc.py**: Numerical results under different computing capability.
- **scheme_Cu.py**: Numerical results under different local cache capability.
- **scheme_multiuser.py**: Numerical results under different number of users.
- **scheme_B.py**: Numerical results under different available bandwidth.

#### Impacts on VR service delay (Fig. 6)
- **bar_t_fc.py**: Numerical results under different computing capability on VR service delay.
- **bar_t_Cu.py**: Numerical results under different local cache capability on VR service delay.
- **bar_t_multiuser.py**: Numerical results under different number of users on VR service delay.
- **bar_t_B.py**: Numerical results under different available bandwidth on VR service delay.

#### Impacts on local VR device energy consumption (Fig. 7)
- **bar_e_fc.py**: Numerical results under different computing capability on local VR device energy consumption.
- **bar_e_Cu.py**: Numerical results under different local cache capability on local VR device energy consumption.
- **bar_e_multiuser.py**: Numerical results under different number of users on local VR device energy consumption.
- **bar_e_B.py**: Numerical results under different available bandwidth on local VR device energy consumption.
