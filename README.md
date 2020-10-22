# DNN噪声标签相关论文

## 综述

- [ ] [Learning from Noisy Labels with Deep Neural Networks: A Survey](http://www.researchgate.net/publication/343005449_Learning_from_Noisy_Labels_with_Deep_Neural_Networks_A_Survey)

| 类型                     | 方法                                                         |
| ------------------------ | ------------------------------------------------------------ |
| Robust Loss Function     | Robust MAE, Generalized Cross Entropy,Symmetric Cross Entropy, Curriculum Loss |
| Robust Architecture      | Webly Learning, Noise Model, Dropout Noise Model, S–model , C–model  , NLNN , Probabilistic Noise Model , Masking, Contrastive-Additive Noise Network |
| Robust Regularization    | Adversarial Training , Label Smoothing , Mixup , Bilevel Learning , Annotator Confusion , Pre-training |
| Loss Adjustment          | Gold Loss Correction, Importance Reweighting, Active Bias , Bootstrapping, Dynamic Bootstrapping , D2L , SELFIE |
| Sample Selection         | Decouple , MentorNet , Co-teaching , Co-teaching+, Iterative Detection , ITLM , INCV , SELFIE , SELF , Curriculum Loss |
| Meta Learning            | Meta-Regressor , Knowledge Distillation , L2LWS , CWS , Automatic Reweighting , MLNT , Meta-Weight-Net, Data Coefficients |
| Semi-supervised Learning | Label Aggregation , Two-Stage Framework , SELF , DivideMix   |

### Robust Loss Function

  - [x] (**Robust MAE**) Robust Loss Functions under Label Noise for Deep Neural Networks [pdf](http://arxiv.org/pdf/1712.09482)
  - [x] (**Generalized Cross Entropy**) Generalized cross entropy loss for training deep neural networks with noisy labels [pdf](http://arxiv.org/pdf/1805.07836)
  - [x] (**Symmetric Cross Entropy** )Symmetric cross entropy for robust learning with noisy labels [pdf](http://ieeexplore.ieee.org/document/9010653/)
  - [ ] (**Curriculum Loss** ) Curriculum loss: Robust learning and generalization against label corruption[pdf](http://arxiv.org/abs/1905.10045)

### Robust Architecture

- [ ] (**Webly Learning**) Webly supervised learning of convolutional networks  [pdf](https://arxiv.org/pdf/1505.01554.pdf)
- [ ] (**Noise Model**)Training convolutional networks with noisy labels [pdf](http://de.arxiv.org/pdf/1406.2080)
- [ ] (**Dropout Noise Model**)Learning deep networks from noisy labels with dropout regularization [pdf](https://arxiv.org/pdf/1705.03419.pdf)
- [ ] (**S–model**) (**C–model**)Training deep neural-networks using a noise adaptation layer [pdf](https://openreview.net/pdf?id=H12GRgcxg)
- [x] (**NLNN**)Training deep neural-networks based on unreliable labels [pdf](http://www.eng.biu.ac.il/goldbej/files/2012/05/icassp_2016_Alan.pdf)
- [ ] (**Probabilistic Noise Model**) Learning from massive noisy labeled data for image classification [pdf](http://www.ee.cuhk.edu.hk/~xgwang/papers/xiaoXYHWcvpr15.pdf)
- [ ] **(Masking**) Masking: A new perspective of noisy supervision [pdf](https://arxiv.org/abs/1805.08193)
- [ ] (**Contrastive-Additive Noise Network**) Deep learning from noisy image labels with quality embedding [pdf](https://arxiv.org/abs/1711.00583)

### Robust Regularization

- [ ] (**Adversarial Training)** Explaining and harnessing adversarial examples [pdf](https://arxiv.org/pdf/1412.6572.pdf) 
- [ ] (**Label Smoothing**) Regularizing neural networks by penalizing confident output distributions [pdf](https://arxiv.org/pdf/1701.06548.pdf)
- [ ] (**Mixup**)mixup: Beyond empirical risk minimization [pdf](https://arxiv.org/pdf/1710.09412.pdf)
- [ ] (**Bilevel Learning**)Deep bilevel learning[pdf](https://arxiv.org/pdf/1809.01465.pdf)
- [ ] (**Annotator Confusion**) Learning from noisy labels by regularized estimation of annotator confusion [pdf](https://ieeexplore.ieee.org/document/8953406/)
- [ ] (**Pre-training**) Using Pre-Training Can Improve Model Robustness and Uncertainty [pdf](https://arxiv.org/abs/1901.09960?context=cs.CV)

### Loss Adjustment

- [ ] (**Gold Loss Correction**) Using trusted data to train deep networks on labels corrupted by severe noise[pdf](http://arxiv.org/pdf/1802.05300)
- [ ] (**Importance Reweighting**) Multiclass learning with partially corrupted labels [pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7929355)
- [ ] **(Active Bias**) Active Bias: Training more accurate neural networks by emphasizing high variance  samples [pdf](https://arxiv.org/abs/1704.07433?context=cs.LG)
- [ ] (**Bootstrapping**) Training deep neural networks on noisy labels with bootstrapping [pdf](http://de.arxiv.org/pdf/1412.6596)
- [ ] (**Dynamic Bootstrapping**) Unsupervised label noise modeling and loss correction [pdf](https://arxiv.org/abs/1904.11238v1)
- [ ] (**D2L**) Dimensionality-driven learning with noisy labels [pdf](http://arxiv.org/pdf/1806.02612)
- [ ] (**SELFIE** ) SELFIE: Refurbishing unclean samples for robust deep learning [pdf](https://www.researchgate.net/publication/332779371_SELFIE_Refurbishing_Unclean_Samples_for_Robust_Deep_Learning)

### Sample Selection

- [ ] **Decouple** “Decoupling” when to update” from” how to update”  [pdf](https://arxiv.org/abs/1706.02613)
- [ ] **MentorNet** MentorNet: Learning data-driven curriculum for very deep neural networks on  corrupted label  [pdf](https://arxiv.org/pdf/1712.05055v2.pdf)
- [ ] **Co-teaching** Co-teaching: Robust training of deep neural networks with extremely noisy labels [pdf](https://arxiv.org/abs/1804.06872)
- [ ] **Co-teaching+** How does disagreement help generalization against label corruption?  [pdf](https://arxiv.org/abs/1901.04215)
- [ ] **Iterative** Detection Iterative learning with open-set noisy labels  [pdf](https://arxiv.org/abs/1804.00092)
- [ ] **ITLM**  Iterative learning with open-set noisy labels [pdf](https://arxiv.org/abs/1804.00092)
- [ ] **INCV** Understanding and utilizing deep neural networks trained with noisy labels  [pdf](https://arxiv.org/abs/1905.05040)
- [ ] **SELFIE**  SELFIE: Refurbishing unclean samples for robust deep learning [pdf](http://proceedings.mlr.press/v97/song19b/song19b.pdf)
- [ ] **SELF** Self: Learning to filter noisy labels with self-ensembling [pdf](http://arxiv.org/abs/1910.01842v1) 
- [ ] **Curriculum Loss** Curriculum loss: Robust learning and generalization against label corruption [pdf](https://arxiv.org/abs/1905.10045)

### Meta Learning

- [ ] **Meta-Regressor** Noise detection in the meta-learning level [pdf](https://dl.acm.org/doi/10.1016/j.neucom.2014.12.100)
- [ ] **Knowledge Distillation** Learning from noisy labels with distillation [pdf](https://arxiv.org/abs/1703.02391)
- [ ] **L2LWS** Learning to Learn from Weak Supervision by Full Supervision [pdf](https://arxiv.org/abs/1711.11383)
- [ ] **CWS** Avoiding your teacher’s mistakes: Training neural networks with controlled weak supervision [pdf](https://arxiv.org/abs/1711.00313)
- [ ] **Automatic Reweighting**  Learning to reweight examples for robust deep learning [pdf](https://arxiv.org/abs/1711.00313)
- [ ] **MLNT** Learning to learn from noisy labeled data [pdf](https://arxiv.org/pdf/1812.05214.pdf)
- [ ] **Meta-Weight-Net** MetaWeight-Net: Learning an explicit mapping for sample weighting [pdf](https://arxiv.org/abs/1902.07379)
- [ ] **Data Coefficients** MetaWeight-Net: Learning an explicit mapping for sample weighting [pdf](https://arxiv.org/abs/1902.07379)

### Semi-supervised Learning

- [ ] **Label Aggregation** Robust semisupervised learning through label aggregation [pdf](http://zhongwen.ai/pdf/ROSSEL.pdf) 
- [ ] **Two-Stage Framework** A semi-supervised two-stage approach to learning from noisy labels [pdf](http://arxiv.org/pdf/1802.02679)
- [ ] **SELF** Self: Learning to filter noisy labels with self-ensembling [pdf](https://arxiv.org/abs/1910.01842v1)
- [ ] **DivideMix** DivideMix: Learning with noisy labels as semi-supervised learning [pdf](http://arxiv.org/abs/2002.07394?context=cs.CV)



## 其他


- [x] Robust deep supervised hashing for image retrieval

## ICLR 2021
- [ ] NOISE-ROBUST CONTRASTIVE LEARNING [pdf](https://openreview.net/pdf?id=D1E1h-K3jso)
- [ ] ROBUST CURRICULUM LEARNING: FROM CLEAN LABEL DETECTION TO NOISY LABEL SELF-CORRECTION [pdf](https://openreview.net/pdf?id=lmTWnm3coJJ)
- [ ] ROBUST LEARNING VIA GOLDEN SYMMETRIC LOSS OF (UN)TRUSTED LABELS [pdf](https://openreview.net/pdf?id=20qC5K2ICZL)
- [ ] LONG-TAIL ZERO AND FEW-SHOT LEARNING VIA CONTRASTIVE PRETRAINING ON AND FOR SMALL DATA [pdf](https://openreview.net/pdf?id=_cadenVdKzF)
- [ ] A SIMPLE APPROACH TO DEFINE CURRICULA FOR TRAINING NEURAL NETWORKS  [pdf](https://openreview.net/pdf?id=SVP44gujOBL)
- [ ] INTERACTIVE WEAK SUPERVISION: LEARNING USEFUL HEURISTICS FOR DATA LABELING [pdf](https://openreview.net/pdf?id=IDFQI9OY6K)
- [ ] PROTOTYPICAL REPRESENTATION LEARNING FOR RELATION EXTRACTION [pdf](https://openreview.net/pdf?id=aCgLmfhIy_f)
- [ ] NEIGHBOR CLASS CONSISTENCY ON UNSUPERVISEDDOMAIN ADAPTATION [pdf](https://openreview.net/pdf?id=defQ1AG6IWn)
- [ ] ON THE IMPORTANCE OF LOOKING AT THE MANIFOLD [pdf](https://openreview.net/pdf?id=zFM0Uo_GnYE)
- [ ] SHAPE OR TEXTURE: UNDERSTANDING DISCRIMINATIVE FEATURES IN CNNS [pdf](https://openreview.net/pdf?id=NcFEZOi-rLa)
- [ ] LEARNING TO EXPLORE WITH PLEASURE [pdf](https://openreview.net/pdf?id=XqQQlvHvtI)
- [ ]  [pdf]()
- [ ]  [pdf]()
- [ ]  [pdf]()


