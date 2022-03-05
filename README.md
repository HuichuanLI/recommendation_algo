# recommendation_algo

推荐算法深度学习实现

- General Recommendation

| ** 模型 **  | 会议  |论文 |年份 |
| :---  |  :--- | :---  |  :--- |
| KNN| TOIS    |《Item-based top-N recommendation algorithms》|2005
| popularity| -    |- |-
| BPR | UAI    |《BPR: Bayesian Personalized Ranking from Implicit Feedback》|2009
| NeuMF    | TWWW|《Neural Collaborative Filtering》|2017
| ConvNCF | IJCAI    |《Outer Product-based Neural Collaborative Filtering》|2017
| DMF | IJCAI    |《Deep Matrix Factorization Models for Recommender Systems》|2017
| FISM     | SIGKDD        |《FISM: Factored Item Similarity Models for Top-N Recommender Systems》|2013
|NAIS|TKDE        |《NAIS: Neural Attentive Item Similarity Model for Recommendation》|2018
|SpectralCF|RecSys        |《Spectral Collaborative Filtering》|2018
|GCMC|    SIGKDD    |《Graph Convolutional Matrix Completion》|    2018
|NGCF    |SIGIR        |《Neural Graph Collaborative Filtering》|2019
|LightGCN    |SIGIR    |    《LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation》|2020
|DGCF    |SIGIR        |《Disentangled Graph Collaborative Filtering》|2020
|MultiVAE    |WWW        |《Variational Autoencoders for Collaborative Filtering》|2018
|MultiDAE    |WWW        |《Variational Autoencoders for Collaborative Filtering》|2018
|CDAE    |WSDM        |《Collaborative denoising auto-encoders for top-n recommender systems》|2016
|MacridVAE    |NeurIPS |《Learning Disentangled Representations for Recommendation》|2019
|LINE    |WWW        |《Large-scale Information Network Embedding》|2015
|EASE    |WWW        |《Embarrassingly Shallow Autoencoders for Sparse Data》|2019
|RaCT    |ICLR        |《RaCT: Towards Amortized Ranking-Critical Training for Collaborative Filtering.》|2020
|RecVAE    |WSDM        |《RecVAE: A new variational autoencoder for Top-N recommendations with implicit feedback.》|2020
|NNCF    |CIKM        |《A Neural Collaborative Filtering Model with Interaction-based Neighborhood.》|2017
|ENMF    |TOIS        |《Efficient Neural Matrix Factorization without Sampling for Recommendation.》|2020
|SLIMElastic    |ICDM        |《SLIM: Sparse Linear Methods for Top-N Recommender Systems》|2011


- Context-aware Recommendation

|                 Model                  | Paper                                                                                                                                                           |
| :------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  Convolutional Click Prediction Model  | [CIKM 2015][A Convolutional Click Prediction Model](http://ir.ia.ac.cn/bitstream/173211/12337/1/A%20Convolutional%20Click%20Prediction%20Model.pdf)             |
| Factorization-supported Neural Network | [ECIR 2016][Deep Learning over Multi-field Categorical Data: A Case Study on User Response Prediction](https://arxiv.org/pdf/1601.02376.pdf)                    |
|      Product-based Neural Network      | [ICDM 2016][Product-based neural networks for user response prediction](https://arxiv.org/pdf/1611.00144.pdf)                                                   |
|              Wide & Deep               | [DLRS 2016][Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf)                                                                 |
|                 DeepFM                 | [IJCAI 2017][DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](http://www.ijcai.org/proceedings/2017/0239.pdf)                           |
|        Piece-wise Linear Model         | [arxiv 2017][Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction](https://arxiv.org/abs/1704.05194)                                 |
|          Deep & Cross Network          | [ADKDD 2017][Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123)                                                                   |
|   Attentional Factorization Machine    | [IJCAI 2017][Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](http://www.ijcai.org/proceedings/2017/435) |
|      Neural Factorization Machine      | [SIGIR 2017][Neural Factorization Machines for Sparse Predictive Analytics](https://arxiv.org/pdf/1708.05027.pdf)                                               |
|                xDeepFM                 | [KDD 2018][xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/pdf/1803.05170.pdf)                         |
|         Deep Interest Network          | [KDD 2018][Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1706.06978.pdf)     |
|                AutoInt                 | [CIKM 2019][AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/abs/1810.11921)                              |
|    Deep Interest Evolution Network     | [AAAI 2019][Deep Interest Evolution Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1809.03672.pdf)                                            |
|                FwFM                    | [WWW 2018][Field-weighted Factorization Machines for Click-Through Rate Prediction in Display Advertising](https://arxiv.org/pdf/1806.03514.pdf)                |
|                  ONN                  | [arxiv 2019][Operation-aware Neural Networks for User Response Prediction](https://arxiv.org/pdf/1904.12579.pdf)                                                |
|                 FGCNN                  | [WWW 2019][Feature Generation by Convolutional Neural Network for Click-Through Rate Prediction ](https://arxiv.org/pdf/1904.04447)                             |
|     Deep Session Interest Network      | [IJCAI 2019][Deep Session Interest Network for Click-Through Rate Prediction ](https://arxiv.org/abs/1905.06482)                                                |
|                FiBiNET                 | [RecSys 2019][FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09433.pdf)   |
|                FLEN                    | [arxiv 2019][FLEN: Leveraging Field for Scalable CTR Prediction](https://arxiv.org/pdf/1911.04690.pdf)   |
|                 BST                   | [DLP-KDD 2019][Behavior sequence transformer for e-commerce recommendation in Alibaba](https://arxiv.org/pdf/1905.06874.pdf)                           | 
|                IFM                 | [IJCAI 2019][An Input-aware Factorization Machine for Sparse Prediction](https://www.ijcai.org/Proceedings/2019/0203.pdf)   |
|                DCN V2                    | [arxiv 2020][DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems](https://arxiv.org/abs/2008.13535)   |
|                DIFM                 | [IJCAI 2020][A Dual Input-aware Factorization Machine for CTR Prediction](https://www.ijcai.org/Proceedings/2020/0434.pdf)   |
|   FEFM and DeepFEFM                    | [arxiv 2020][Field-Embedded Factorization Machines for Click-through rate prediction](https://arxiv.org/abs/2009.09931)                                         |
|              SharedBottom               | [arxiv 2017][An Overview of Multi-Task Learning in Deep Neural Networks](https://arxiv.org/pdf/1706.05098.pdf)  |
|   ESMM                    | [SIGIR 2018][Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate](https://arxiv.org/abs/1804.07931)                       |
|   MMOE                    | [KDD 2018][Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/abs/10.1145/3219819.3220007)                   |
|   PLE                    | [RecSys 2020][Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations](https://dl.acm.org/doi/10.1145/3383313.3412236)                   |
