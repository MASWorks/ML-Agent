# ML-Agent: Reinforcing LLM Agents for Autonomous Machine Learning Engineering
This repository is the official implementation of [ML-Agent: Reinforcing LLM Agents for Autonomous Machine Learning Engineering](https://arxiv.org/pdf/2505.23723).

## Machine Learning Tasks
In our paper, we organize 9 training tasks (held-in) for both exploration-enriched fine-tuning and step-wise reinforcement learning(RL) and 10 tasks (held-out) for further evaluation. All tasks all collected from [MLAgentBench](https://arxiv.org/abs/2310.03302) (MLA) and [MLEBench](https://arxiv.org/abs/2410.07095) (MLE). 

The tasks are listed as follows:
| Task Name                                           | Data Type | Task Type    | Metric   | Source |
|-----------------------------------------------------|-----------|--------------|----------|--------|
| **Training**                                         |           |              |          |        |
| cifar-10                                            | Image     | Classification | Acc. (%) ↑ | MLA    |
| aerial-cactus-identification                        | Image     | Classification | AUC ↑    | MLE    |
| dogs-vs-cats-redux-kernels-edition                  | Image     | Classification | Logloss ↓ | MLE    |
| plant-pathology-2020-fgvc7                          | Image     | Classification | AUC ↑    | MLE    |
| home-data-for-ml-course                             | Tabular   | Regression    | MAE ↓    | MLA    |
| spaceship-titanic                                   | Tabular   | Regression    | Acc. (%) ↑ | MLA    |
| nomad2018-predict-transparent-conductors            | Tabular   | Regression    | RMSLE ↓  | MLE    |
| feedback-prize-english-language-learning            | Text      | Classification | MCRMSE ↓ | MLA    |
| ogbn-arxiv                                          | Graph     | Classification | Acc. (%) ↑ | MLA    |
| **Testing**                                          |           |              |          |        |
| denoising-dirty-documents                           | Image     | Generation    | RMSE ↓    | MLE    |
| leaf-classification                                 | Image     | Classification | Logloss ↓ | MLE    |
| statoil-iceberg-classifier-challenge                | Image     | Classification | Logloss ↓ | MLE    |
| whale-categorization-playground                     | Image     | Classification | MAP@5 ↑   | MLE    |
| learning-agency-lab-automated-essay-scoring-2        | Text      | Regression    | QWK ↑     | MLE    |
| detecting-insults-in-social-commentary              | Text      | Classification | Acc. (%) ↑ | MLE    |
| spooky-author-identification                         | Text      | Classification | Logloss ↓ | MLE    |
| jigsaw-toxic-comment-classification-challenge       | Text      | Classification | AUC ↑     | MLE    |
| us-patent-phrase-to-phrase-matching                 | Tabular   | Regression    | PCC ↑     | MLE    |
| tabular-playground-series-dec-2021                  | Tabular   | Regression    | Acc. (%) ↑ | MLE    |




## Citation
If you find the project helpful, please cite:
```
@article{liu2025mlagentreinforcingllmagents,
      title={ML-Agent: Reinforcing LLM Agents for Autonomous Machine Learning Engineering}, 
      author={Zexi Liu and Jingyi Chai and Xinyu Zhu and Shuo Tang and Rui Ye and Bo Zhang and Lei Bai and Siheng Chen},
      year={2025},
      journal = {arXiv preprint arXiv: 2505.23723}
}
```
