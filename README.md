# Pytorch graph2vec

This is a pytorch implementation of the `graph2vec: Learning Distributed Representations of Graphs` by Narayanan et. al (2017, https://arxiv.org/abs/1707.05005v1).

The repo has been developed from scratch in PyTorch. The official repository for the graph2vec in Tensorflow is available in https://github.com/MLDroid/graph2vec_tf. Therefore, if you make advantage of this repository in your research, please cite the following:

```
@article{
  narayanangraph2vec,
  title={graph2vec: Learning distributed representations of graphs},
  author={Narayanan, Annamalai and Chandramohan, Mahinthan and Venkatesan, Rajasekar and Chen, Lihui and Liu, Yang}
}
  ```

# 

# Performances

This implementation achieves results at par with results stated in original graph2vec work by Narayanan et. al (2017). The results are stated on MUTAG, PTC_MR, PROTEINS, NCI1, NCI109.

| Dataset | Accuracy |
| ------------- | ------------- |
| MUTAG | 76.02% |
| PTC_MR | 62.5% |
| MROTEINS | 71.42% |
| NCI1 | 83.02% |
| NCI109 | 81.68% |


# Instructions

To execute model training mention the dataset name in `dataset` variable in `main.py` and `model_data.py` file.

Execute the following command to convert the graph into subgraph units after WL relabeling:
```sh
$ python main.py
```

Post creation of subgraph, execute following command to initiate model training:
```sh
$ python model_data.py
```

# Requirements

This repository relies on Python 3.5 and PyTorch 1.3.0.

# Issues/Pull Requests/Feedbacks

Don't hesitate to contact for any feedback or create issues/pull requests.
