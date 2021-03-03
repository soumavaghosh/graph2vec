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

# Performances

For the branch **master**, the training of the transductive learning on Cora task on a Titan Xp takes ~0.9 sec per epoch and 10-15 minutes for the whole training (~800 epochs). The final accuracy is between 84.2 and 85.3 (obtained on 5 different runs). For the branch **similar_impl_tensorflow**, the training takes less than 1 minute and reach ~83.0.

A small note about initial sparse matrix operations of https://github.com/tkipf/pygcn: they have been removed. Therefore, the current model take ~7GB on GRAM.


# Requirements

This repository relies on Python 3.5 and PyTorch 1.3.0.

# Issues/Pull Requests/Feedbacks

Don't hesitate to contact for any feedback or create issues/pull requests.
