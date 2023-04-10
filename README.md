# Hard-Negative-Sampling-via-Regularized-Optimal-Transport-for-Contrastive-Representation-Learning

This is the official code for the paper "Hard Negative Sampling via Regularized Optimal Transport for Contrastive Representation Learning". This repository contains the implementation of [model name] and related experiments described in the paper.

## Implenment on image dataset
For example, here is how we run the code for dataset "STL10" using entropy OT with regularization parameter $\epsilon = 0.3$
```
python main.py --dataset_name "STL10" --reg 0.3 --new_cost True --kappa 1
```

## Citation

If you find this repo useful for your research, please consider citing the paper:

```
@article{jiang2021hard,
  title={Hard Negative Sampling via Regularized Optimal Transport for Contrastive Representation Learning},
  author={Jiang, Ruijie and Ishwar, Prakash and Aeron, Shuchin},
  journal={arXiv preprint arXiv:2111.03169},
  year={2021}
}
```
For any questions, please contact Ruijie Jiang (Ruijie.Jiang@tufts.edu)

## Acknowledgements
This code is a modified version of the HCL implementation by [Josh/HCL](https://github.com/joshr17/HCL). The only difference from the their code is a minor alteration in the hard negative sampling approach, we change it from function "criterion" in their code to "OT_hard". To ensure a fair comparison, we have maintained all hyperparameters in Josh's implementation as they were in the original code.

Part of this code is inspired by [leftthomas/SimCLR](https://github.com/leftthomas/SimCLR), by [Josh/HCL](https://github.com/joshr17/HCL), and by [fanyun-sun/InfoGraph](https://github.com/fanyun-sun/InfoGraph).
