# Developer Guide

## 🌤🌤 Goals
Any kernel implementation is welcome. This repository is primarily for learning/practice, and achieving optimal performance is not the final goal. LeetCUDA focus on "using first, then using well". For optimal performance, it is recommended to directly use official implementations such as cuBLAS, cuDNN, FlashAttention, TensorRT, etc. If you are interested in implementing a specific kernel in this repository, feel free to send a new PR to LeetCUDA, for example:

## 👨‍💻👨‍💻 Pre-commit

Before submitting code, configure pre-commit, for example:

```bash
# fork xlite-dev/LeetCUDA to your own github page, then:
git clone git@github.com:your-github-page/your-fork-LeetCUDA.git
cd your-fork-LeetCUDA && git checkout -b test
# update submodule
git submodule update --init --recursive --force
# install pre-commit
pip3 install pre-commit
pre-commit install
pre-commit run --all-files
```

## 👨‍💻👨‍💻 Add a new kernel
Please check [./kernels/elementwise/](./kernels/elementwise/) directory as an example.

```bash
# add new xxx_kernel
# add your commits
git add .
git commit -m "feat: add xxx kernel"
git push
# then, open a PR from your personal branch to LeetCUDA:main
```
