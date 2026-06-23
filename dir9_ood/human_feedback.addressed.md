cupbear: https://github.com/ejnnr/cupbearer has lots of tools for OOD
Please use its methods as baseline.
Do not use PyPi installation. Use github repo.
Remember to check if the repo can run in the current environment
DO NOT uninstall the current cuda, pytorch, numpy and other important package's version because they are shared with other people.
If you have to use a different environment, make your own conda or virtual env.
The GPU should be compatible with CUDA >= 13.0. Use nvidia-smi to check.