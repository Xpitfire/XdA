# Overcoming Catastrophic Forgetting with Context-Dependent Activations (XdA) and Synaptic Stabilization

## Abstract

Overcoming Catastrophic Forgetting in neural networks is crucial to solving continuous learning problems.
Deep Reinforcement Learning uses neural networks to make predictions of actions according to the current state space of an environment.
In a dynamic environment, robust and adaptive life-long learning algorithms mark the cornerstone of their success.
In this thesis we will examine an elaborate subset of algorithms countering catastrophic forgetting in neural networks and reflect on their weaknesses and strengths.
Furthermore, we present an enhanced alternative to promising synaptic stabilization methods, such as Elastic Weight Consolidation or Synaptic Intelligence.
Our method uses context-based information to switch between different pathways throughout the neural network, reducing destructive activation interference during the forward pass and destructive weight updates during the backward pass.
We call this method Context-Dependent Activations (XdA).
We show that XdA enhanced methods outperform basic synaptic stabilization methods and are a better choice for long task sequences.

## Authors

Marius-Constantin Dinu, Günter Klambauer, Sepp Hochreiter

## Reference and Link to thesis

```
@ARTICLE{Dinu,
  author = {Dinu, M.C. and Klambauer, G. and Hochreiter, S.},
  title = {Overcoming Catastrophic Forgetting with Context-Dependent Activations (XdA) and Synaptic Stabilization},
  howpublished = {\url{https://www.dinu.at/profile/home/overcoming-catastrophic-forgetting-with-context-dependent-activations-and-synaptic-stabilization/}},
  year = 2019
}
```

[Link](https://www.dinu.at/wp-content/uploads/2019/11/Overcoming-Catastrophic-Forgetting-with-Context-Dependent-Activations-and-Synaptic-Stabilization.pdf)

## Installing

1. Create a python 3 conda environment (check the requirements.txt file)

2. The following folder structure is expected at runtime. From the git folder:
    * dat/ : Place to put/download all data sets
    * res/ : Place to save results
    * tmp/ : Place to store temporary files

3. The main experiment file is the project.ipynb.

## Notes

* If using this code, parts of it, or developments from it, please cite the above reference.
* We do not provide any support or assistance for the supplied code nor we offer any other compilation/variant of it.
* We assume no responsibility regarding the provided code.

## Related work

This repository is based on the work from Serrà et al. (2018) "Overcoming catastrophic forgetting with hard attention to the task".

[GitHub](https://github.com/joansj/hat)

[Paper](https://arxiv.org/abs/1801.01423)
