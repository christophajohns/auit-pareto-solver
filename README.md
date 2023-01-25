# Pareto Optimal UI Placements for Adaptive Mixed Reality

> A Python-based Pareto solver for the Adaptive User Interfaces Toolkit (AUIT).

## Abstract

Adaptive mixed reality applications adjust their user interfaces based on the context in which they are used to provide a smooth experience for different users and environments.
This involves carefully positioning UI elements, which can be challenging due to the many possible placements and need to balance competing goals.
Current approaches employ global criterion optimization methods like weighted sums, which can be difficult to use, inflexible, and might not find preferred solutions.
This can prevent the adaptations from meeting end-user expectations.
We suggest using _a posteriori_ multi-objective optimization methods which generate a set of Pareto optimal potential adaptations made available in the UI, promising more control for users and more flexible computational decision-making.
We explore the feasibility of our approach by generating adaptations for a basic synthetic example and discuss relevant dimensions for a formal evaluation with end-users, including the choice of algorithm, decomposition technique, and objectives.

## Python Pareto Solver

This is a Python-based solver for the Adaptive User Interface Toolkit (AUIT) 
returning Pareto optimal adapations.

### Development

The project is developed using Python 3.9.
You can create the necessary environment using conda with the following 
command:

```zsh
conda env create -f environment.yml
```
