# Experimental Setup

This document describes the experimental setup used to evaluate the Pareto solver. In particular, it describes how to run the evaluations and how to reproduce the results presented in the paper.

## Research Questions

We aim to answer the following primary research questions:
- RQ1 - Non-Convexity: Can our solver produce higher maximal utility than the weighted sum baseline for non-convex Pareto frontier shapes?
- RQ2 - Non-Linearity: Can our solver produce higher maximal utility than the weighted sum baseline for utility functions that are non-linear combinations of the optimization objectives?

We are further interested in the following secondary research questions:
- RQ3 - Weights: Can our solver produce higher maximal utility than the weighted sum baseline without the use of weights defined by the user, designer, or developer?
- RQ4 - Runtime: How does the runtime of our solver compare to the runtime of the weighted sum baseline for the above scenarios?

## Dependent Variables

We are interested in the following two dependent variables:
- Maximal utility value (U)
- Runtime (T)

## Independent Variables

We investigate the effect of the following independent variables:
- Pareto frontier shape (PFS): convex, non-convex (concave)
- Utility function (UF): linear, non-linear combination of objectives
- Optimization method (OM): weighted sum, Pareto solver
- Maximum number of proposals (MNP): 1, 10

The first two independent variables are used to construct two scenarios of interest:
1. Convex Pareto frontier shape and linear utility function (CONV+LIN)
2. Non-convex Pareto frontier shape and non-linear utility function (NCONV+NLIN)

## Fixed Variables

The following independent variables are fixed:
- Other objectives: Neck ergonomics (NE), Reachability (RE)
- Solver: NSGA-III
    - Population size: 100
    - Number of generations: 100
    - Initialization: Riesz-energy reference directions
- Decomposition technique: AASF
    - Decomposition weights for 1 proposal: equal weights for all objectives
    - Decomposition weights for 10 proposals: well-spaced weights for all objectives using Riesz-energy reference directions
- Number of runs per optimization with different seeds: 10
- Weight distribution: uniform ranging from 0 to 1
- Sample size for utility functions: 100

## Additional Baseline: Random Solver (RND)

Each evaluation further includes a random solver (RND) that generates 1,000 random solutions. It is used to estimate a baseline maximal utility to compare against the Pareto solver and weighted sum.

## Objectives

The objectives are based on the RULA workplace ergonomics metrics. The objectives are:
- Neck ergonomics (NE)
- Shoulder ergonomics (SE)
- Reachability (RE)

### Utility Function

Each utility is defined as a linear combination of these objectives and as a function of an adaptation. Since each objective is formulated as a cost function, the utility is defined as a linear combination of the inverse of the objectives. The utility function is defined as:

$U(\mathbf{x}) = \sum_{i=1}^n w_i (1 - f_i(\mathbf{x})) = \sum_{i=1}^n w_i - \sum_{i=1}^n w_i f_i(\mathbf{x})$

where $u$ is the utility, $w_i$ is the weight of the $i$th objective, $f_i$ is the $i$th objective and $\mathbf{x}$ is the adaptation. The weights are set to $1/n$ for all $n$ objectives in the scenarios and to 0 for all objectives not in the scenario:

$U(\mathbf{x}) = 1 - \frac{\sum_{i=1}^n f_i(\mathbf{x})}{n}$

For example:

$U(\mathbf{x}) = 1 - \frac{F_{NE}(\mathbf{x}) + F_{SE}(\mathbf{x}) + F_{RE}(\mathbf{x})}{4}$

The utility function is used as a placeholder for a true preference function that we do not know. It has a maximum value of 1 and a minimum value of 0.

For our evaluations, we create a population of utility functions by sampling the weights from a uniform distribution.

### Objective Functions

The objective functions are dependent on the individual solver. See the paper for more details.

#### Neck Ergonomics (NE)

The neck ergonomics objective (NE) is based on the RULA neck ergonomics metric. The objective is to minimize the neck load. The objective is defined as the angle between a vector from the eyes straight forward and a vector from the eyes to the UI element (i.e., adaptation). The objective has a minimum value of 0 if the UI element is at eye level of the user and a maximum value of 1 if the UI element is directly above or below the user's eyes. If the element is at the user's eye position, the objective has a value of 1.

The goal is to minimize the angle between the vectors and thereby to minimize the neck load.

#### Shoulder Ergonomics (SE)

The shoulder ergonomics objective (SE) is based on the RULA shoulder ergonomics metric. The objective is to minimize the shoulder load. The objective is defined as the angle between a vector from the shoulder straight down and a vector from the shoulder to the UI element (i.e., adaptation). The objective has a minimum value of 0 if the UI element is directly below the shoulder of the user and a maximum value of 1 if the UI element is directly above the user's shoulders. If the element is at the user's shoulder position, the objective has a value of 1.

The goal is to minimize the angle between the vectors and thereby to minimize the shoulder load.

##### Convex Pareto Frontier Shape

To generate a convex Pareto frontier shape, we use the formulation described where the cost grows linearly with the angle:

$SE_{conv}(\mathbf{x}) = \frac{1}{2} \left(1 - \frac{\theta}{\pi}\right)$

where $\theta$ is the angle between the vectors.

##### Non-Convex (Concave) Pareto Frontier Shape

To generate a non-convex (concave) Pareto frontier shape, we use the formulation described where the cost grows exponentially with the angle:

$SE_{\overline{conv}}(\mathbf{x}) = \frac{1}{2} \left(1 - \exp\left(-\frac{a}{\theta}\right)\right)$

where $\theta$ is the angle between the vectors and $a$ is a constant that controls the steepness of the curve. We set $a = 10$.

#### Reachability (RE)

The reachability objective (RE) is based on a simple distance metric. The objective is to maximize the reachability. The objective is defined as the distance between the user's shoulder joint and the UI element (i.e., adaptation). The objective has a minimum value of 0 if the UI element is within reach of the user as defined by a reachability threshold (i.e., arm length) and a maximum value striving toward 1 if the UI element is outside of the reachability threshold. If the element is at the user's shoulder position, the objective has a value of 1.

The goal is to minimize the distance between the user's shoulder joint and the UI element to maximize the reachability.

## Scenarios

The scenarios are used to control for the relationship between the assumed true preference function that our objectives are trying to approximate and that our adaptations are trying to satisfy. Each scenario is designed to test a different level of approximation between the true preference function and the objectives. All scenarios are based on workplace ergonomics metrics inspired by RULA (i.e., neck ergonomics, shoulder ergonomics and torso ergonomics).

The goal is to minimize the angle between the vectors and thereby to minimize the torso load.

### Scenario 1 (CONV+LIN): Convex Pareto Frontier Shape and Linear Utility Function

In this scenario, the preference criteria are a superset of the objectives. This means that the preference criteria are more specific than the objectives and that the objectives are a poor approximation of the preference criteria. A good adaptation should be able to satisfy the preference criteria based solely on the limited information provided by the objectives. Our experiments use the following sets.

**Preference Criteria**
- Minimize the neck load (NE)
- Minimize the shoulder load producing a convex Pareto frontier (SE_conv)
- Minimize the reachability cost (RE)

We implement this superset scenario by using the following utility function:

$U(\mathbf{x}) = 1 - (w_{NE} F_{NE}(\mathbf{x}) + w_{SE} F_{SE_{conv}}(\mathbf{x}) + w_{RE} F_{RE}(\mathbf{x}))$

A sampled utility function may take the following form:

$U_{example}(\mathbf{x}) = 1 - (0.33 F_{NE}(\mathbf{x}) + 0.33 F_{SE_{conv}}(\mathbf{x}) + 0.33 F_{RE}(\mathbf{x}))$

**Objectives**
- Minimize the neck load (NE)
- Minimize the shoulder load producing a convex Pareto frontier (SE_conv)
- Minimize the reachability cost (RE)


This scenario represents a optimal case for weighted sum optimization where the objectives are accurate representations of the true preference criteria, all preference criteria have been included and the preferences can perfectly be represented by a linear combination of the objectives.


### Scenario 2 (NCONV+NLIN): Non-Convex Pareto Frontier Shape and Non-Linear Utility Function

In this scenario, the preference criteria are a superset of the objectives. This means that the preference criteria are more specific than the objectives and that the objectives are a poor approximation of the preference criteria. A good adaptation should be able to satisfy the preference criteria based solely on the limited information provided by the objectives. Our experiments use the following sets.

**Preference Criteria**
- Minimize the neck load (NE)
- Minimize the shoulder load producing a non-convex Pareto frontier (SE_nconv)
- Minimize the reachability cost (RE)

We implement this superset scenario by using the following utility function:

$U(\mathbf{x}) = 1 - (w_{NE} F_{NE}(\mathbf{x}) + w_{SE} F_{SE_{nconv}}(\mathbf{x}) + w_{RE} F_{RE}(\mathbf{x}))$

A sampled utility function may take the following form:

$U_{example}(\mathbf{x}) = 1 - (0.33 F_{NE}(\mathbf{x}) + 0.33 F_{SE_{nconv}}(\mathbf{x}) + 0.33 F_{RE}(\mathbf{x}))$

**Objectives**
- Minimize the neck load (NE)
- Minimize the shoulder load producing a non-convex Pareto frontier (SE_nconv)

This scenario represents a realistic failure case for weighted sum optimization where the objectives are still accurate representations of the true preference criteria but not all preference criteria have been included, the preferences cannot perfectly be represented by a linear combination of the objectives, and the objectives form a concave Pareto frontier.


## Procedure

Before the experiments are conducted, a sample of the weights in the true preference functions are generated. These weights are reused across the experiments.

First, we generate 10 adaptation proposals for each scenario, optimization method and seed, measuring the time until completion (i.e., until the solutions have been computed) for each run. Since these proposals include the equally weighted adaptation solution for the Pareto solver, it can be reused to evaluate a fully automatic adaptation. The weighted sum optimization is run 10 times in sequence for each seed. To evaluate a fully automatic adaptation, only the first result of the runs for the first seed is used.

Second, we compute the utilities for these proposals across the sample of utility functions and determine the maximum utility for each utility function. These distributions of maximum utilities are then used to compare the adaptation techniques.

## Analysis

The results are reported in the following format:

| Scenario | Optimization Method | Mean $U_{max}$ (SD) 1 Proposal | Mean $U_{max}$ (SD) 10 Proposals | Mean $t$ (SD) 1 Proposal | Mean $t$ (SD) 10 Proposals |
|-|-|-|-|-|-|
| CONV+LIN | Pareto | 0.99 (0.01) | 0.99 (0.01) | 0.01 (0.00) | 0.01 (0.00) |
| CONV+LIN | Weighted Sum | 0.99 (0.01) | 0.99 (0.01) | 0.01 (0.00) | 0.01 (0.00) |
| NCONV+NLIN | Pareto | 0.99 (0.01) | 0.99 (0.01) | 0.01 (0.00) | 0.01 (0.00) |
| NCONV+NLIN | Weighted Sum | 0.99 (0.01) | 0.99 (0.01) | 0.01 (0.00) | 0.01 (0.00) |

The random solver results are reported in the following format:

| Scenario | Mean $U$ (SD) |
|-|-|
| CONV+LIN | 0.99 (0.01) |
| NCONV+NLIN | 0.99 (0.01) |

We further visualize the results in box plots comparing the distributions of maximum utilities and runtimes of the opimization methods for each scenario.

The research questions are answered as follows:

- **RQ1** and **RQ2**: Does the Pareto solver outperform the weighted sum optimization for the NCONV+NLIN scenario in terms of maximum utility?
- **RQ3**: Does the single adaptation proposed by the Pareto solver (i.e., the equal weight compromise solution computed via AASF) outperform the single weighted sum optimization result for both scenarios in terms of maximum utility?
- **RQ4**: Does the Pareto solver outperform the weighted sum optimization for both scenarios in terms of runtime?