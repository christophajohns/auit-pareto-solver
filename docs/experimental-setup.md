# Experimental Setup

This document describes the experimental setup used to evaluate the Pareto solver. In particular, it describes how to run the evaluations and how to reproduce the results presented in the paper.

We evaluate the Pareto solver against the following baselines:
- Static weighted sum solver (WS)
- Multiple single-objective solvers (MSO)
- Random solver (RND)

Each of the baselines is described in more detail below and in the paper.

We compare the Pareto solver against the baselines using a set of four different scenarios. Each scenario features a computational user model and a set of objectives. Their relationship is described by a utility function. The preference criteria in the utility function and their combinations determine the scenarios. Each scenario features two or three objectives/preference criteria based on RULA workplace ergonomics. The scenarios are:
- Scenario 1: Preference criteria are superset of objectives (SUP)
- Scenario 2: Preference criteria are subset of objectives (SUB)
- Scenario 3: Preference criteria are disjoint from objectives (DIS)
- Scenario 4: Preference criteria are equal to objectives (EQU)

The scenarios are described in more detail below and in the paper.

For each of the scenarios, we evaluate the Pareto solver with the U-NSGA-III algorithm with a population size of 100 and 100 generations. The algorithm is initialized with Riesz-energy reference directions. The decomposition technique used is AASF. The decomposition weights are set to retrieve the single-objective optima and one compromise solution using equal weights for all objectives. The Pareto solver is evaluated against the baselines on the basis of the maximal utility value.

## Research Questions

With our experiments, we aim to answer two separate research questions:
- RQ1: Do multiple suggestions achieve higher maximal utility value than single adaptations?
- RQ2: Do Pareto optimal multiple suggestions achieve higher maximal utility value than non-Pareto optimal multiple suggestions and than single-objective optimal solutions?

## Baselines

All baselines are implemented in Python using the package pymoo and use the same genetic algorithm U-NSGA-III to find solutions.

### Static Weighted Sum Solver (WS)

The static weighted sum solver (WS) is a baseline that uses a static weighted sum utility/cost function to generate suggestions. For our experiment, we weight the objectives equally. The WS solver is primarily used to answer RQ1. In our implementation, we first generate the Pareto optimal solutions using the Pareto solver and then use weighted sum decomposition to find the optimal solution. The weighted sum solver always weighs all objectives equally.

### Multiple Single-Objective Solvers (MSO)

The multiple single-objective solvers (MSO) is a baseline that uses multiple single-objective solvers to generate suggestions. The MSO solver is primarily used to answer RQ2. For RQ1, it is used to generate suggestions for each objective separately. In our implementation, we first generate the Pareto optimal solutions using the Pareto solver and then use weighted sum decomposition to find the single-objective optimal solutions.

### Random Solver (RND)

The random solver (RND) is a baseline that generates one ore multiple random solutions. The RND solver is primarily used to answer RQ2. For RQ1, it is used to generate a single random suggestion. For RQ2, it is used to generate the same number of suggestions as the Pareto solver.

## Objectives

The objectives are based on the RULA workplace ergonomics metrics. The objectives are:
- Neck ergonomics (NE)
- Shoulder ergonomics (SE)
- Torso ergonomics (TE)
- Reachability (RE)

### Utility Function

Each utility is defined as a linear combination of these objectives and as a function of an adaptation. Since each objective is formulated as a cost function, the utility is defined as a linear combination of the inverse of the objectives. The utility function is defined as:

$u(\mathbf{x}) = \sum_{i=1}^n w_i (1 - f_i(\mathbf{x})) = \sum_{i=1}^n w_i - \sum_{i=1}^n w_i f_i(\mathbf{x})$

where $u$ is the utility, $w_i$ is the weight of the $i$th objective, $f_i$ is the $i$th objective and $\mathbf{x}$ is the adaptation. The weights are set to $1/n$ for all $n$ objectives in the scenarios and to 0 for all objectives not in the scenario:

$u(\mathbf{x}) = 1 - \frac{\sum_{i=1}^n f_i(\mathbf{x})}{n}$

For example:

$u(\mathbf{x}) = 1 - \frac{f_{NE}(\mathbf{x}) + f_{SE}(\mathbf{x}) + f_{TE}(\mathbf{x}) + f_{RE}(\mathbf{x})}{4}$

The utility function is used as a placeholder for a true preference function that we do not know. It has a maximum value of 1 and a minimum value of 0.

### Objective Functions

The objective functions are dependent on the individual solver. See the paper for more details.

### Neck Ergonomics (NE)

The neck ergonomics objective (NE) is based on the RULA neck ergonomics metric. The objective is to minimize the neck load. The objective is defined as the angle between a vector from the eyes straight forward and a vector from the eyes to the UI element (i.e., adaptation). The objective has a minimum value of 0 if the UI element is at eye level of the user and a maximum value of 1 if the UI element is directly above or below the user's eyes. If the element is at the user's eye position, the objective has a value of 1.

The goal is to minimize the angle between the vectors and thereby to minimize the neck load.

### Shoulder Ergonomics (SE)

The shoulder ergonomics objective (SE) is based on the RULA shoulder ergonomics metric. The objective is to minimize the shoulder load. The objective is defined as the angle between a vector from the shoulder straight down and a vector from the shoulder to the UI element (i.e., adaptation). The objective has a minimum value of 0 if the UI element is directly below the shoulder of the user and a maximum value of 1 if the UI element is directly above the user's shoulders. If the element is at the user's shoulder position, the objective has a value of 1.

The goal is to minimize the angle between the vectors and thereby to minimize the shoulder load.

### Torso Ergonomics (TE)

The torso ergonomics objective (TE) is based on the RULA torso ergonomics metric. The objective is to minimize the torso load. The objective is defined as the angle between a vector from the waist straight forward and a vector from the waist to the UI element (i.e., adaptation) if the UI element is below waist level and as zero if the UI element is above waist level. The objective has a minimum value of 0 if the UI element is at waist level of the user or above and a maximum value of 1 if the UI element is directly below the user's waist. If the element is at the user's waist position, the objective has a value of 1.

The goal is to minimize the angle between the vectors and thereby to minimize the torso load.

### Reachability (RE)

The reachability objective (RE) is based on a simple distance metric. The objective is to maximize the reachability. The objective is defined as the distance between the user's shoulder joint and the UI element (i.e., adaptation). The objective has a minimum value of 0 if the UI element is within reach of the user as defined by a reachability threshold (i.e., arm length) and a maximum value striving toward 1 if the UI element is outside of the reachability threshold. If the element is at the user's shoulder position, the objective has a value of 1.

The goal is to minimize the distance between the user's shoulder joint and the UI element to maximize the reachability.

## Scenarios

The scenarios are used to control for the relationship between the assumed true preference function that our objectives are trying to approximate and that our adaptations are trying to satisfy. Each scenario is designed to test a different level of approximation between the true preference function and the objectives. All scenarios are based on workplace ergonomics metrics inspired by RULA (i.e., neck ergonomics, shoulder ergonomics and torso ergonomics).

The goal is to minimize the angle between the vectors and thereby to minimize the torso load.

### Scenario 1: Preference Criteria are Superset of Objectives (SUP)

In this scenario, the preference criteria are a superset of the objectives. This means that the preference criteria are more specific than the objectives and that the objectives are a poor approximation of the preference criteria. A good adaptation should be able to satisfy the preference criteria based solely on the limited information provided by the objectives. Our experiments use the following sets.

**Preference Criteria**
- Minimize the neck load (NE)
- Minimize the shoulder load (SE)
- Minimize the torso load (TE)

We implement this superset scenario by using the following utility function:

$u(\mathbf{x}) = 1 - (0.33 f_{NE}(\mathbf{x}) + 0.33 f_{SE}(a) + 0.33 f_{TE}(\mathbf{x}))$

**Objectives**
- Minimize the neck load (NE)
- Minimize the shoulder load (SE)


This scenario represents a generous case where the objectives are accurate representations of the true preference criteria but not all preference criteria have been included (e.g., because the user did not know about them, decided to ignore them or because they are costly to evaluate).


### Scenario 2: Preference Criteria are Subset of Objectives (SUB)

If the preference criteria are a subset of the objectives, it means that the preference criteria are more general than the objectives. A good adaptation should be able to satisfy the preference criteria irrespective of the noise introduced by the overly specific objectives. Our experiments use the following sets.

**Preference Criteria**
- Minimize the neck load (NE)
- Minimize the shoulder load (SE)

**Objectives**
- Minimize the neck load (NE)
- Minimize the shoulder load (SE)
- Minimize the torso load (TE)

This scenario represents a limited optimistic case where the objectives are accurate representations of the true preference criteria but too many objectives have been specified and included and not all are relevant (e.g., because the user is easier to satisfy than the objectives suggest or because the objectives are too specific).

### Scenario 3: Preference Criteria are Disjoint from Objectives (DIS)

If the preference criteria are disjoint from the objectives, it means that the preference criteria are not related to the objectives. A good adaptation should be able to satisfy the preference criteria even if the choice of objectives is random or otherwise unrelated to the preference criteria. Our experiments use the following sets.

**Preference Criteria**
- Minimize the neck load (NE)
- Minimize the shoulder load (SE)

**Objectives**
- Minimize the torso load (TE)
- Minimize the reachability cost (RE)

### Scenario 4: Preference Criteria are Equal to Objectives (EQU)

If the preference criteria are equal to the objectives, it means that the preference criteria are the same as the objectives. A good adaptation should be able to satisfy the preference criteria if the objectives are accurate representations of the preference criteria and combined in the same way as in the true preference function. Our experiments use the following sets.

**Preference Criteria**
- Minimize the neck load (NE)
- Minimize the shoulder load (SE)

**Objectives**
- Minimize the neck load (NE)
- Minimize the shoulder load (SE)

This scenario represents the most optimistic case where the objectives are perfect representations of the true preference criteria and all preference criteria have been included. In case of the static weighted sum solver, the objectives are even combined in the same way as in the true preference function.