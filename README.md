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

This is a Python-based solver for the [Adaptive User Interface Toolkit (AUIT)](https://github.com/joaobelo92/auit)
returning Pareto optimal adapations.

### Usage

The solver can be used as a command line tool to start a server providing the Pareto optimal solutions for a given set of objectives and constraints.
It makes requests to the AUIT server via a [ZeroMQ](https://pyzmq.readthedocs.io/en/latest/#) client to generate the solutions and returns the Pareto optimal solutions as a response.
The server is exposed on port 5555.
The client connects to the AUIT server on port 5556.
Both the Python solver and the AUIT server need to be running in order to use the solver.

By default, the solver uses the U-NSGA-III [1] algorithm and High Trade-Off Points [2] decomposition technique.

```zsh
$ python solver.py
Listening on port 5555...
```

### Development

The project is developed using Python 3.9.
You can create the necessary environment using conda with the following
command:

```zsh
$ conda env create -f environment.yml
```

To export the environment to a new `environment.yml` file, run:

```zsh
$ conda env export | grep -v "^prefix: " > environment.yml
```

For testing, we have included a Python script that mocks the AUIT server and client and calls the solver.
The script can be run with the following command:

```zsh
$ python AUIT.py
```

The functionality of the AUIT mock can be tested by running the following command:

```zsh
$ python tests/test_AUIT.py
```

### Networking Protocol

Details on the networking between AUIT and the Python solver can be found in the [documentation](docs/protocol.md).

## References

[1] H. Seada and K. Deb. A unified evolutionary optimization procedure for single, multiple, and many objectives. IEEE Transactions on Evolutionary Computation, 20(3):358–369, June 2016. [doi:10.1109/TEVC.2015.2459718](https://doi.org/10.1109/TEVC.2015.2459718).

[2] L. Rachmawati and D. Srinivasan. Multiobjective evolutionary algorithm with controllable focus on the knees of the pareto front. IEEE Transactions on Evolutionary Computation, 13(4):810–824, Aug 2009. [doi:10.1109/TEVC.2009.2017515](https://doi.org/10.1109/TEVC.2009.2017515).
