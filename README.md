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

### Networking Protocol

The solver uses the [ZeroMQ](https://pyzmq.readthedocs.io/en/latest/#) library to communicate with the AUIT server.
The solver acts both as a client and a server to AUIT.
It implements the following communication protocol:

Each message is a string encoded in UTF-8.
The first character of the message indicates the message type.
The following characters are the message payload encoded in JSON.

#### Message Types

- `O`: Optimization request
- `o`: Optimization response
- `E`: Evaluation request
- `e`: Evaluation response

#### General Object Types

##### Element

An element is a UI element, including its ID, position and rotation.

```json
{
    "id": <string>,
    "position": {
        "x": <number>,
        "y": <number>,
        "z": <number>
    },
    "rotation": {
        "x": <number>,
        "y": <number>,
        "z": <number>,
        "w": <number>
    }
}
```

Example element:

```json
{
  "id": "button:1",
  "position": {
    "x": 0.0,
    "y": 0.0,
    "z": 0.0
  },
  "rotation": {
    "x": 0.0,
    "y": 0.0,
    "z": 0.0,
    "w": 1.0
  }
}
```

##### Layout

A layout is an object containing a list of UI elements.

```json
{
    "elements": [<Element>, <Element>, ...]
}
```

Example layout:

```json
{
  "elements": [
    {
      "id": "button:1",
      "position": {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0
      },
      "rotation": {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "w": 1.0
      }
    },
    {
      "id": "button:2",
      "position": {
        "x": 1.0,
        "y": 0.0,
        "z": 0.0
      },
      "rotation": {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "w": 1.0
      }
    }
  ]
}
```

#### Optimization Request

An optimization request (message type: `O`) is sent from the AUIT client to the solver to request a set of Pareto optimal solutions for a given set of objectives and constraints.

```json
{
    "nObjectives": <int>,
    "nConstraints": <int>,
    "initialLayout": <Layout>
}
```

Example optimization request:

```json
{
  "nObjectives": 3,
  "nConstraints": 2,
  "initialLayout": {
    "elements": [
      {
        "id": "button:1",
        "position": {
          "x": 0.0,
          "y": 0.0,
          "z": 0.0
        },
        "rotation": {
          "x": 0.0,
          "y": 0.0,
          "z": 0.0,
          "w": 1.0
        }
      },
      {
        "id": "button:2",
        "position": {
          "x": 1.0,
          "y": 0.0,
          "z": 0.0
        },
        "rotation": {
          "x": 0.0,
          "y": 0.0,
          "z": 0.0,
          "w": 1.0
        }
      }
    ]
  }
}
```

#### Optimization Response

An optimization response (message type: `o`) is sent by the solver to the AUIT server with a set of Pareto optimal solutions for a given set of objectives and constraints.

```json
{
    "solutions": [<Layout>, <Layout>, ...]
}
```

Example optimization response:

```json
{
  "solutions": [
    {
        "elements":   {
            "id": "button:1",
            "position": {
            "x": 1.0,
            "y": 0.0,
            "z": 0.0
            },
            "rotation": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "w": 1.0
            }
        },
        {
            "id": "button:2",
            "position": {
            "x": 2.0,
            "y": 0.0,
            "z": 0.0
            },
            "rotation": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "w": 1.0
            }
        }
    },
    {
        "elements":   {
            "id": "button:1",
            "position": {
            "x": 2.0,
            "y": 0.0,
            "z": 0.0
            },
            "rotation": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "w": 1.0
            }
        },
        {
            "id": "button:2",
            "position": {
            "x": 3.0,
            "y": 0.0,
            "z": 0.0
            },
            "rotation": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "w": 1.0
            }
        }
    }
  ]
}
```

#### Evaluation Request

An evaluation request (message type: `E`) is sent from the Python solver client to the AUIT server to request the evaluation of a given set of candidate layouts.

```json
{
    "layouts": [<Layout>, <Layout>, ...]
}
```

Example evaluation request:

```json
{
  "layouts": [
    {
        "elements":   {
            "id": "button:1",
            "position": {
            "x": 1.0,
            "y": 0.0,
            "z": 0.0
            },
            "rotation": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "w": 1.0
            }
        },
        {
            "id": "button:2",
            "position": {
            "x": 2.0,
            "y": 0.0,
            "z": 0.0
            },
            "rotation": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "w": 1.0
            }
        }
    },
    {
        "elements":   {
            "id": "button:1",
            "position": {
            "x": 2.0,
            "y": 0.0,
            "z": 0.0
            },
            "rotation": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "w": 1.0
            }
        },
        {
            "id": "button:2",
            "position": {
            "x": 3.0,
            "y": 0.0,
            "z": 0.0
            },
            "rotation": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "w": 1.0
            }
        }
    }
  ]
}
```

#### Evaluation Response

An evaluation response (message type: `e`) is sent by the AUIT server to the Python solver client with a set of cost vectors and a set of constraint violation vectors for a given set of candidate layouts.
The costs and violations are ordered in the same order as the layouts in the evaluation request.
Each cost vector is a list of costs for each objective function given the layout.
Each violation vector is a list of constraint violations for each constraint given the layout.
The length of the cost vector is equal to the number of objectives specified in the optimization request.
The length of the violation vector is equal to the number of constraints specified in the optimization request.

```json
{
    "costs": [[<float>, <float>, ...], [<float>, <float>, ...], ...],
    "violations": [[<float>, <float>, ...], [<float>, <float>, ...], ...]
}
```

Example evaluation response:

```json
{
  "costs": [
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0]
  ],
  "violations": [
    [0.0, 0.0],
    [0.0, 0.0]
  ]
}
```

## References

[1] H. Seada and K. Deb. A unified evolutionary optimization procedure for single, multiple, and many objectives. IEEE Transactions on Evolutionary Computation, 20(3):358–369, June 2016. [doi:10.1109/TEVC.2015.2459718](https://doi.org/10.1109/TEVC.2015.2459718).

[2] L. Rachmawati and D. Srinivasan. Multiobjective evolutionary algorithm with controllable focus on the knees of the pareto front. IEEE Transactions on Evolutionary Computation, 13(4):810–824, Aug 2009. [doi:10.1109/TEVC.2009.2017515](https://doi.org/10.1109/TEVC.2009.2017515).
