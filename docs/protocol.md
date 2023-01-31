# Networking Protocol

The solver uses the [ZeroMQ](https://pyzmq.readthedocs.io/en/latest/#) library to communicate with the AUIT server.
The solver acts both as a client and a server to AUIT.
It implements the following communication protocol:

Each message is a string encoded in UTF-8.
The first character of the message indicates the message type.
The following characters are the message payload encoded in JSON.

## Message Types

- `O`: Optimization request
- `o`: Optimization response
- `E`: Evaluation request
- `e`: Evaluation response

## General Object Types

### Element

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

### Layout

A layout is an object containing a list of UI elements.

```json
{
    "items": [<Element>, <Element>, ...]
}
```

Example layout:

```json
{
  "items": [
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

## Optimization Request

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
    "items": [
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

## Optimization Response

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
      "items": [
        {
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
      ]
    },
    {
      "items": [
        {
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
      ]
    }
  ]
}
```

## Evaluation Request

An evaluation request (message type: `E`) is sent from the Python solver client to the AUIT server to request the evaluation of a given set of candidate layouts.

```json
{
    "items": [<Layout>, <Layout>, ...]
}
```

Example evaluation request:

```json
{
  "items": [
    {
      "items": [
        {
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
      ]
    },
    {
      "items": [
        {
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
      ]
    }
  ]
}
```

## Evaluation Response

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
