# purl
**P**athfinding **U**sing **R**einforcement **L**earning

## Algorithms
### MDP (using the `FullyObsWrapper`)
* **Q-table**
* **Q-network**
### POMDP
* **PPO**
* **DQN** (with the Double DQN extension) - work in progress, not currently working as intended

## Getting Started
### Prerequisites

`python` + `pip`, version 3.6 or greater

### Installing
**Note**: It's recommended that you install the Python dependencies in a virtual environment.  `virtualenv` and `virtualenvwrapper`:

```
pip install virtualenv
pip install virtualenvwrapper
```

**Note**: To get `matplotlib` to work on a Mac with a virtual environment you have to use `venv` instead


#### Setting up a virtual environment

```mkvirtualenv purl```

or with `venv`:

```python -m venv purl-venv```

#### Installing Python dependencies

First, switch to the virtual environment:

```workon purl```

or with `venv`:

```source purl-venv/bin/activate```

**Note**: You can set up an alias in your shell to make the virtual environment more accessible,
e.g `alias actpurl='source /path/to/purl/purl-venv/bin/activate'`

Then, install the dependencies by running:

```pip install -r requirements.txt```

or if you have `pip-sync` installed:

```pip-sync```



## Running `purl`

There are two main subcommands to PURL

### `train`

To train a model, run:

```
./purl train
```

For example, to train a model using the PPO algorithm on the `MiniGrid-LavaCrossingS9N1-v0` environment, use the following arguments:

```
./purl train --algorithm ppo --environment MiniGrid-LavaCrossingS9N1-v0
```

### `visualize`

To visualize a model, run:

```
./purl vizualize
```


## Development

### Updating dependencies
Python dependencies are managed by [`pip-compile`](https://github.com/jazzband/pip-tools#installation)

To add a new package, simply add it to the list in `requirements.in`.

You then update the `requirements.txt`-file by running

```pip-compile --output-file requirements.txt requirements.in```



## Built With

* [gym-minigrid](https://github.com/maximecb/gym-minigrid) - Minimalistic gridworld environment for OpenAI Gym
* [PyTorch](https://github.com/pytorch/pytorch) - Tensors and Dynamic neural networks in Python



## Authors

* Anne Engström
* Joel Lidin
* Gustav Molander
* Olle Månsson
* Noa Onoszko
* Hugo Ölund



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
