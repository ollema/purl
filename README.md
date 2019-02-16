# purl
**P**athfinding **U**sing **R**einforcement **L**earning



## Features of `purl`

* It has cool name!
* ...



## Getting Started
### Prerequisites

`python` + `pip`, version 3.6 or greater

### Installing
**Note**: It's recommended that you install the Python dependencies in a virtual environment.  `virtualenv` and `virtualenvwrapper`:

```
pip install virtualenv
pip install virtualenvwrapper
```

#### Setting up a virtual environment

```mkvirtualenv purl```

#### Installing Python dependencies

First, switch to the virtual environment:

```workon purl```

Then, install the dependencies by running:

```pip install -r requirements.txt```

or if you have `pip-sync` installed:

```pip-sync```



## Running `purl`

There are three main subcommands to PURL

### `train`

To train a model, run:

```
purl train
```

### `visualize`

To visualize a model, run:

```
purl vizualize
```

### `evaluate`

To evaluate a model, run:

```
purl evaluate
```


## Development

### Updating dependencies
Python dependencies are managed by [`pip-compile`](https://github.com/jazzband/pip-tools#installation)

To add a new package, simply add it to the list in `requirements.in`.

You then update the `requirements.txt`-file by running

```pip-compile --output-file requirements.txt requirements.in```



## Built With

* [gym-minigrid](https://github.com/maximecb/gym-minigrid) - Minimalistic gridworld environment for OpenAI Gym



## Authors

* Anne Engström
* Joel Lidin
* Gustav Molander
* Olle Månsson
* Noa Onoszko
* Hugo Ölund



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details



## Acknowledgments

* Blockchain
* Internet of Things
* AI
* Machine Learning
* Cloud Computing
