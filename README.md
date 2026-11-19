# Style classification from MIDI drum tracks

## Introduction

This project aims at classify MIDI drum tracks into musical genres.
We use the [Groove MIDI Dataset](https://magenta.tensorflow.org/datasets/groove)
from Google Magenta.

Multiple classifiers are implemented, in order to compare their performances.

## Disclaimer

This repository contains the code I used to produce the results presented in the article ["Improved symbolic drum style classification with grammar-based hierarchical representations"](https://hal.science/hal-04660056) published in ISMIR 2024 proceedings.
The objective of the paper is to evaluate several representations on a well-defined benchmark, in order to evaluate how sequence-based classifiers would behave with each one.
The codebase has not really been cleaned (and contains some coding tricks that I am not particularly proud of), but I still wanted to publish it publicily as reference for future works.

## Usage

I had trouble building [`qparse`](https://qparse.gitlabpages.inria.fr/) on some machines, so I ended up running all my code in a Docker container for reproducible builds:
```bash
cd /path/to/project
docker build -t groove .  # build the image
docker run -d --name groove-container -v "$(pwd)":/app groove  # start the container, mapping the cwd into the container
docker exec -it my-running-container /bin/bash  # log into the container
```

If you prefer running without Docker, the `Dockerfile` contains all needed information to build the environment bare metal.
