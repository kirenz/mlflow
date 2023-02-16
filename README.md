# MLflow

MLflow is an open source platform for managing the end-to-end machine learning lifecycle. It tackles four primary functions:

- Tracking experiments to record and compare parameters and results (MLflow Tracking).

- Packaging ML code in a reusable, reproducible form in order to share with other data scientists or transfer to production (MLflow Projects).

- Managing and deploying models from a variety of ML libraries to a variety of model serving and inference platforms (MLflow Models).

- Providing a central model store to collaboratively manage the full lifecycle of an MLflow Model, including model versioning, stage transitions, and annotations (MLflow Model Registry).

## Prerequisites

- You need to have Anaconda or Miniconda installed on your machine
  - [Miniconda installation guide](https://kirenz.github.io/codelabs/codelabs/miniconda/#0) (recommended)
  - [Anaconda installation guide](https://kirenz.github.io/codelabs/codelabs/anaconda-install/#0)


## Python Setup

We first setup a Python environment for MLflow (you also need this environment if you want to use the R API).

- Open your shell and create a new directory named *ml* within the current directory:

```bash
mkdir ml
```

- Change directory inside *ml*

```bash
cd ml
```

- Clone (i.e. copy) the GitHub-repo *mlflow*:

```bash
git clone https://github.com/kirenz/mlflow
```

- Change directory into mlflow

```bash
cd mlflow
```


- We use the file environment.yml to create a new Anaconda environment with all necessary Python packages:

```bash
conda env create -f environment.yml
```



- Follow the steps outlined in [mlflow-setup.md](mlflow-setup.md)