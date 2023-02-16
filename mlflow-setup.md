# MLflow Python Setup


## Prerequisites

- You need to have Anaconda or Miniconda installed on your machine
  - [Miniconda installation guide](https://kirenz.github.io/codelabs/codelabs/miniconda/#0) (recommended)
  - [Anaconda installation guide](https://kirenz.github.io/codelabs/codelabs/anaconda-install/#0)


## Prepare working environment

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

## Create Anaconda environment

- We use the file environment.yml to create a new Anaconda environment called *mlflow* and install all necessary Python packages:

```bash
conda env create -f environment.yml
```

- Activate the environment:

```bash
conda activate mlflow
```

## MLflow user interface

- Start the MLflow user interface

```
mlflow ui
```

- This should output something like the following:

```bash
[2023-02-16 17:31:16 +0100] [79245] [INFO] Starting gunicorn 20.1.0
[2023-02-16 17:31:16 +0100] [79245] [INFO] Listening at: http://127.0.0.1:5000 (79245)
[2023-02-16 17:31:16 +0100] [79245] [INFO] Using worker: sync
[2023-02-16 17:31:16 +0100] [79246] [INFO] Booting worker with pid: 79246
[2023-02-16 17:31:16 +0100] [79247] [INFO] Booting worker with pid: 79247
[2023-02-16 17:31:16 +0100] [79248] [INFO] Booting worker with pid: 79248
[2023-02-16 17:31:16 +0100] [79249] [INFO] Booting worker with pid: 79249
```
- Copy the URL in line 2 into your browser. This should open the MLflow user interface:

![](/img/mlflow-ui.png)


*Congratulations, MLflow is up and running!*

If your are done, you can shut down the MLflow ui inside your terminal with the shortcut `Control` + `C`

You can now proceed with the examples in the folder Python or R.