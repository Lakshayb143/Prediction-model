# Prediction-model


### 1. Installation and setup

Install ZenML - https://docs.zenml.io/getting-started/installation

Once virtualenv environment is activated, run following command:
pip install -r requirements.txt

If you are running the run_deployment.py script, you will also need to install some
integrations using ZenML:

```
zenml integration install mlflow -y
```

The project can only be executed with a ZenML stack that has an MLflow experiment
tracker and model deployer as a component. Configuring a new stack with the two
components are as follows:

```bash

zenml integration install mlflow -y

zenml experiment-tracker register mlflow_tracker --flavor=mlflow

zenml model-deployer register mlflow --flavor=mlflow

zenml stack register local-mlflow-stack -a default -o default -d mlflow -e mlflow_tracker
--set
```