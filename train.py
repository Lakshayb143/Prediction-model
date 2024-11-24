import click
from pipelines.train_pipeline import ml_pipeline
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

@click.command()
def main():

    run = ml_pipeline()

    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the experiment."
    )

if __name__ == "__main__":
    main()




 # mlflow ui --backend-store-uri 'file:C:\Users\coder\AppData\Roaming\zenml\local_stores\fd312ea6-3e6b-445e-8588-b449d13f6bb6\mlruns'
