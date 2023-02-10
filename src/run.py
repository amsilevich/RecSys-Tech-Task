import click
import pandas as pd
import yaml

from src.orchestrators.collaborative_filtering_orchestrator import CFOrchestrator
from src.datasets.interactions import InteractionsDataset


@click.command()
@click.option('--data', default='./data/interactions.csv', help='Path to data')
@click.option('--config', default='./src/configs/cf_config.yaml', help='Path to pipeline configuration')
def main(data: str, config: str):
    interactions = InteractionsDataset(pd.read_csv(data))

    with open(config, "r") as config:
        config = yaml.safe_load(config)

    orchestrator = CFOrchestrator(interactions=interactions, config=config)
    print(f"MAP@10 of ALS model is {orchestrator.run()}")


if __name__ == '__main__':
    main()
