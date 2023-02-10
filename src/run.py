import pandas as pd
import yaml

from src.orchestrators.collaborative_filtering_orchestrator import CFOrchestrator
from src.datasets.interactions import InteractionsDataset


def main():
    interactions = InteractionsDataset(pd.read_csv('./data/interactions.csv'))

    with open("./src/configs/cf_config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    orchestrator = CFOrchestrator(interactions=interactions, config=config)
    print(f"MAP@10 of ALS model is {orchestrator.run()}")


if __name__ == '__main__':
    main()
