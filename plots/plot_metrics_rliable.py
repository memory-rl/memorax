from rliable import library as rly
from rliable import metrics
from rliable import plot_utils
 

import pandas as pd

algorithms = ["RPPO"]

path = "/home/farr/memory-rl/logs/Craftax-Symbolic-v1/rppo/GRUCell/0/20250831-212507/evaluation/episodic_returns.csv"

df = pd.read_csv(path)

print(df.to_dict().keys())
