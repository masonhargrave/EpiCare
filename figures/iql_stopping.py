import matplotlib.pyplot as plt
from tueplots import bundles
import numpy as np
import pandas as pd

plt.ion()
plt.rcParams.update(bundles.icml2022())

# Load the data and rename the columns from their super verbose titles.
Ns = 2 ** np.arange(8, 18)
df = pd.read_csv("IQL_scores.csv", index_col="Step")
renamer = {f"episodes_avail: {N} - normalized_score": f"N{N}" for N in Ns}
renamer |= {
    k + "__" + metric.upper(): v + "_" + metric
    for k, v in renamer.items()
    for metric in ["min", "max"]
}
df.rename(
    inplace=True,
    errors="raise",
    columns=renamer,
)

# Perform strong exponential moving average smoothing, then compute the optimal
# stopping point as the point where the smoothed curve is maximized.
df = df.ewm(alpha=0.01).mean()
peaks = df.idxmax()[[f"N{N}" for N in Ns]]

f = plt.figure("IQL Data Restriction", figsize=(5.5, 3))
A, B = f.subplots(1, 2, gridspec_kw={"width_ratios": [2, 1]})

for i, N in enumerate(Ns):
    A.semilogx(df.index, df[f"N{N}"], label=f"${N = }$")
    A.fill_between(df.index, df[f"N{N}_min"], df[f"N{N}_max"], color=f"C{i}", alpha=0.1)

B.loglog(Ns, peaks)

A.set_xlabel("Training Steps")
A.set_ylabel("Mean Episode Reward")
A.legend(ncols=2)

B.set_xlabel("Episodes Trained On")
B.set_ylabel("Optimal Stopping Point (Training Steps)")

f.savefig("iql_es.pdf")
