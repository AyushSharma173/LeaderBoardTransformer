# run_sweep.py
import wandb
import modeling  # make sure modeling.py is in the same folder or properly importable
import regex
if __name__ == "__main__":
    wandb.agent(
        "ayushs5-university-of-illinois-urbana-champaign/LeaderBoardTransformer-assignment1-basics_cs336_basics_cs336_basics/bw4th99i",
        function=modeling.sweep_run
    )

    # text = """I'll say supercalifragilistixecplfidious"""
    # segments = regex.findall(r"\w+| .", text)
    # print(segments)