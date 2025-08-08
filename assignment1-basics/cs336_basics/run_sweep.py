# run_sweep.py
import wandb
import modeling  # make sure modeling.py is in the same folder or properly importable
import regex
if __name__ == "__main__":
    wandb.agent(
        "ayushs5-university-of-illinois-urbana-champaign/assignment1-basics-cs336_basics/lkjbtese",
        function=modeling.sweep_run
    )

    # text = """I'll say supercalifragilistixecplfidious"""
    # segments = regex.findall(r"\w+| .", text)
    # print(segments)