import pandas as pd
from pathlib import Path
from tira.check_format import TrecEvalLeaderboard, _fmt
from statistics import mean, stdev
from typing import Optional

class TrecLeaderboardEvaluation():
    def __init__(self, truth_leaderboard: Optional[Path], truth_measure: Optional[str]):
        if truth_leaderboard and truth_measure:
            parsed_leaderboard = self.load_leaderboard(truth_leaderboard)
            self.ground_truth_ranking = self.extract_ranking(parsed_leaderboard, truth_measure)
        else:
            self.ground_truth_ranking = None

    def load_leaderboard(self, leaderboard: Path):
        if not leaderboard or not Path(leaderboard).is_file():
            raise ValueError(f"I expected that {leaderboard} is a file.")

        reader = TrecEvalLeaderboard()
        reader.apply_configuration_and_throw_if_invalid({})
        c, m = reader.check_format(leaderboard)
        if c != _fmt.OK:
            raise ValueError(f"Can not load {leaderboard}. {m}")

        return reader.all_lines(leaderboard)

    def extract_ranking(self, leaderboard, measure):
        ret = {}
        all_measuers = set()
        for i in leaderboard:
            if i["query"] != "all":
                continue
            all_measuers.add(i["metric"])
            if str(i["metric"]).strip() == str(measure).strip():
                ret[i["run"]] = i["value"]
        if len(ret) == 0:
            raise ValueError(f"Measure {measure} does not exist, I found: {sorted(list(all_measuers))}")
        return ret

    def evaluate(self, leaderboard_file):
        leaderboard = self.load_leaderboard(leaderboard_file)
        measures = set([i["metric"] for i in leaderboard])
        ret = {}

        for m in measures:
            if self.ground_truth_ranking:
                ret[m] = self.correlation_to_truth(self.extract_ranking(leaderboard, m))
            else:
                ret[m] = self.basic_statistics(leaderboard, m)

        return ret

    def basic_statistics(self, leaderboard, measure):
        vals = []

        for i in leaderboard:
            if str(i["metric"]).strip() == str(measure).strip():
                vals.append(float(i["value"]))

        if len(vals) == 0:
            raise ValueError(f"Measure {measure} does not exist.")

        return {"mean-value": mean(vals), "stdev-value": stdev(vals)}

    def correlation_to_truth(self, ranking):
        a, b = [], []

        for system, truth_score in self.ground_truth_ranking.items():
            a.append(float(truth_score))
            b.append(float(ranking[system]))

        return {
            "kendall": correlation(a, b, "kendall"),
            "pearson": correlation(a, b, "pearson"),
            "spearman": correlation(a, b, "spearman"),
            "tauap_b": tauap_b(a, b)
        }


def _check_input_or_raise(a, b):
    if len(a) < 3:
        raise ValueError(f"Can not calculate correlations on only {len(a)} elements.")
    if len(a) != len(b):
        raise ValueError(f"Can not calculate correlations on unequal elements: {len(a)} != {len(b)}")

def correlation(a, b, method):
    _check_input_or_raise(a, b)

    df = pd.DataFrame([{"a": i, "b": j} for i, j in zip(a, b)])

    return float(df.corr(method).iloc[0]["b"])

def tauap_b(a, b):
    from .pyircore import tauap_b as method

    _check_input_or_raise(a, b)
    return method(a, b)
