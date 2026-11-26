"""Helpers for the Slice Crawl puzzle: name the best shop in few slices.

The estimator uses only the standard library, so it runs both in a notebook
and on the practice site (Pyodide). Only the plot functions need matplotlib.

    from slice_helpers import estimate          # default settings
    from slice_crawl import evaluate
    evaluate(estimate)

Method (successive elimination with focused tasting):
    1. taste every shop N0 times
    2. repeat until one shop is left:
       - leader = shop with the best average so far
       - drop rival i if the leader's average beats it by more than
         c * sqrt(1 + a / n) standard errors of the difference, where n is
         the fewer slices of the two (strict on thin evidence, looser later)
       - taste the leader and the M rivals closest to being dropped
    3. stop at one shop, or name the leader after CAP slices

Each shop's spread is estimated from its own slices combined with the range
the crawl generator draws spreads from (sd log-uniform on [0.5, 2.5]): the
posterior mean of the variance under that prior. With 2 slices this leans
on the range; as slices pile up, the shop's own slices take over.
"""
import math

from slice_crawl import SliceCrawl

# spread range of the crawl generator (see SliceCrawl.sample)
SD_LO, SD_HI = .5, 2.5

# grid over the shop's sd, evenly spaced in log(sd): log-uniform prior
_N_GRID = 60
_SD = [math.exp(math.log(SD_LO) + (math.log(SD_HI) - math.log(SD_LO))
                * j / (_N_GRID - 1)) for j in range(_N_GRID)]
_LOG_SD = [math.log(s) for s in _SD]
_VAR = [s * s for s in _SD]


def spread_estimate(ss, df):
    """Posterior mean of a shop's variance.

    Args:
        ss (float): sum of squared deviations of the shop's slices from
            their average
        df (int): degrees of freedom (slices - 1)

    Returns:
        var (float): estimated variance of the shop's ratings
    """
    log_w = [-df * lsd - ss / (2 * v) for lsd, v in zip(_LOG_SD, _VAR)]
    top = max(log_w)
    w = [math.exp(lw - top) for lw in log_w]
    return sum(wi * v for wi, v in zip(w, _VAR)) / sum(w)


def make_estimator(c=.8, a=40, n0=2, m=2, cap=3000):
    """Build an estimate(crawl) function.

    Args:
        c (float): elimination cutoff once evidence is plentiful, in
            standard errors. Higher is safer and uses more slices
            (0.80: ~96.6% right, ~75 slices; 0.75: ~95.8%, ~70 slices)
        a (float): how much stricter cuts are while slices are few
        n0 (int): first slices per shop
        m (int): rivals tasted each round alongside the leader
        cap (int): most slices one crawl may taste

    Returns:
        estimate (function): estimate(crawl) tastes the crawl and returns
            the index of the shop it names
    """
    def estimate(crawl):
        k = crawl.n_shop
        n, tot, tot2 = [0] * k, [0.] * k, [0.] * k

        def taste(i):
            r = crawl.taste(i)
            n[i] += 1
            tot[i] += r
            tot2[i] += r * r

        for i in range(k):
            for _ in range(n0):
                taste(i)

        alive = list(range(k))
        var_at = {}                          # shop -> (n, var) cache
        while True:
            mean = [tot[i] / n[i] for i in range(k)]
            var = [0.] * k
            for i in alive:
                if var_at.get(i, (None,))[0] != n[i]:
                    ss = max(tot2[i] - n[i] * mean[i] ** 2, 1e-9)
                    var_at[i] = (n[i], spread_estimate(ss, n[i] - 1))
                var[i] = var_at[i][1]

            lead = max(alive, key=mean.__getitem__)
            keep, closeness = [lead], {}
            for i in alive:
                if i == lead:
                    continue
                z = (mean[lead] - mean[i]) / math.sqrt(var[lead] / n[lead]
                                                       + var[i] / n[i])
                cut = c * math.sqrt(1 + a / min(n[lead], n[i]))
                if z <= cut:
                    keep.append(i)
                    closeness[i] = z / cut   # near 1: about to be dropped
            alive = keep
            if len(alive) == 1 or crawl.n_taste >= cap:
                return int(lead)

            taste(lead)
            for i in sorted(closeness, key=closeness.get)[:m]:
                taste(i)

    return estimate


# default estimator, ready for evaluate() or the practice site
estimate = make_estimator()


def taste_each(k):
    """Baseline from the prompt: k slices from every shop, best average."""
    def estimate_k(crawl):
        avg = [sum(crawl.taste(i) for _ in range(k)) / k
               for i in range(crawl.n_shop)]
        return avg.index(max(avg))
    return estimate_k


def run(estimate_fn, n_crawl=2000, seed=1, **setup):
    """Like slice_crawl.evaluate (same crawls for the same seed and setup),
    but also records each crawl's gap between its best two shops. The gap is
    hidden truth, used only for scoring, never by the estimator.

    Returns:
        result (dict): accuracy, accuracy_se, mean_slices, and per-crawl
            lists slices, correct, gap
    """
    slices, correct, gap = [], [], []
    for idx in range(n_crawl):
        crawl = SliceCrawl.sample(**setup, seed=seed * 1_000_003 + idx)
        guess = estimate_fn(crawl)
        slices.append(crawl.n_taste)
        correct.append(guess == crawl._best)
        top = sorted(crawl._mean)
        gap.append(top[-1] - top[-2])
    acc = sum(correct) / n_crawl
    return dict(accuracy=acc,
                accuracy_se=math.sqrt(acc * (1 - acc) / n_crawl),
                mean_slices=sum(slices) / n_crawl,
                slices=slices, correct=correct, gap=gap)


def plot_tradeoff(ax, curves, target=.95):
    """Accuracy vs. mean slices, one line per estimator family.

    Args:
        ax: matplotlib axes
        curves (dict): name -> (point labels, list of run() results)
        target (float): accuracy bar to draw
    """
    from matplotlib.ticker import ScalarFormatter
    for name, (labels, res) in curves.items():
        x = [r['mean_slices'] for r in res]
        y = [r['accuracy'] for r in res]
        err = [2 * r['accuracy_se'] for r in res]
        ax.errorbar(x, y, yerr=err, marker='o', capsize=3, label=name)
        for lab, xx, yy in zip(labels, x, y):
            ax.annotate(lab, (xx, yy), textcoords='offset points',
                        xytext=(5, -10), fontsize=7)
    ax.axhline(target, color='k', ls='--', label=f'{target:.0%} bar')
    ax.set_xscale('log')
    ax.set_xticks([50, 100, 200, 400, 800])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel('slices tasted per crawl (mean)')
    ax.set_ylabel('fraction of crawls naming the best shop')
    ax.grid(alpha=.3)
    ax.legend()


def plot_accuracy_by_gap(ax, res, edges=(.3, .4, .5, .65, .85, 1.1, 1.5)):
    """Accuracy and mean slices per bin of the gap between the top two shops.

    Args:
        ax: matplotlib axes (accuracy on the left axis, slices on the right)
        res (dict): a run() result
        edges (tuple): gap bin edges
    """
    mids, acc, se, sl = [], [], [], []
    for lo, hi in zip(edges, edges[1:]):
        idx = [j for j, g in enumerate(res['gap']) if lo <= g < hi
               or (hi == edges[-1] and g == hi)]
        if not idx:
            continue
        p = sum(res['correct'][j] for j in idx) / len(idx)
        mids.append(math.sqrt(lo * hi))
        acc.append(p)
        se.append(2 * math.sqrt(p * (1 - p) / len(idx)))
        sl.append(sum(res['slices'][j] for j in idx) / len(idx))
    ax.errorbar(mids, acc, yerr=se, marker='o', capsize=3, color='C0',
                label='right (±2 SE)')
    ax.axhline(.95, color='k', ls='--', lw=1)
    ax.set_xscale('log')
    ax.set_xticks(list(edges))
    ax.set_xticklabels([f'{e:g}' for e in edges])
    from matplotlib.ticker import NullFormatter
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel('gap between best and second-best shop (rating points)')
    ax.set_ylabel('fraction of crawls naming the best shop', color='C0')
    ax.grid(alpha=.3)
    ax2 = ax.twinx()
    ax2.plot(mids, sl, marker='s', color='C1', label='slices (mean)')
    ax2.set_ylabel('slices tasted per crawl (mean)', color='C1')
    return ax2
