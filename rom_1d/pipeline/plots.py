"""Shared plotting helpers for the analysis tools (one home for the per-motor grid)."""
import matplotlib; matplotlib.use('Agg')          # noqa: E402
import matplotlib.pyplot as plt
import numpy as np


def motor_grid(figsize=(15, 16)):
    """A 4x2 grid of axes (one per actuator, last cell spare). Returns (fig, axes_flat)."""
    fig, axes = plt.subplots(4, 2, figsize=figsize, sharex=True)
    return fig, axes.ravel()


def plot_true_vs_virtual(ax, t, true, virt, color, title, post=None, extra=None):
    """One panel: measured truth (solid) vs model virtual (dashed).

    post : boolean mask of the dropped/predicted window. None -> virtual over the whole run;
           otherwise the pre-window truth is drawn faded and virtual only over `post`.
    extra: optional (label, series) tuple to overlay (e.g. battery/torso temp).
    """
    ax.plot(t, true, lw=1.5, color=color, label='true (withheld)')
    if post is None:
        ax.plot(t, virt, '--', lw=1.6, color='k', label='virtual (model)')
    else:
        ax.plot(t[post], virt[post], '--', lw=1.6, color='k', label='virtual (model)')
        ax.plot(t[~post], true[~post], lw=1.5, color=color, alpha=.35)
    if extra is not None:
        ax.plot(t, extra[1], ':', lw=1.1, color='tab:brown', label=extra[0])
    ax.set_title(title, fontsize=9.5); ax.set_ylabel('motor T [C]')
    ax.grid(axis='x', alpha=.3); ax.legend(fontsize=8, loc='best')


def finish_grid(fig, axes, n_used, suptitle, out, dpi=600):
    """Hide spare axes, label the bottom row, title, save."""
    for j in range(n_used, len(axes)):
        axes[j].axis('off')
    for ax in axes[max(0, n_used - 2):n_used]:
        ax.set_xlabel('time [min]')
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(); fig.savefig(out, dpi=dpi)
    return out


def score(virt, true, mask=None):
    """(max|err|, RMSE, end-err) of virt vs true over finite truth (optionally within mask)."""
    if mask is not None:
        virt, true = virt[mask], true[mask]
    ok = np.isfinite(true)
    if not ok.any():
        return np.nan, np.nan, np.nan
    e = virt[ok] - true[ok]
    return float(np.nanmax(np.abs(e))), float(np.sqrt(np.nanmean(e ** 2))), float(e[-1])
