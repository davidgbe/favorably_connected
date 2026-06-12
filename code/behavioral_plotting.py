import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import beta, stats
from copy import deepcopy as copy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler

from aux_funcs import format_plot
from nb_analysis_tools import (
    load_trajectory_data,
    parse_behavioral_data,
    get_session_summaries,
    get_all_session_summaries_pkl,
)

color_high_reward = '#d95f02'
color_low_reward  = '#1b9e77'
color_unrewarded  = '#7570b3'
color_intersite   = '#808080'
color_interpatch  = '#b3b3b3'

odor_colors = [
    color_unrewarded,
    color_low_reward,
    color_high_reward,
]


def clopper_pearson(k, n, alpha=0.05):
    lower = np.where(k == 0, 0, beta.ppf(alpha / 2, k, n - k + 1))
    upper = np.where(k == n, 1., beta.ppf(1 - alpha / 2, k + 1, n - k))
    return np.stack([lower, upper])


def plot_session(data_path, session_idx, xlim=None, max_reward_param=30,
                 max_reward_sites=8, use_fixed_colors=True, save_dir=None, color_by_reward_param=True):

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    b_data_raw = load_trajectory_data(data_path)
    b_data = parse_behavioral_data(b_data_raw[session_idx])
    ss = get_session_summaries([b_data])[0]

    rewards_at_positions = b_data['rewards_at_positions']
    reward_attempted_at_positions = b_data['reward_attempted_at_positions']
    all_patch_nums = b_data['current_patch_num']
    all_patch_reward_params = b_data['patch_reward_param']
    if not color_by_reward_param:
        all_patch_reward_prob_prefactors = b_data['patch_reward_prob_prefactor']
        print(np.unique(all_patch_reward_prob_prefactors))

    scale = 0.8
    fig, axs = plt.subplots(1, 1, figsize=(10 * scale, 3 * scale))

    axs.plot(np.arange(len(b_data['dwell_times_at_positions'])),
             1 / np.array(b_data['dwell_times_at_positions']), c='black', zorder=0)
    axs.scatter(np.arange(len(rewards_at_positions))[rewards_at_positions > 0],
                rewards_at_positions[rewards_at_positions > 0] * 2, c='blue', marker='*')
    axs.scatter(np.arange(len(reward_attempted_at_positions))[reward_attempted_at_positions > 0],
                reward_attempted_at_positions[reward_attempted_at_positions > 0] * 2.5,
                c='black', marker='s', s=3)

    cmap = mpl.colormaps['magma']

    s_0 = 1.1
    s_1 = 1.75
    last_pstart = None
    pb = None
    last_reward_site_start = None
    reward_site_start = None
    patch_count = 0
    rw_site_counter = 0

    for i, pstart in enumerate(b_data['current_patch_start']):
        if last_pstart is None or (pstart != last_pstart).any():
            patch_count += 1
            pb = [pstart, pstart]
            if use_fixed_colors:
                c = odor_colors[all_patch_nums[i]]
            else:
                if color_by_reward_param:
                    c = (cmap(all_patch_reward_params[i] / max_reward_param)
                     if all_patch_reward_params[i] > 0 else 'black')
                else:
                    c = (cmap(all_patch_reward_prob_prefactors[i])
                         if all_patch_reward_prob_prefactors[i] > 0 else 'black')
            rw_site_counter = 0

        rwsb = copy(b_data['reward_bounds'][i])
        reward_site_start = int(rwsb[0])
        if last_reward_site_start is None or not np.isclose(last_reward_site_start, reward_site_start):
            axs.fill_between(list(rwsb), s_0 * np.ones(2), y2=s_1 * np.ones(2),
                             alpha=0.5, color=c, zorder=-1)
            axs.fill_between(list(rwsb), np.zeros(2), y2=s_0 * np.ones(2),
                             alpha=0.2, color=c, zorder=-1)

            if np.sum(reward_attempted_at_positions[int(rwsb[0]):int(rwsb[1])]) == 0:
                pb[1] = rwsb[1]
                axs.fill_between(pb, s_0 * np.ones(2), s_1 * np.ones(2),
                                 alpha=0.2, color=c, zorder=-2)
                axs.fill_between(pb, np.zeros(2), s_0 * np.ones(2),
                                 alpha=0.05, color=c, zorder=-2)
            else:
                rw_site_counter += 1

        last_pstart = pstart
        last_reward_site_start = reward_site_start

    if xlim is not None:
        axs.set_xlim(xlim[0], xlim[1])
    axs.set_ylim(0)
    axs.set_ylabel('Avg. running speed')
    axs.set_xlabel('Position')
    axs.set_yticks([0, 1])
    format_plot(axs, axislabelsize=13, ticklabelsize=12, leftspine=False)
    fig.tight_layout()
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, 'raw_behavioral_trace.png'))

    labels = ['Unrewarded', 'Low reward', 'High reward']
    if color_by_reward_param:
        non_fixed_colors = [
            (cmap(rp / max_reward_param) if rp > 0 else 'black')
            for rp in ss['reward_param_for_patch_type']
        ]
    else:
        non_fixed_colors = [
            (cmap(rpp) if rpp > 0 else 'black')
            for rpp in ss['reward_prob_prefactor_for_patch_type']
        ]

    scale = 1.5
    fig, axs = plt.subplots(3, 1, figsize=(3 * scale, 3 * scale), sharex=True, sharey=True)
    x = np.arange(ss['site_stops_for_patch_type'].shape[1])
    for k in range(ss['site_stops_for_patch_type'].shape[0]):
        c = odor_colors[k] if use_fixed_colors else non_fixed_colors[k]
        cis = clopper_pearson(ss['site_stops_for_patch_type'][k, :],
                              ss['site_stop_opportunities_for_patch_type'][k, 0])
        axs[k].fill_between(x, cis[0, :], cis[1, :], color=c, alpha=0.2)
        axs[k].plot(x, ss['site_stops_for_patch_type'][k, :] /
                    ss['site_stop_opportunities_for_patch_type'][k, 0], color=c, label=labels[k])
        axs[k].set_ylim(0, 1.1)
    axs[1].set_ylabel('Fraction attempted')
    axs[2].set_xlabel('Reward site number in patch')
    fig.legend()
    format_plot(axs, axislabelsize=13, ticklabelsize=12)
    fig.tight_layout()
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, 'behavioral_stops_stats_rws_frac_given_entered.png'))

    scale = 1.5
    fig, axs = plt.subplots(3, 1, figsize=(3 * scale, 3 * scale), sharex=True, sharey=True)
    x = np.arange(ss['site_stops_for_patch_type'].shape[1])
    for k in range(ss['site_stops_for_patch_type'].shape[0]):
        c = odor_colors[k] if use_fixed_colors else non_fixed_colors[k]
        ss['site_stop_opportunities_for_patch_type'][k, :] = np.where(
            ss['site_stop_opportunities_for_patch_type'][k, :] == 0,
            np.nan, ss['site_stop_opportunities_for_patch_type'][k, :])
        cis = clopper_pearson(ss['site_stops_for_patch_type'][k, :],
                              ss['site_stop_opportunities_for_patch_type'][k, :])
        axs[k].fill_between(x, cis[0, :], cis[1, :], color=c, alpha=0.2)
        axs[k].plot(x, ss['site_stops_for_patch_type'][k, :] /
                    ss['site_stop_opportunities_for_patch_type'][k, :], color=c, label=labels[k])
        axs[k].set_ylim(0, 1.1)
    axs[1].set_ylabel('Fraction attempted')
    axs[2].set_xlabel('Reward site number in patch')
    fig.legend()
    format_plot(axs, axislabelsize=13, ticklabelsize=12)
    fig.tight_layout()
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, 'behavioral_stops_stats_rws_frac_given_site.svg'))
        fig.savefig(os.path.join(save_dir, 'behavioral_stops_stats_rws_frac_given_site.png'))

    scale = 1.5
    fig, axs = plt.subplots(3, 1, figsize=(3 * scale, 3 * scale), sharex=True, sharey=True)
    x = np.arange(ss['site_stops_for_patch_type'].shape[1])
    for k in range(ss['site_stops_for_patch_type'].shape[0]):
        c = odor_colors[k] if use_fixed_colors else non_fixed_colors[k]
        pt = k
        if k == 0:
            axs[k].bar(x, ss['site_stop_opportunities_for_patch_type'][pt, :],
                       color='gray', alpha=0.5, label='Sites entered')
        else:
            axs[k].bar(x, ss['site_stop_opportunities_for_patch_type'][pt, :],
                       color='gray', alpha=0.5)
        axs[k].bar(x, ss['site_stops_for_patch_type'][pt, :], color=c, label=labels[k])
    axs[1].set_ylabel('Counts')
    axs[2].set_xlabel('Reward site number in patch')
    fig.legend()
    format_plot(axs, axislabelsize=13, ticklabelsize=12)
    fig.tight_layout()
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, 'behavioral_stops_stats_rws_counts.svg'))
        fig.savefig(os.path.join(save_dir, 'behavioral_stops_stats_rws_counts.png'))

    scale = 1.5
    fig, axs = plt.subplots(3, 1, figsize=(3 * scale, 3 * scale), sharex=True, sharey=True)
    x = np.arange(ss['acc_reward_stops_for_patch_type'].shape[1])
    for k in range(ss['acc_reward_stops_for_patch_type'].shape[0]):
        c = odor_colors[k] if use_fixed_colors else non_fixed_colors[k]
        opp  = ss['acc_reward_stop_opportunities_for_patch_type'][k, :]
        runs = ss['acc_reward_stops_for_patch_type'][k, :]
        runs = np.where(opp == 0, np.nan, runs)
        opp = np.where(opp == 0, np.nan, opp)
        cis = clopper_pearson(runs, opp)
        axs[k].fill_between(x, cis[0, :], cis[1, :], color=c, alpha=0.2)
        axs[k].plot(x, runs / opp, color=c, label=labels[k])
        axs[k].set_ylim(0, 1.1)
    axs[0].set_xlim(-1)
    axs[1].set_ylabel('Fraction attempted')
    axs[2].set_xlabel('Total reward in patch')
    fig.legend()
    format_plot(axs, axislabelsize=13, ticklabelsize=12)
    fig.tight_layout()
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, 'behavioral_stops_stats_reward_frac.svg'))
        fig.savefig(os.path.join(save_dir, 'behavioral_stops_stats_reward_frac.png'))

    scale = 1.5
    fig, axs = plt.subplots(3, 1, figsize=(3 * scale, 3 * scale), sharex=True, sharey=True)
    x = np.arange(ss['acc_reward_stops_for_patch_type'].shape[1])
    for k in range(ss['acc_reward_stops_for_patch_type'].shape[0]):
        c = odor_colors[k] if use_fixed_colors else non_fixed_colors[k]
        if k == 0:
            axs[k].bar(x, ss['acc_reward_stop_opportunities_for_patch_type'][k, :],
                       color='grey', alpha=0.5, label='Stop opportunities')
        else:
            axs[k].bar(x, ss['acc_reward_stop_opportunities_for_patch_type'][k, :],
                       color='grey', alpha=0.5)
        axs[k].bar(x, ss['acc_reward_stops_for_patch_type'][k, :], color=c, label=labels[k])
    axs[0].set_xlim(-1, max_reward_param)
    axs[1].set_ylabel('Counts')
    axs[2].set_xlabel('Total reward in patch')
    fig.legend()
    format_plot(axs, axislabelsize=13, ticklabelsize=12)
    fig.tight_layout()
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, 'behavioral_stops_stats_reward_counts.svg'))
        fig.savefig(os.path.join(save_dir, 'behavioral_stops_stats_reward_counts.png'))


def load_odor_site_df(load_path, nn_num=0, lim=None, brief=False):
    all_session_data = get_all_session_summaries_pkl(load_path, lim=lim, brief=brief)
    odor_site_dfs = []
    for i_ss, ss in enumerate(all_session_data):
        odor_sites_for_session = []
        for odor_site_df in ss['all_odor_site_data']:
            odor_sites_for_session.append(odor_site_df)
        df = pd.concat(odor_sites_for_session)
        df['env_quality'] = np.sum(np.unique(df['patch_reward_param']))
        df['session_number'] = i_ss
        odor_site_dfs.append(df)
    dfs = pd.concat(odor_site_dfs)
    dfs['index / reward_param'] = dfs['index'] / (dfs['patch_reward_param'] + 1e-6)
    dfs['rewards_seen_in_patch / reward_param'] = (
        dfs['rewards_seen_in_patch'] / (dfs['patch_reward_param'] + 1e-6)
    )
    if 'patch_reward_prob_prefactor' in dfs.columns:
        dfs['index / prefactor'] = dfs['index'] / (dfs['patch_reward_prob_prefactor'] + 1e-6)
        dfs['rewards_seen_in_patch / prefactor'] = (
            dfs['rewards_seen_in_patch'] / (dfs['patch_reward_prob_prefactor'] + 1e-6)
        )
    dfs['network_num'] = nn_num
    return dfs


def plot_patch_statistics_by_session(df, max_reward_param=40, fixed_colors=False,
                                      x_axis='patch_reward_param', offsets=None,
                                      x_mask=None, x_tick_labels=None, ylim_right=None):
    """Per-session version: groups by session_number × x_axis (patch_reward_param or patch_type).

    x_mask : sequence of 0/1 (or bool) aligned to the sorted x_vals, controlling which
             x-values are plotted.  Colors are assigned before masking, so the palette
             is unaffected by which values are hidden.  Example: x_mask=[0,1,1] with
             x_axis='patch_type' plots only the 2nd and 3rd patch types.
    """
    by_type = x_axis == 'patch_type'
    x_col   = 'patch_type' if by_type else 'patch_reward_param'
    x_vals  = sorted(df[x_col].unique(), reverse=not by_type)

    # Tau lookup for reward_prob: mean patch_reward_param per x-value
    tau_for = (
        df.groupby('patch_type')['patch_reward_param'].mean().to_dict()
        if by_type
        else {v: v for v in x_vals}
    )

    if 'patch_reward_prob_prefactor' in df.columns:
        prefactor_for = (
            df.groupby('patch_type')['patch_reward_prob_prefactor'].mean().to_dict()
            if by_type
            else {v: v for v in x_vals}
        )

    session_data = []
    for session_num in sorted(df['session_number'].unique()):
        for i_xv, xv in enumerate(x_vals):
            offset = offsets[i_xv] if offsets is not None else 0.8
            crit = (df['session_number'] == session_num) & (df[x_col] == xv)
            mean_count = df[crit].groupby('patch_number')['rewards_seen_in_patch'].max().mean()
            tau = tau_for.get(xv, 0)
            if 'patch_reward_prob_prefactor' in df.columns:
                prefactor = prefactor_for.get(xv, 0)
            else:
                prefactor = offset
            session_data.append({
                'session_number': session_num,
                'x_val': xv,
                'rewards_collected': mean_count,
                'reward_prob': prefactor * np.exp(-mean_count / tau) if tau > 0 else 0,
            })

    summary_df = pd.DataFrame(session_data)

    # Build full color list first (preserves palette regardless of mask)
    cmap = mpl.colormaps['magma']
    if fixed_colors:
        colors = list(odor_colors[:len(x_vals)])
    elif by_type and 'patch_reward_prob_prefactor' in df.columns:
        prefactors = [prefactor_for[xv] for xv in x_vals]
        pf_min, pf_max = min(prefactors), max(prefactors)
        colors = (
            [cmap((pf - pf_min) / (pf_max - pf_min)) for pf in prefactors]
            if pf_max > pf_min else [cmap(0.5)] * len(x_vals)
        )
    elif by_type:
        colors = [cmap(i / max(len(x_vals) - 1, 1)) for i in range(len(x_vals))]
    else:
        colors = [(cmap(xv / max_reward_param) if xv > 0 else 'black') for xv in x_vals]

    # Apply mask to determine which x-values (and their colors) to plot
    if x_mask is not None:
        mask_bool    = [bool(m) for m in x_mask]
        x_vals_plot  = [xv for xv, m in zip(x_vals,  mask_bool) if m]
        colors_plot  = [c  for c,  m in zip(colors,  mask_bool) if m]
    else:
        x_vals_plot = x_vals
        colors_plot = colors

    plot_df = summary_df[summary_df['x_val'].isin(x_vals_plot)]

    box_kwargs = dict(
        width=0.4, palette=colors_plot, showcaps=True,
        boxprops={'edgecolor': 'black', 'linewidth': 1.2},
        medianprops={'color': 'black', 'linewidth': 1.5},
        whiskerprops={'color': 'black', 'linewidth': 1.0},
        capprops={'color': 'black', 'linewidth': 1.0},
        flierprops={'marker': 'o', 'markerfacecolor': 'white',
                    'markeredgecolor': 'black', 'markersize': 6, 'alpha': 1.0},
    )

    def _add_sig_bar(ax, y_col, y_offset, text_offset):
        if len(x_vals_plot) != 2:
            return
        a = plot_df[plot_df['x_val'] == x_vals_plot[0]][y_col]
        b = plot_df[plot_df['x_val'] == x_vals_plot[1]][y_col]
        _, pval = stats.ttest_rel(a, b)
        y_max = plot_df[y_col].max() + y_offset
        ax.plot([0, 1], [y_max, y_max], color='black')
        stars = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
        ax.text(0.5, y_max + text_offset, stars, ha='center', va='bottom', fontsize=12)

    n_plot = len(x_vals_plot)
    fig, axes = plt.subplots(1, 2, figsize=(n_plot * 1.2 + 1.2, 3), sharey=False)

    ax = axes[0]
    sns.boxplot(data=plot_df, x='x_val', y='rewards_collected', order=x_vals_plot, ax=ax,
                **box_kwargs)
    for _, group in plot_df.groupby('session_number'):
        ax.plot(np.arange(len(group)), group['rewards_collected'].values,
                color='grey', alpha=1, zorder=-1, lw=0.5)
    _add_sig_bar(ax, 'rewards_collected', 1, 0.3)
    ax.set_xlabel('')
    ax.tick_params(axis='x', bottom=False, labelbottom=False)
    ax.set_ylabel('Rewards collected')
    sns.despine(ax=ax, bottom=True)

    ax = axes[1]
    sns.boxplot(data=plot_df, x='x_val', y='reward_prob', order=x_vals_plot, ax=ax,
                **box_kwargs)
    for _, group in plot_df.groupby('session_number'):
        ax.plot(np.arange(len(group)), group['reward_prob'].values,
                color='grey', alpha=1, zorder=-1, lw=0.5)
    _add_sig_bar(ax, 'reward_prob', 0.05, 0.02)
    ax.set_xlabel('')
    ax.tick_params(axis='x', bottom=False, labelbottom=False)
    ax.set_ylabel('P(reward) at leaving')
    if ylim_right is not None:
        ax.set_ylim(ylim_right)
    sns.despine(ax=ax, bottom=True)

    plt.tight_layout()
    plt.show()
    return summary_df


def plot_patch_statistics_per_session(df, ylim_right=None):
    """Two plots per session plus one cross-session summary.

    Per-session figures: box over all individual patch visits, coloured by
    patch_reward_prob_prefactor.

    Summary figure: one line per session showing mean P(reward) at leaving
    for each patch type.

    Returns (figs, fig_summary) where figs is a list of (session_number, fig).
    """
    patch_types = sorted(df['patch_type'].unique())
    n_types     = len(patch_types)
    has_prefactor = 'patch_reward_prob_prefactor' in df.columns

    tau_for = df.groupby('patch_type')['patch_reward_param'].mean().to_dict()
    cmap    = mpl.colormaps['magma']

    box_base = dict(
        width=0.4, showcaps=True,
        boxprops={'edgecolor': 'black', 'linewidth': 1.2},
        medianprops={'color': 'black', 'linewidth': 1.5},
        whiskerprops={'color': 'black', 'linewidth': 1.0},
        capprops={'color': 'black', 'linewidth': 1.0},
        flierprops={'marker': 'o', 'markerfacecolor': 'white',
                    'markeredgecolor': 'black', 'markersize': 4, 'alpha': 1.0},
    )

    figs = []
    session_means = []  # collect (session_num, {patch_type: mean_reward_prob})

    for session_num in sorted(df['session_number'].unique()):
        sess_df = df[df['session_number'] == session_num]

        # Per-session prefactor per patch type
        if has_prefactor:
            prefactor_for = sess_df.groupby('patch_type')['patch_reward_prob_prefactor'].mean().to_dict()
        else:
            prefactor_for = {pt: 0.8 for pt in patch_types}

        prefactors = [prefactor_for.get(pt, 0.8) for pt in patch_types]
        colors = [cmap(pf) for pf in prefactors]
        palette_dict = {pt: col for pt, col in zip(patch_types, colors)}
        box_kwargs = dict(box_base, hue='patch_type', palette=palette_dict,
                          dodge=False, legend=False)

        # One row per patch visit: max rewards seen in that visit
        patch_df = (
            sess_df.groupby(['patch_number', 'patch_type'])['rewards_seen_in_patch']
            .max().reset_index()
            .rename(columns={'rewards_seen_in_patch': 'rewards_collected'})
        )
        patch_df['tau']       = patch_df['patch_type'].map(tau_for)
        patch_df['prefactor'] = patch_df['patch_type'].map(prefactor_for)
        patch_df['reward_prob'] = np.where(
            patch_df['tau'] > 0,
            patch_df['prefactor'] * np.exp(-patch_df['rewards_collected'] / patch_df['tau']),
            0.0,
        )

        # Store per-patch-type means for the summary plot
        means = patch_df.groupby('patch_type')['reward_prob'].mean().to_dict()
        means_rewards = patch_df.groupby('patch_type')['rewards_collected'].mean().to_dict()
        session_means.append((session_num, means, prefactor_for, means_rewards))

        fig, axes = plt.subplots(1, 2, figsize=(n_types * 1.2 + 1.2, 3))

        ax = axes[0]
        sns.boxplot(data=patch_df, x='patch_type', y='rewards_collected',
                    order=patch_types, ax=ax, **box_kwargs)
        pf_str = ', '.join(f'{prefactor_for.get(pt, 0):.2f}' for pt in patch_types)
        ax.set_title(f'Session {session_num}  [{pf_str}]', fontsize=8)
        ax.set_xlabel('')
        ax.tick_params(axis='x', bottom=False, labelbottom=False)
        ax.set_ylabel('Rewards collected')
        sns.despine(ax=ax, bottom=True)
        format_plot(ax)

        ax = axes[1]
        sns.boxplot(data=patch_df, x='patch_type', y='reward_prob',
                    order=patch_types, ax=ax, **box_kwargs)
        ax.set_xlabel('')
        ax.tick_params(axis='x', bottom=False, labelbottom=False)
        ax.set_ylabel('P(reward) at leaving')
        if ylim_right is not None:
            ax.set_ylim(ylim_right)
        sns.despine(ax=ax, bottom=True)
        format_plot(ax)

        fig.tight_layout()
        figs.append((session_num, fig))

    # --- Summary: one line per session, P(reward) at leaving ---
    fig_summary, ax_sum = plt.subplots(figsize=(n_types * 1.2 + 1.2, 3))
    for _, means, prefactor_for, _ in session_means:
        pts = sorted(
            [(prefactor_for.get(pt, np.nan), means.get(pt, np.nan)) for pt in patch_types],
            key=lambda p: p[0]
        )
        x, y = zip(*pts)
        ax_sum.plot(x, y, marker='o', markersize=4, linewidth=1.2,
                    color='steelblue', alpha=0.3)

    ax_sum.set_xlabel('Patch reward prob prefactor', fontsize=8)
    ax_sum.set_ylabel('Mean P(reward) at leaving', fontsize=8)
    if ylim_right is not None:
        ax_sum.set_ylim(ylim_right)
    sns.despine(ax=ax_sum)
    format_plot(ax_sum)
    fig_summary.tight_layout()

    # --- Summary: one line per session, rewards collected ---
    fig_summary_rewards, ax_rew = plt.subplots(figsize=(n_types * 1.2 + 1.2, 3))
    for _, _, prefactor_for, means_rewards in session_means:
        pts = sorted(
            [(prefactor_for.get(pt, np.nan), means_rewards.get(pt, np.nan)) for pt in patch_types],
            key=lambda p: p[0]
        )
        x, y = zip(*pts)
        ax_rew.plot(x, y, marker='o', markersize=4, linewidth=1.2,
                    color='steelblue', alpha=0.3)

    ax_rew.set_xlabel('Patch reward prob prefactor', fontsize=8)
    ax_rew.set_ylabel('Rewards collected', fontsize=8)
    sns.despine(ax=ax_rew)
    format_plot(ax_rew)
    fig_summary_rewards.tight_layout()

    return figs, fig_summary, fig_summary_rewards


def plot_patch_statistics_across_dfs(dfs, figsize=(5, 3), xticks=None, xticklabels=None):
    """
    Aggregate statistics across a list of dfs (e.g. different checkpoints).

    Produces two figures:
      1. Mean rewards collected per patch type vs df index.
      2. Mean P(reward) at leaving per patch type vs df index.

    Each df is reduced to a single point per patch type (mean over all patches
    and sessions in that df).  The three patch types are shown as separate
    coloured lines.
    """
    type_colors = {0: '#7570b3', 1: '#2ca02c', 2: '#d95f02'}

    records = []
    for df_idx, df in enumerate(dfs):
        patch_types = sorted(df['patch_type'].unique())
        tau_for = df.groupby('patch_type')['patch_reward_param'].mean().to_dict()
        has_prefactor = 'patch_reward_prob_prefactor' in df.columns

        patch_df = (
            df.groupby(['session_number', 'patch_number', 'patch_type'])['rewards_seen_in_patch']
            .max().reset_index()
            .rename(columns={'rewards_seen_in_patch': 'rewards_collected'})
        )

        if has_prefactor:
            prefactor_for = df.groupby('patch_type')['patch_reward_prob_prefactor'].mean().to_dict()
        else:
            prefactor_for = {pt: 0.8 for pt in patch_types}

        patch_df['tau']       = patch_df['patch_type'].map(tau_for)
        patch_df['prefactor'] = patch_df['patch_type'].map(prefactor_for)
        patch_df['reward_prob'] = np.where(
            patch_df['tau'] > 0,
            patch_df['prefactor'] * np.exp(-patch_df['rewards_collected'] / patch_df['tau']),
            0.0,
        )

        for pt in patch_types:
            pt_df = patch_df[patch_df['patch_type'] == pt]
            records.append({
                'df_idx':           df_idx,
                'patch_type':       pt,
                'mean_rewards':     pt_df['rewards_collected'].mean(),
                'sem_rewards':      pt_df['rewards_collected'].sem(),
                'mean_reward_prob': pt_df['reward_prob'].mean(),
                'sem_reward_prob':  pt_df['reward_prob'].sem(),
            })

    agg = pd.DataFrame(records)
    patch_types = sorted(agg['patch_type'].unique())

    fig1, ax1 = plt.subplots(figsize=figsize)
    fig2, ax2 = plt.subplots(figsize=figsize)

    for pt in patch_types:
        pt_df = agg[agg['patch_type'] == pt].sort_values('df_idx')
        color = type_colors.get(pt, 'steelblue')
        ax1.errorbar(pt_df['df_idx'], pt_df['mean_rewards'], yerr=pt_df['sem_rewards'],
                     color=color, lw=1.2, marker='o', markersize=3,
                     capsize=2, capthick=0.8, elinewidth=0.8, label=f'Type {pt}')
        ax2.errorbar(pt_df['df_idx'], pt_df['mean_reward_prob'], yerr=pt_df['sem_reward_prob'],
                     color=color, lw=1.2, marker='o', markersize=3,
                     capsize=2, capthick=0.8, elinewidth=0.8, label=f'Type {pt}')

    for ax in (ax1, ax2):
        if xticks is not None:
            ax.set_xticks(xticks)
        if xticklabels is not None:
            ax.set_xticklabels(xticklabels, fontsize=7)

    ax1.set_xlabel('Network index', fontsize=8)
    ax1.set_ylabel('Mean rewards collected', fontsize=8)
    ax1.legend(fontsize=7, frameon=False)
    sns.despine(ax=ax1)
    format_plot(ax1)
    fig1.tight_layout()

    ax2.set_xlabel('Network index', fontsize=8)
    ax2.set_ylabel('P(reward) at leaving', fontsize=8)
    ax2.legend(fontsize=7, frameon=False)
    sns.despine(ax=ax2)
    format_plot(ax2)
    fig2.tight_layout()

    return fig1, ax1, fig2, ax2


def plot_first_patch_rewards_vs_other_prefactors(df, first_patch_type=None, figsize=(4, 3.5)):
    """
    For each session compute the mean rewards collected in `first_patch_type`
    (defaults to the lowest patch-type index), then plot this as a function of
    the prefactors of the other two patch types.

    Each point is one session (or the mean over sessions sharing the same
    prefactor pair).  x = prefactor of the second patch type, y = prefactor of
    the third, colour = mean rewards in the first patch type.
    """
    if 'patch_reward_prob_prefactor' not in df.columns:
        print('patch_reward_prob_prefactor not in df')
        return None

    patch_types = sorted(df['patch_type'].unique())
    if first_patch_type is None:
        first_patch_type = patch_types[0]
    other_types = [pt for pt in patch_types if pt != first_patch_type]
    if len(other_types) != 2:
        print(f'Expected exactly 2 other patch types, got {len(other_types)}')
        return None
    pt1, pt2 = other_types

    rows = []
    for session_num in sorted(df['session_number'].unique()):
        sess_df = df[df['session_number'] == session_num]

        pf_map = sess_df.groupby('patch_type')['patch_reward_prob_prefactor'].mean().to_dict()
        pf1 = pf_map.get(pt1, np.nan)
        pf2 = pf_map.get(pt2, np.nan)

        first_df = sess_df[sess_df['patch_type'] == first_patch_type]
        if first_df.empty:
            continue
        mean_rewards = first_df.groupby('patch_number')['rewards_seen_in_patch'].max().mean()
        rows.append({'pf1': pf1, 'pf2': pf2, 'mean_rewards': mean_rewards})

    if not rows:
        print('No data found')
        return None

    plot_df = pd.DataFrame(rows)
    plot_df = plot_df.groupby(['pf1', 'pf2'], as_index=False)['mean_rewards'].mean()

    # Try heatmap if data sits on a regular grid, otherwise scatter
    pf1_vals = sorted(plot_df['pf1'].unique())
    pf2_vals = sorted(plot_df['pf2'].unique())
    is_grid  = len(plot_df) == len(pf1_vals) * len(pf2_vals)

    fig, ax = plt.subplots(figsize=figsize)

    if is_grid and len(pf1_vals) > 1 and len(pf2_vals) > 1:
        grid = (plot_df
                .pivot(index='pf2', columns='pf1', values='mean_rewards')
                .sort_index(ascending=False))
        im = ax.imshow(grid.values, aspect='auto',
                       extent=[min(pf1_vals) - 0.005, max(pf1_vals) + 0.005,
                                min(pf2_vals) - 0.005, max(pf2_vals) + 0.005],
                       origin='lower',
                       cmap=sns.color_palette('Oranges_d', as_cmap=True))
        cb = fig.colorbar(im, ax=ax)
    else:
        sc = ax.scatter(plot_df['pf1'], plot_df['pf2'],
                        c=plot_df['mean_rewards'],
                        cmap=sns.color_palette('Oranges_d', as_cmap=True),
                        s=20, edgecolors='none')
        cb = fig.colorbar(sc, ax=ax)

    cb.set_label(f'Mean rewards — patch type {first_patch_type}', fontsize=7)
    cb.ax.tick_params(labelsize=7)
    ax.set_xlabel(f'Prefactor (patch type {pt1})', fontsize=8)
    ax.set_ylabel(f'Prefactor (patch type {pt2})', fontsize=8)
    format_plot(ax)
    fig.tight_layout()
    return fig


def plot_rewards_by_patch_sequence(df, figsize=(5, 3), by_session=False, n_cols=1,
                                   y_col='rewards_seen_in_patch', y_label=None):
    """
    For each patch type, plots a per-patch statistic as a function of within-session
    patch visit number (x-axis resets at the start of each session).

    y_col: column to aggregate with max per patch (default 'rewards_seen_in_patch').
    by_session=False: one panel, mean ± SEM across sessions.
    by_session=True: one subplot per session, rows=sessions, cols=patch types.
    """
    if y_label is None:
        y_label = y_col.replace('_', ' ')

    patch_types = sorted(df['patch_type'].unique())
    type_colors = {0: '#7570b3', 1: '#2ca02c', 2: '#d95f02'}
    sessions = sorted(df['session_number'].unique())

    records = []
    for session_num in sessions:
        sess_df = df[df['session_number'] == session_num]
        patch_df = (
            sess_df.groupby(['patch_number', 'patch_type'])[y_col]
            .max().reset_index()
            .rename(columns={y_col: 'y_val'})
        )
        patch_df = patch_df.sort_values('patch_number').reset_index(drop=True)
        patch_df['visit_idx'] = np.arange(len(patch_df))
        patch_df['session_number'] = session_num
        records.append(patch_df)

    all_df = pd.concat(records, ignore_index=True)

    if not by_session:
        fig, ax = plt.subplots(figsize=figsize)
        for pt in patch_types:
            pt_df = all_df[all_df['patch_type'] == pt]
            stats = pt_df.groupby('visit_idx')['y_val'].agg(['mean', 'sem']).reset_index()
            color = type_colors.get(pt, 'steelblue')
            ax.plot(stats['visit_idx'], stats['mean'], color=color, lw=1.2, label=f'Patch type {pt}')
            ax.fill_between(stats['visit_idx'],
                            stats['mean'] - stats['sem'],
                            stats['mean'] + stats['sem'],
                            color=color, alpha=0.2)
        ax.set_xlabel('Patch visit number (within session)', fontsize=8)
        ax.set_ylabel(f'Mean {y_label}', fontsize=8)
        ax.legend(fontsize=7, frameon=False)
        sns.despine(ax=ax)
        format_plot(ax)
        fig.tight_layout()
        return fig

    # --- per-session grid: rows=sessions, cols=patch types ---
    n_pt = len(patch_types)
    n_rows = len(sessions)
    fig, axes = plt.subplots(n_rows, n_pt,
                             figsize=(figsize[0] * n_pt, figsize[1] * n_rows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    for row, session_num in enumerate(sessions):
        sess_df = all_df[all_df['session_number'] == session_num]
        for col, pt in enumerate(patch_types):
            ax = axes[row, col]
            pt_df = sess_df[sess_df['patch_type'] == pt]
            color = type_colors.get(pt, 'steelblue')
            if not pt_df.empty:
                ax.plot(pt_df['visit_idx'], pt_df['y_val'], color=color, lw=1.0)
                ax.scatter(pt_df['visit_idx'], pt_df['y_val'],
                           color=color, s=10, zorder=3)
            if row == 0:
                ax.set_title(f'Patch type {pt}', fontsize=7)
            if col == 0:
                ax.set_ylabel(y_label, fontsize=7)
                ax.annotate(f'Session {session_num}', xy=(0, 0.5),
                            xycoords='axes fraction', xytext=(-0.45, 0.5),
                            textcoords='axes fraction', fontsize=6,
                            ha='center', va='center', rotation=90,
                            annotation_clip=False)
            sns.despine(ax=ax)
            format_plot(ax)

    fig.supxlabel('Patch visit number (within session)', fontsize=8)
    fig.tight_layout()
    return fig


def plot_patch_statistics(df):
    """Per-network version: groups by network_num × tau."""
    taus = sorted(df['patch_reward_param'].unique(), reverse=True)[:2]
    session_data = []

    for nn_num in sorted(df['network_num'].unique()):
        for tau in taus:
            if tau != 0:
                crit = (
                    (df['network_num'] == nn_num)
                    & (df['rewarded'] == 1)
                    & (df['patch_reward_param'] == tau)
                )
                mean_count = (df[crit]
                              .groupby(['patch_number', 'session_number'])
                              .count()['rewarded'].mean())
                session_data.append({
                    'network_num': nn_num,
                    'tau': tau,
                    'rewards_collected': mean_count,
                    'reward_prob': 0.8 * np.exp(-mean_count / tau),
                })

    summary_df = pd.DataFrame(session_data)
    fig, axes = plt.subplots(1, 2, figsize=(5, 3), sharey=False)
    colors = ['#d95f02', '#1b9e77']

    ax = axes[0]
    sns.boxplot(data=summary_df, x='tau', y='rewards_collected', order=taus,
                width=0.4, palette=colors, ax=ax, showcaps=True,
                boxprops={'edgecolor': 'black', 'linewidth': 1.2},
                medianprops={'color': 'black', 'linewidth': 1.5},
                whiskerprops={'color': 'black', 'linewidth': 1.0},
                capprops={'color': 'black', 'linewidth': 1.0},
                flierprops={'marker': 'o', 'markerfacecolor': 'white',
                            'markeredgecolor': 'black', 'markersize': 6, 'alpha': 1.0})
    for _, group in summary_df.groupby('network_num'):
        ax.plot([0, 1], group['rewards_collected'].values,
                color='grey', alpha=1, zorder=-1, lw=0.5)
    if len(taus) == 2:
        high = summary_df[summary_df['tau'] == taus[0]]['rewards_collected']
        low  = summary_df[summary_df['tau'] == taus[1]]['rewards_collected']
        _, pval = stats.ttest_rel(high, low)
        y_max = summary_df['rewards_collected'].max() + 1
        ax.plot([0, 1], [y_max, y_max], color='black')
        stars = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
        ax.text(0.5, y_max + 0.3, stars, ha='center', va='bottom', fontsize=12)
    ax.set_xticklabels(['High', 'Low'])
    ax.set_xlabel('')
    ax.set_ylabel('Rewards collected')
    sns.despine(ax=ax)
    format_plot(ax)

    ax = axes[1]
    sns.boxplot(data=summary_df, x='tau', y='reward_prob', order=taus,
                width=0.4, palette=colors, ax=ax, showcaps=True,
                boxprops={'edgecolor': 'black', 'linewidth': 1.2},
                medianprops={'color': 'black', 'linewidth': 1.5},
                whiskerprops={'color': 'black', 'linewidth': 1.0},
                capprops={'color': 'black', 'linewidth': 1.0},
                flierprops={'marker': 'o', 'markerfacecolor': 'white',
                            'markeredgecolor': 'black', 'markersize': 6, 'alpha': 1.0})
    for _, group in summary_df.groupby('network_num'):
        ax.plot([0, 1], group['reward_prob'].values,
                color='grey', alpha=1, zorder=-1, lw=0.5)
    if len(taus) == 2:
        high = summary_df[summary_df['tau'] == taus[0]]['reward_prob']
        low  = summary_df[summary_df['tau'] == taus[1]]['reward_prob']
        _, pval = stats.ttest_rel(high, low)
        y_max = summary_df['reward_prob'].max() + 0.05
        ax.plot([0, 1], [y_max, y_max], color='black')
        stars = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
        ax.text(0.5, y_max + 0.02, stars, ha='center', va='bottom', fontsize=12)
    ax.set_xticklabels(['High', 'Low'])
    ax.set_xlabel('')
    ax.set_ylabel('P(reward) at leaving')
    sns.despine(ax=ax)
    format_plot(ax)
    plt.tight_layout()
    return fig


def plot_multi_df_accuracy_heatmap(odor_site_dfs, predictors, df_labels=None, figsize=(5, 4)):
    """Heatmap of cross-validated logistic regression accuracy for each predictor × dataset."""
    if df_labels is None:
        df_labels = [f'DF_{i}' for i in range(len(odor_site_dfs))]

    accuracy_matrix = np.zeros((len(odor_site_dfs), len(predictors)))
    predictor_names = [p if isinstance(p, str) else ' + '.join(p) for p in predictors]

    for df_idx, odor_site_df in enumerate(odor_site_dfs):
        print(f'Processing {df_labels[df_idx]}...')
        for pred_idx, predictor in enumerate(predictors):
            predictor_cols = [predictor] if isinstance(predictor, str) else predictor
            X = odor_site_df[predictor_cols].values
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            y = odor_site_df['stopped'].values.astype(int)
            model = LogisticRegression()
            rus = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = rus.fit_resample(X, y)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_resampled, y_resampled, cv=skf, scoring='accuracy')
            accuracy_matrix[df_idx, pred_idx] = np.mean(cv_scores)

    plt.figure(figsize=figsize)
    sns.heatmap(accuracy_matrix,
                xticklabels=[n.replace('_', ' ') for n in predictor_names],
                yticklabels=df_labels,
                annot=False, fmt='.3f', cmap='Blues', vmin=0.5, vmax=1.0,
                cbar_kws={'label': 'Average CV Accuracy'},
                square=False, linewidths=0.5, linecolor='white')
    plt.xlabel('Predictors', fontsize=12)
    plt.ylabel('Type', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_leave_probabilities(odor_site_dfs, predictors, dataset_labels=None,
                             x_labels=None):
    """Leave (stop) probability vs. each predictor for multiple datasets.

    x_labels : optional list of x-axis label strings, one per predictor.
                Falls back to the predictor name (underscores replaced) if None.
    """
    scale = 0.5
    if dataset_labels is None:
        dataset_labels = [f'Dataset {i+1}' for i in range(len(odor_site_dfs))]
    if x_labels is None:
        x_labels = [p.replace('_', ' ') for p in predictors]

    for predictor, x_label in zip(predictors, x_labels):
        fig, ax = plt.subplots(1, 1, figsize=(scale * 7, scale * 4), constrained_layout=True)
        for df, label in zip(odor_site_dfs, dataset_labels):
            agg  = df.groupby(predictor)['stopped'].agg(['sum', 'count'])
            k, n = agg['sum'].values, agg['count'].values
            p    = k / n
            ci   = clopper_pearson(k, n)
            yerr = np.array([p - ci[0], ci[1] - p])
            x    = agg.index.values
            ax.errorbar(x, p, yerr=yerr, label=label,
                        fmt='o-', ms=3, lw=0.8, elinewidth=0.8, capsize=2)
        ax.set_ylabel('Stay probability', fontsize=11)
        ax.set_xlabel(x_label, fontsize=11)
        format_plot(ax)
    fig.tight_layout()
    plt.show()


def plot_rewards_vs_interpatch_distance(odor_site_dfs, dataset_labels=None):
    """Mean rewards_seen_in_patch vs interpatch_distance for non-stopped sites.

    Produces two figures:
    1. Mean ± SEM at each unique interpatch_distance, one line per network.
    2. Same metric split into two bins relative to the global median interpatch
       distance (< median vs >= median), one line per network.
    """
    if dataset_labels is None:
        dataset_labels = [f'Dataset {i+1}' for i in range(len(odor_site_dfs))]

    all_subs = [df[df['stopped'] == 0] for df in odor_site_dfs]
    global_median = pd.concat(all_subs)['interpatch_distance'].median()

    scale = 0.5

    # --- Plot 1: continuous interpatch distance ---
    fig, ax = plt.subplots(figsize=(scale * 7, scale * 4), constrained_layout=True)
    for sub, label in zip(all_subs, dataset_labels):
        agg  = sub.groupby('interpatch_distance')['rewards_seen_in_patch'].agg(['mean', 'sem'])
        ax.errorbar(agg.index.values, agg['mean'].values, yerr=agg['sem'].values,
                    label=label, fmt='o-', ms=3, lw=0.8, elinewidth=0.8, capsize=2)
    ax.set_xlabel('Interpatch distance', fontsize=11)
    ax.set_ylabel('Rewards collected', fontsize=11)
    format_plot(ax)
    plt.show()

    # --- Plot 2: split at global median ---
    bin_labels = ['IPI < median', r'IPI $\geq$ median']
    fig2, ax2 = plt.subplots(figsize=(scale * 5, scale * 4), constrained_layout=True)
    for sub, label in zip(all_subs, dataset_labels):
        below = sub[sub['interpatch_distance'] <  global_median]['rewards_seen_in_patch']
        above = sub[sub['interpatch_distance'] >= global_median]['rewards_seen_in_patch']
        means = [below.mean(), above.mean()]
        sems  = [below.sem(),  above.sem()]
        ax2.errorbar([0, 1], means, yerr=sems, label=label,
                     fmt='o-', ms=3, lw=0.8, elinewidth=0.8, capsize=2)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(bin_labels, fontsize=9)
    ax2.set_ylabel('Rewards collected', fontsize=11)
    format_plot(ax2)
    plt.show()


def append_index_phase(df):
    df['site_number_norm_patches'] = (
        df.groupby(['network_num', 'session_number', 'patch_number'])['index']
        .transform(lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0)
    )
    df['site_number_norm_patches'] = df['site_number_norm_patches'].round(2)
    df['segment'] = np.where(df['site_number_norm_patches'] < 0.5, 'early', 'late')
    return df


def plot_grouped_mean_std(df, x_col, hue_col, feature, dataset_col='dataset'):
    """Mean ± std of feature across datasets, grouped by x_col and hue_col."""
    grouped = (
        df.groupby([dataset_col, x_col, hue_col])[feature]
          .mean()
          .transform(lambda x: 1 - x)
          .reset_index()
    )
    summary = (
        grouped.groupby([x_col, hue_col])[feature]
               .agg(['mean', 'std'])
               .reset_index()
    )
    fig, ax = plt.subplots(figsize=(3, 3))
    colors = {'early': '#909090', 'late': '#000000'}
    for hue, sub in summary.groupby(hue_col):
        x_vals = sub[x_col].values
        y_mean = sub['mean'].values
        y_std  = sub['std'].values
        color  = colors.get(hue, None)
        ax.plot(x_vals, y_mean, label=hue, color=color, linewidth=1.5)
        ax.fill_between(x_vals, y_mean - y_std, y_mean + y_std, alpha=0.2, color=color)
    ax.set_xlabel('Intersite interval', fontsize=11)
    ax.set_ylabel('P(stop)', fontsize=11)
    ax.set_ylim(-0.05, 0.5)
    ax.legend(title='Patch depth', loc='upper right', frameon=True)
    format_plot(ax)
    plt.tight_layout()
    plt.show()
    return fig


def plot_stop_fraction(df, x_col, y_col, condition=None, invert_y=True,
                       figsize=(5, 5), reverse=False, label=None, cmap='YlGnBu',
                       xlabel=None, ylabel=None, high_res_save=False, min_n=1):
    """Heatmap of P(stop) over two columns in df."""
    if condition is not None:
        df = condition(df)

    grouped = (
        df.groupby([y_col, x_col])
          .filter(lambda g: len(g) >= min_n)
          .groupby([y_col, x_col])['stopped']
          .mean()
          .reset_index()
    )
    heatmap_data = grouped.pivot(index=y_col, columns=x_col, values='stopped')

    fig, ax = plt.subplots(figsize=figsize, dpi=300 if high_res_save else 100)
    if label is None:
        label = 'P(leave)'
    sns.heatmap(1 - heatmap_data if reverse else heatmap_data,
                cmap=cmap, vmin=0, vmax=1,
                cbar_kws={'label': label}, ax=ax)

    ax.set_xlabel(xlabel if xlabel is not None else x_col.replace('_', ' '))
    ax.set_ylabel(ylabel if ylabel is not None else y_col.replace('_', ' '))
    if invert_y:
        ax.invert_yaxis()

    format_plot(ax, bottomspine=False, leftspine=False, ticklabelsize=14)
    cbar = ax.collections[0].colorbar
    format_plot(cbar.ax, bottomspine=False, leftspine=False, ticklabelsize=14)
    plt.show()
    matrix = 1 - heatmap_data if reverse else heatmap_data
    return fig, matrix
