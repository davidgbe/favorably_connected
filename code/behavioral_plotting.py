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
                 max_reward_sites=8, use_fixed_colors=True, save_dir=None):

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    b_data_raw = load_trajectory_data(data_path)
    b_data = parse_behavioral_data(b_data_raw[session_idx])
    ss = get_session_summaries([b_data])[0]

    rewards_at_positions = b_data['rewards_at_positions']
    reward_attempted_at_positions = b_data['reward_attempted_at_positions']
    all_patch_nums = b_data['current_patch_num']
    all_patch_reward_params = b_data['patch_reward_param']

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
                c = (cmap(all_patch_reward_params[i] / max_reward_param)
                     if all_patch_reward_params[i] > 0 else 'black')
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
    non_fixed_colors = [
        (cmap(rp / max_reward_param) if rp > 0 else 'black')
        for rp in ss['reward_param_for_patch_type']
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
    dfs['network_num'] = nn_num
    return dfs


def plot_patch_statistics_by_session(df, max_reward_param=40, fixed_colors=False):
    """Per-session version: groups by session_number × tau."""
    taus = sorted(df['patch_reward_param'].unique(), reverse=True)
    session_data = []

    for session_num in sorted(df['session_number'].unique()):
        for tau in taus:
            crit = (
                (df['session_number'] == session_num)
                & (df['patch_reward_param'] == tau)
            )
            mean_count = df[crit].groupby('patch_number')['rewards_seen_in_patch'].max().mean()
            session_data.append({
                'session_number': session_num,
                'tau': tau,
                'rewards_collected': mean_count,
                'reward_prob': 0.8 * np.exp(-mean_count / tau) if tau > 0 else 0,
            })

    summary_df = pd.DataFrame(session_data)
    fig, axes = plt.subplots(1, 2, figsize=(5, 3), sharey=False)
    cmap = mpl.colormaps['magma']
    colors = [(cmap(tau / max_reward_param) if tau > 0 else 'black') for tau in taus]
    if fixed_colors:
        colors = [c for c in reversed(odor_colors)]

    ax = axes[0]
    sns.boxplot(data=summary_df, x='tau', y='rewards_collected', order=taus,
                width=0.4, palette=colors, ax=ax, showcaps=True,
                boxprops={'edgecolor': 'black', 'linewidth': 1.2},
                medianprops={'color': 'black', 'linewidth': 1.5},
                whiskerprops={'color': 'black', 'linewidth': 1.0},
                capprops={'color': 'black', 'linewidth': 1.0},
                flierprops={'marker': 'o', 'markerfacecolor': 'white',
                            'markeredgecolor': 'black', 'markersize': 6, 'alpha': 1.0})
    for session_num, group in summary_df.groupby('session_number'):
        ax.plot(np.arange(len(group['rewards_collected'].values)),
                group['rewards_collected'].values, color='grey', alpha=1, zorder=-1, lw=0.5)
    if len(taus) == 2:
        high = summary_df[summary_df['tau'] == taus[0]]['rewards_collected']
        low  = summary_df[summary_df['tau'] == taus[1]]['rewards_collected']
        _, pval = stats.ttest_rel(high, low)
        y_max = summary_df['rewards_collected'].max() + 1
        ax.plot([0, 1], [y_max, y_max], color='black')
        stars = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
        ax.text(0.5, y_max + 0.3, stars, ha='center', va='bottom', fontsize=12)
    ax.set_xlabel('')
    ax.set_ylabel('Rewards collected')
    sns.despine(ax=ax)

    ax = axes[1]
    sns.boxplot(data=summary_df, x='tau', y='reward_prob', order=taus,
                width=0.4, palette=colors, ax=ax, showcaps=True,
                boxprops={'edgecolor': 'black', 'linewidth': 1.2},
                medianprops={'color': 'black', 'linewidth': 1.5},
                whiskerprops={'color': 'black', 'linewidth': 1.0},
                capprops={'color': 'black', 'linewidth': 1.0},
                flierprops={'marker': 'o', 'markerfacecolor': 'white',
                            'markeredgecolor': 'black', 'markersize': 6, 'alpha': 1.0})
    for session_num, group in summary_df.groupby('session_number'):
        ax.plot(np.arange(len(group['reward_prob'].values)),
                group['reward_prob'].values, color='grey', alpha=1, zorder=-1, lw=0.5)
    if len(taus) == 2:
        high = summary_df[summary_df['tau'] == taus[0]]['reward_prob']
        low  = summary_df[summary_df['tau'] == taus[1]]['reward_prob']
        _, pval = stats.ttest_rel(high, low)
        y_max = summary_df['reward_prob'].max() + 0.05
        ax.plot([0, 1], [y_max, y_max], color='black')
        stars = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
        ax.text(0.5, y_max + 0.02, stars, ha='center', va='bottom', fontsize=12)
    ax.set_xlabel('')
    ax.set_ylabel('P(reward) at leaving')
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.show()
    return summary_df


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
    return fig
