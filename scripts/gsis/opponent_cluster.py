"""
GSIS Model M3 — Opponent Archetype Classifier
Clusters all 30 NBA teams into playing-style archetypes and maps
each archetype to a recommended counter-strategy for the configured team.
"""

import json, os, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from scripts.gsis.team_config import get_team, load_cache

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
CACHE = ROOT / "web" / "cache"

GOLD = "#FFC72C"
BLUE = "#1D428A"
GREEN = "#2ecc71"
RED = "#e74c3c"
ORANGE = "#e67e22"
PURPLE = "#9b59b6"
WHITE = "#e8e8e8"
COLORS = ["#FFC72C", "#1D428A", "#2ecc71", "#e74c3c", "#9b59b6", "#e67e22"]

plt.rcParams.update({
    "figure.facecolor": "#0f1923", "axes.facecolor": "#0f1923",
    "axes.edgecolor": "#2a3f52", "text.color": WHITE,
    "axes.labelcolor": WHITE, "xtick.color": "#a0a0a0",
    "ytick.color": "#a0a0a0", "grid.color": "#1e3044",
    "grid.alpha": 0.5, "font.family": "sans-serif", "font.size": 11,
    "axes.titlesize": 14, "axes.titleweight": "bold",
})

ARCHETYPE_ICONS = ["🏃", "🏰", "🎯", "💪", "⚖️", "🔥"]
ARCHETYPE_NAMES_MAP = {}  # filled dynamically


def _load(name):
    return load_cache(name)


def _rs_to_df(data, idx=0):
    rs = data.get("resultSets", data)
    if isinstance(rs, list):
        return pd.DataFrame(rs[idx]["rowSet"], columns=rs[idx]["headers"])
    raise ValueError(f"Unexpected resultSets type: {type(rs)}")


# ══════════════════════════════════════════════════════════════════
# DATA: BUILD TEAM PROFILES
# ══════════════════════════════════════════════════════════════════

def build_team_profiles():
    """
    Build a per-team feature matrix from standings data.
    Uses season-level stats available for all 30 teams.
    """
    st = _load("standings")
    teams = _rs_to_df(st)

    profiles = []
    for _, row in teams.iterrows():
        tid = row["TeamID"]
        wins = int(row["WINS"])
        losses = int(row["LOSSES"])
        gp = wins + losses
        ppg = float(row["PointsPG"])
        opp_ppg = float(row["OppPointsPG"])
        diff = float(row["DiffPointsPG"])

        # Derive pace-proxy from PPG (higher scoring ≈ faster pace)
        # Derive defensive quality from OppPPG
        profiles.append({
            "TEAM_ID": tid,
            "TEAM": f"{row['TeamCity']} {row['TeamName']}",
            "ABBREV": _infer_abbrev(row["TeamCity"], row["TeamName"]),
            "WINS": wins,
            "LOSSES": losses,
            "WIN_PCT": float(row["WinPCT"]),
            "PPG": ppg,
            "OPP_PPG": opp_ppg,
            "NET_PPG": diff,
            # Derived features
            "SCORING_VOLUME": ppg,
            "DEFENSIVE_QUALITY": -opp_ppg,  # lower OPP PPG = better defense
            "POINT_DIFF": diff,
            "CONSISTENCY": _consistency_score(row),
            "HOME_STRENGTH": _record_pct(row.get("HOME", "0-0")),
            "ROAD_STRENGTH": _record_pct(row.get("ROAD", "0-0")),
            "CLOSE_GAME_ABILITY": _record_pct(row.get("ThreePTSOrLess", "0-0")),
            "BLOWOUT_RATE": _record_pct(row.get("TenPTSOrMore", "0-0")),
        })

    return pd.DataFrame(profiles)


def _infer_abbrev(city, name):
    """Best-effort abbreviation from city+name."""
    abbrev_map = {
        "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
        "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
        "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
        "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
        "LA Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
        "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
        "New Orleans Pelicans": "NOP", "New York Knicks": "NYK",
        "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL",
        "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
        "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC",
        "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR",
        "Utah Jazz": "UTA", "Washington Wizards": "WAS",
    }
    return abbrev_map.get(f"{city} {name}", name[:3].upper())


def _record_pct(record_str):
    """Parse '18-11' record to win%."""
    if not record_str or record_str == "0-0":
        return 0.5
    try:
        parts = str(record_str).split("-")
        w, l = int(parts[0]), int(parts[1])
        return w / max(w + l, 1)
    except Exception:
        return 0.5


def _consistency_score(row):
    """Score based on how well the team performs in different situations."""
    home = _record_pct(row.get("HOME", "0-0"))
    road = _record_pct(row.get("ROAD", "0-0"))
    # Consistency = 1 - |home - road| (more consistent = higher)
    return 1 - abs(home - road)


# ══════════════════════════════════════════════════════════════════
# CLUSTERING
# ══════════════════════════════════════════════════════════════════

def cluster_teams(profiles):
    """Run K-Means clustering on team profiles. Auto-select k via silhouette."""
    feature_cols = [
        "SCORING_VOLUME", "DEFENSIVE_QUALITY", "POINT_DIFF",
        "CONSISTENCY", "HOME_STRENGTH", "ROAD_STRENGTH",
        "CLOSE_GAME_ABILITY", "BLOWOUT_RATE", "WIN_PCT",
    ]
    X = profiles[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Test k=3 through k=6
    best_k, best_sil = 4, -1
    for k in range(3, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        if sil > best_sil:
            best_k, best_sil = k, sil

    # Final clustering
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    profiles["CLUSTER"] = km.fit_predict(X_scaled)

    # PCA for visualization
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    profiles["PCA_1"] = coords[:, 0]
    profiles["PCA_2"] = coords[:, 1]

    # Name clusters based on centroid characteristics
    cluster_names = name_clusters(profiles, feature_cols)
    profiles["ARCHETYPE"] = profiles["CLUSTER"].map(cluster_names)

    return profiles, best_k, best_sil, feature_cols, km, scaler, pca


def name_clusters(profiles, feature_cols):
    """Assign descriptive archetype names based on cluster centroids."""
    names = {}
    for c in profiles["CLUSTER"].unique():
        grp = profiles[profiles["CLUSTER"] == c]
        avg_ppg = grp["PPG"].mean()
        avg_opp = grp["OPP_PPG"].mean()
        avg_diff = grp["POINT_DIFF"].mean()
        avg_wp = grp["WIN_PCT"].mean()
        avg_close = grp["CLOSE_GAME_ABILITY"].mean()

        # Classify based on relative strengths
        if avg_diff > 5:
            names[c] = f"{ARCHETYPE_ICONS[0]} Elite Powerhouse"
        elif avg_diff > 0 and avg_ppg > 115:
            names[c] = f"{ARCHETYPE_ICONS[2]} High-Octane Offense"
        elif avg_diff > 0 and avg_opp < 112:
            names[c] = f"{ARCHETYPE_ICONS[1]} Defensive Fortress"
        elif abs(avg_diff) <= 2:
            names[c] = f"{ARCHETYPE_ICONS[4]} Balanced Contender"
        elif avg_diff < -3 and avg_ppg > 112:
            names[c] = f"{ARCHETYPE_ICONS[5]} Leaky Offense"
        else:
            names[c] = f"{ARCHETYPE_ICONS[3]} Rebuilding / Developing"
    return names


# ══════════════════════════════════════════════════════════════════
# WARRIORS' RECORD VS EACH ARCHETYPE
# ══════════════════════════════════════════════════════════════════

def team_vs_archetypes(profiles):
    """Compute the configured team's record against each opponent archetype."""
    from scripts.gsis.features import _load as feat_load, _rs_to_df as feat_rs

    gl_data = feat_load("gamelog")
    gl = feat_rs(gl_data)
    gl = gl.iloc[::-1].reset_index(drop=True)

    # Map opponent abbreviations
    from scripts.gsis.features import _extract_opponent_abbrev
    gl["OPP_ABBREV"] = gl["MATCHUP"].apply(_extract_opponent_abbrev)

    abbrev_to_archetype = dict(zip(profiles["ABBREV"], profiles["ARCHETYPE"]))

    records = {}
    for _, row in gl.iterrows():
        arch = abbrev_to_archetype.get(row["OPP_ABBREV"], "Unknown")
        if arch == "Unknown":
            continue
        if arch not in records:
            records[arch] = {"W": 0, "L": 0, "PTS_FOR": [], "PTS_AGAINST": []}
        if row["WL"] == "W":
            records[arch]["W"] += 1
        else:
            records[arch]["L"] += 1
        records[arch]["PTS_FOR"].append(float(row["PTS"]))

    # Summarize
    summary = []
    for arch, rec in records.items():
        gp = rec["W"] + rec["L"]
        summary.append({
            "ARCHETYPE": arch,
            "GP": gp,
            "W": rec["W"],
            "L": rec["L"],
            "WIN_PCT": rec["W"] / max(gp, 1),
            "AVG_PTS": np.mean(rec["PTS_FOR"]) if rec["PTS_FOR"] else 0,
        })

    return pd.DataFrame(summary)


# ══════════════════════════════════════════════════════════════════
# COUNTER-STRATEGIES
# ══════════════════════════════════════════════════════════════════

def generate_counter_strategies(profiles, vs_summary):
    """Generate counter-strategy recommendations for each archetype."""
    strategies = {}
    for _, row in vs_summary.iterrows():
        arch = row["ARCHETYPE"]
        wp = row["WIN_PCT"]

        # Get archetype teams
        arch_teams = profiles[profiles["ARCHETYPE"] == arch]
        avg_ppg = arch_teams["PPG"].mean()
        avg_opp = arch_teams["OPP_PPG"].mean()

        tips = []
        if "Elite" in arch or "Fortress" in arch:
            tips = [
                "Maximize PnR to create mismatches against elite defenses",
                "Target < 12 turnovers (elite teams capitalize on mistakes)",
                "Slow the pace — limit transition opportunities",
            ]
        elif "High-Octane" in arch or "Offense" in arch:
            tips = [
                "Match their pace with uptempo play",
                "Prioritize 3PT defense (close out aggressively)",
                "Push transition offense — attack before defense sets",
            ]
        elif "Balanced" in arch:
            tips = [
                "Execute default game plan with discipline",
                "Exploit specific matchup advantages (scout individuals)",
                "Control the boards — rebounding often decides even games",
            ]
        elif "Leaky" in arch:
            tips = [
                "Attack aggressively — push pace and get to the rim",
                "Be disciplined on defense despite easy offense",
                "Use these games to build confidence and chemistry",
            ]
        else:  # Rebuilding
            tips = [
                "Rest veterans when possible (load management opportunity)",
                "Give developmental players extended minutes",
                "Maintain focus — avoid complacency against weaker teams",
            ]

        status = "✅ Strong" if wp > 0.6 else "⚠️ Neutral" if wp > 0.4 else "❌ Struggling"
        strategies[arch] = {
            "win_pct": wp,
            "gp": row["GP"],
            "record": f"{row['W']}-{row['L']}",
            "status": status,
            "tips": tips,
            "avg_opp_ppg": avg_ppg,
            "teams": arch_teams["TEAM"].tolist(),
        }
    return strategies


# ══════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════

def plot_cluster_map(profiles, img_dir):
    """PCA scatter of all 30 teams colored by archetype."""
    fig, ax = plt.subplots(figsize=(12, 8))
    archetypes = profiles["ARCHETYPE"].unique()

    for i, arch in enumerate(archetypes):
        grp = profiles[profiles["ARCHETYPE"] == arch]
        color = COLORS[i % len(COLORS)]
        ax.scatter(grp["PCA_1"], grp["PCA_2"], color=color, s=120,
                   edgecolor="white", linewidth=0.5, label=arch, alpha=0.85, zorder=5)

    # Label every team
    team = get_team()
    for _, row in profiles.iterrows():
        is_our_team = row["ABBREV"] == team
        fontweight = "bold" if is_our_team else "normal"
        fontsize = 10 if is_our_team else 8
        color = GOLD if is_our_team else WHITE
        ax.annotate(row["ABBREV"], (row["PCA_1"], row["PCA_2"]),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=fontsize, color=color, fontweight=fontweight)

    ax.set_xlabel("PCA Component 1 (Offensive Quality →)")
    ax.set_ylabel("PCA Component 2 (Defensive Quality →)")
    ax.set_title("NBA Team Archetypes — 2025-26 Season")
    ax.legend(fontsize=9, loc="upper left", framealpha=0.3)
    plt.tight_layout()
    fig.savefig(img_dir / "opponent_cluster_map.png", dpi=150)
    plt.close(fig)


def plot_archetype_radar(profiles, img_dir):
    """Radar chart comparing archetype profiles."""
    feature_labels = ["PPG", "Defense", "Net PPG", "Home", "Road", "Close Games", "Blowouts"]
    feat_cols = ["SCORING_VOLUME", "DEFENSIVE_QUALITY", "POINT_DIFF",
                 "HOME_STRENGTH", "ROAD_STRENGTH", "CLOSE_GAME_ABILITY", "BLOWOUT_RATE"]

    archetypes = profiles["ARCHETYPE"].unique()
    n_arch = len(archetypes)

    angles = np.linspace(0, 2 * np.pi, len(feature_labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_facecolor("#0f1923")

    for i, arch in enumerate(archetypes):
        grp = profiles[profiles["ARCHETYPE"] == arch]
        vals = []
        for col in feat_cols:
            v = grp[col].mean()
            # Normalize to 0–1 within the full dataset
            vmin = profiles[col].min()
            vmax = profiles[col].max()
            vals.append((v - vmin) / max(vmax - vmin, 0.01))
        vals += vals[:1]
        color = COLORS[i % len(COLORS)]
        ax.plot(angles, vals, color=color, linewidth=2, label=arch)
        ax.fill(angles, vals, color=color, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_labels, fontsize=9, color=WHITE)
    ax.set_title("Archetype Profiles", pad=20)
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    fig.savefig(img_dir / "opponent_archetype_radar.png", dpi=150)
    plt.close(fig)


def plot_team_vs_archetypes(vs_summary, img_dir):
    """Bar chart of the team's record vs each archetype."""
    vs = vs_summary.sort_values("WIN_PCT", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4, len(vs) * 0.8)))

    colors = [GREEN if wp > 0.55 else GOLD if wp > 0.45 else RED
              for wp in vs["WIN_PCT"]]
    bars = ax.barh(range(len(vs)), vs["WIN_PCT"].values * 100,
                   color=colors, edgecolor="white", linewidth=0.3, height=0.6)

    for i, (bar, row_idx) in enumerate(zip(bars, vs.index)):
        r = vs.loc[row_idx]
        ax.text(bar.get_width() + 1, i,
                f"{r['WIN_PCT']*100:.0f}% ({r['W']}-{r['L']})",
                va="center", fontsize=10, color="white", fontweight="bold")

    ax.set_yticks(range(len(vs)))
    ax.set_yticklabels(vs["ARCHETYPE"].values, fontsize=10)
    ax.set_xlabel("Win %")
    ax.set_title(f"{get_team()}'s Record vs Each Opponent Archetype")
    ax.axvline(50, color="white", linestyle="--", alpha=0.3)
    ax.set_xlim(0, 110)
    plt.tight_layout()
    fig.savefig(img_dir / "team_vs_archetypes.png", dpi=150)
    plt.close(fig)


def plot_remaining_schedule(profiles, img_dir):
    """Visualize the remaining schedule by opponent archetype (if data available)."""
    # We'll show the current standings breakdown instead
    fig, ax = plt.subplots(figsize=(10, 6))
    arch_counts = profiles["ARCHETYPE"].value_counts()
    colors_list = [COLORS[i % len(COLORS)] for i in range(len(arch_counts))]

    wedges, texts, autotexts = ax.pie(
        arch_counts.values, labels=None, autopct="%1.0f%%",
        colors=colors_list, startangle=90,
        textprops={"color": "white", "fontsize": 11}
    )
    ax.legend(arch_counts.index, loc="center left", bbox_to_anchor=(1, 0.5),
              fontsize=9, framealpha=0.3)
    ax.set_title("NBA Team Distribution by Archetype")
    plt.tight_layout()
    fig.savefig(img_dir / "archetype_distribution.png", dpi=150)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ══════════════════════════════════════════════════════════════════

def generate_report(profiles, k, sil, vs_summary, strategies, img_dir, report_path):
    """Generate the opponent scouting report."""
    md = []
    p = md.append

    p("# Opponent Archetype Scouting Report")
    p("")
    p(f"*Generated: {datetime.now().strftime('%B %d, %Y')} | Model: K-Means Clustering (k={k}, silhouette={sil:.3f})*")
    p("")
    team = get_team()
    p("This report classifies all 30 NBA teams into playing-style archetypes using unsupervised")
    p(f"machine learning, then maps each archetype to a {team} counter-strategy based on historical")
    p("performance data.")
    p("")
    p("---")
    p("")

    # ── Cluster Map ──
    p("## 1. NBA Team Landscape")
    p("")
    p("**What is this chart?** Every NBA team is plotted on a 2D map using PCA (Principal Component Analysis)")
    p("to compress 9 team-level features into two dimensions. Teams that are close together play similar styles.")
    p(f"Colors represent the {k} discovered archetypes. {get_team()} is highlighted in gold.")
    p("")
    p("**How to read it:** Teams in the same cluster will require similar game plans. The X-axis roughly")
    p("captures offensive quality (right = better offense), while the Y-axis captures defensive quality")
    p("(up = better defense). Teams in the upper-right are the elite two-way teams.")
    p("")
    p("![Team Landscape](figures/opponent_cluster_map.png)")
    p("")

    # ── Archetype Profiles ──
    p("## 2. Archetype Profiles")
    p("")
    p("**What is this chart?** A radar chart comparing the average profile of each archetype across 7")
    p("dimensions: scoring volume, defensive quality, net PPG, home strength, road strength, close-game")
    p("ability, and blowout rate. Larger shapes indicate stronger overall teams.")
    p("")
    p("![Archetype Radar](figures/opponent_archetype_radar.png)")
    p("")

    # Team listing per archetype
    for arch in sorted(profiles["ARCHETYPE"].unique()):
        grp = profiles[profiles["ARCHETYPE"] == arch].sort_values("WIN_PCT", ascending=False)
        p(f"### {arch}")
        p("")
        p(f"**Teams ({len(grp)}):** {', '.join(grp['TEAM'].tolist())}")
        p("")
        p(f"Avg: {grp['PPG'].mean():.1f} PPG | {grp['OPP_PPG'].mean():.1f} OPP PPG | "
          f"Net: {grp['NET_PPG'].mean():+.1f} | Win%: {grp['WIN_PCT'].mean():.1%}")
        p("")

    p("---")
    p("")

    # ── Team vs Archetypes ──
    p(f"## 3. {team}'s Performance vs Each Archetype")
    p("")
    p(f"**How to read this chart:** Each bar shows {team}'s win percentage against teams in that")
    p("archetype. Green = above .500 (winning matchup), red = below .500 (losing matchup),")
    p("gold = roughly even. The record (W-L) is shown alongside each bar.")
    p("")
    p("![Team vs Archetypes](figures/team_vs_archetypes.png)")
    p("")

    # ── Counter-Strategies ──
    p("## 4. Counter-Strategy Playbook")
    p("")
    p(f"For each archetype, here are data-driven tactical recommendations based on {team}'s")
    p("historical performance:")
    p("")

    for arch, strat in sorted(strategies.items(), key=lambda x: -x[1]["win_pct"]):
        p(f"### {arch}")
        p("")
        p(f"**Record:** {strat['record']} ({strat['win_pct']:.0%}) — {strat['status']}")
        p("")
        p("**Counter-Strategy:**")
        for i, tip in enumerate(strat["tips"], 1):
            p(f"{i}. {tip}")
        p("")
        teams_str = ", ".join(strat["teams"][:6])
        if len(strat["teams"]) > 6:
            teams_str += f", +{len(strat['teams'])-6} more"
        p(f"*Teams: {teams_str}*")
        p("")

    # ── Distribution ──
    p("## 5. League Archetype Distribution")
    p("")
    p("![Distribution](figures/archetype_distribution.png)")
    p("")

    # ── Team Scouting Card ──
    our = profiles[profiles["ABBREV"] == team]
    if len(our) > 0:
        our_row = our.iloc[0]
        p(f"## 6. {team}'s Own Archetype")
        p("")
        p(f"{team} is classified as: **{our_row['ARCHETYPE']}**")
        p("")
        p(f"- Record: {our_row['WINS']}-{our_row['LOSSES']} ({our_row['WIN_PCT']:.1%})")
        p(f"- PPG: {our_row['PPG']:.1f} | OPP PPG: {our_row['OPP_PPG']:.1f} | Net: {our_row['NET_PPG']:+.1f}")
        p("")
        p(f"Teams in the same archetype as {team}:")
        same = profiles[profiles["ARCHETYPE"] == our_row["ARCHETYPE"]]
        same = same[same["ABBREV"] != team].sort_values("WIN_PCT", ascending=False)
        for _, t in same.iterrows():
            p(f"- {t['TEAM']} ({t['WINS']}-{t['LOSSES']}, {t['WIN_PCT']:.1%})")
        p("")

    p("---")
    p(f"*Generated: {datetime.now().strftime('%B %d, %Y')} | Data: stats.nba.com 2025-26*")

    report_path.write_text("\n".join(md))
    print(f"  📄 Report: {report_path}")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def run(img_dir=None, report_path=None):
    """Full pipeline."""
    if img_dir is None:
        img_dir = ROOT / "reports" / "game_briefs" / "figures"
    if report_path is None:
        report_path = ROOT / "reports" / "game_briefs" / "opponent_scouting.md"

    img_dir = Path(img_dir)
    report_path = Path(report_path)
    os.makedirs(img_dir, exist_ok=True)

    print("M3 — Opponent Archetype Classifier")
    print("  Building team profiles …")
    profiles = build_team_profiles()
    print(f"  {len(profiles)} teams profiled")

    print("  Clustering …")
    profiles, k, sil, feat_cols, km, scaler, pca = cluster_teams(profiles)
    print(f"  k={k}, silhouette={sil:.3f}")

    print(f"  Analyzing {get_team()} vs archetypes …")
    vs_summary = team_vs_archetypes(profiles)
    strategies = generate_counter_strategies(profiles, vs_summary)

    print("  Generating visualizations …")
    plot_cluster_map(profiles, img_dir)
    plot_archetype_radar(profiles, img_dir)
    plot_team_vs_archetypes(vs_summary, img_dir)
    plot_remaining_schedule(profiles, img_dir)

    print("  Generating report …")
    generate_report(profiles, k, sil, vs_summary, strategies, img_dir, report_path)

    print("  ✅ Opponent Classifier complete.")
    return profiles, strategies


if __name__ == "__main__":
    run()
