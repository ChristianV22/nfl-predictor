"""
NFL Predictor - Weekly Data Updater
====================================
Run this every Monday after games finish to update the app with fresh data.

Usage:
    python update_nfl_data.py

It will:
1. Pull the latest player stats, schedules, injuries from nflreadpy
2. Rebuild all player profiles and actual scores
3. Write a fresh App.jsx to src/App.jsx automatically

Requirements:
    pip install nflreadpy polars
"""

import nflreadpy as nfl
import polars as pl
import json
import os
import sys
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────────────────────
HISTORY_SEASONS = [2023, 2024, 2025]
CURRENT_SEASON  = nfl.get_current_season()
CURRENT_WEEK    = nfl.get_current_week()
LAST_WEEK       = CURRENT_WEEK - 1

# Path to your App.jsx — adjust if your folder structure is different
SCRIPT_DIR = Path(__file__).parent
APP_JSX_PATH = SCRIPT_DIR / "src" / "App.jsx"

print(f"\n🏈 NFL Predictor Data Updater")
print(f"   Season: {CURRENT_SEASON}  |  Current week: {CURRENT_WEEK}  |  Last completed week: {LAST_WEEK}")
print(f"   Output: {APP_JSX_PATH}\n")

# ── STEP 1: LOAD ALL PLAYER STATS ───────────────────────────────────────────
print("📥 Loading player stats (2023–2025 history + current season)...")
ps_hist = nfl.load_player_stats(HISTORY_SEASONS)
ps_curr = nfl.load_player_stats([CURRENT_SEASON])

def skill_filter(df):
    return df.filter(
        (pl.col('season_type') == 'REG') &
        (pl.col('position').is_in(['QB','RB','WR','TE']))
    )

ps_hist_reg = skill_filter(ps_hist)
ps_curr_reg = skill_filter(ps_curr)
ps_all_reg  = pl.concat([ps_hist_reg, ps_curr_reg])

print(f"   Historical rows: {ps_hist_reg.shape[0]}")
print(f"   Current season rows: {ps_curr_reg.shape[0]}")

# ── STEP 2: DEFENSE STRENGTH FROM SCHEDULES ─────────────────────────────────
print("📥 Loading schedules...")
sched_hist = nfl.load_schedules(HISTORY_SEASONS)
sched_curr = nfl.load_schedules([CURRENT_SEASON])

sched_hist_reg = sched_hist.filter(pl.col('game_type') == 'REG')
sched_curr_reg = sched_curr.filter(pl.col('game_type') == 'REG')

# Defense strength from 2025 (most recent full season)
sched_2025 = sched_hist.filter(
    (pl.col('game_type') == 'REG') & (pl.col('season') == 2025)
)
home_pts = sched_2025.select([pl.col('home_team').alias('team'), pl.col('away_score').alias('pts')])
away_pts = sched_2025.select([pl.col('away_team').alias('team'), pl.col('home_score').alias('pts')])
def_stats = pl.concat([home_pts, away_pts])
def_strength = {
    row['team']: round(row['avg_pts'], 1)
    for row in def_stats.group_by('team').agg(pl.col('pts').mean().alias('avg_pts')).iter_rows(named=True)
}

# ── STEP 3: BUILD TEAM SCHEDULE LOOKUP (compact) ────────────────────────────
print("📥 Building schedule lookup...")
all_weeks = {}
team_schedule = {}

for row in sched_curr_reg.select(['week','away_team','home_team','gameday','gametime','spread_line']).iter_rows(named=True):
    wk = str(row['week'])
    g  = [row['away_team'], row['home_team'], (row['gameday'] or '')[:10], row['gametime'], row['spread_line']]
    if wk not in all_weeks:
        all_weeks[wk] = []
    all_weeks[wk].append(g)
    for team, opp, loc in [(row['away_team'], row['home_team'], 'away'), (row['home_team'], row['away_team'], 'home')]:
        if team not in team_schedule:
            team_schedule[team] = {}
        team_schedule[team][wk] = [opp, loc, (row['gameday'] or '')[:10], row['spread_line']]

# ── STEP 4: INJURIES ────────────────────────────────────────────────────────
print("📥 Loading injury report...")
injuries_raw = nfl.load_injuries([CURRENT_SEASON])
injury_dict = {}
for row in injuries_raw.select(['full_name','team','position','report_primary_injury','report_status','practice_status']).iter_rows(named=True):
    if row['report_status'] in ('Out','Questionable','Doubtful'):
        injury_dict[row['full_name']] = {
            'status':   row['report_status'],
            'injury':   row['report_primary_injury'] or 'unknown',
            'team':     row['team'],
            'practice': row['practice_status'] or '',
        }
print(f"   Injury entries: {len(injury_dict)}")

# ── STEP 5: BUILD PLAYER PROFILES ───────────────────────────────────────────
print("📊 Building player profiles...")

def avg_col(df, season, col):
    rows = df.filter(pl.col('season') == season).select(col).to_series().drop_nulls().to_list()
    return round(sum(rows) / len(rows), 1) if rows else 0.0

# Find top players from current season + recent history
top_by_pos = {}
for pos in ['QB','RB','WR','TE']:
    top = (
        ps_all_reg
        .filter(pl.col('position') == pos)
        .group_by('player_display_name')
        .agg([
            pl.col('fantasy_points_ppr').sum().alias('total_fp'),
            pl.col('season').max().alias('latest_season'),
            pl.col('week').count().alias('games'),
        ])
        .filter((pl.col('games') >= 4) & (pl.col('latest_season') >= 2024))
        .sort('total_fp', descending=True)
        .head(25 if pos in ['WR','RB'] else 15)
    )
    top_by_pos[pos] = top['player_display_name'].to_list()

all_top_players = [p for players in top_by_pos.values() for p in players]
print(f"   Top players identified: {len(all_top_players)}")

player_profiles = {}
stat_profiles   = {}

for name in all_top_players:
    pdata = ps_all_reg.filter(pl.col('player_display_name') == name)
    if pdata.shape[0] == 0:
        continue

    pos  = pdata.sort('season','week').select('position').to_series()[-1]
    team = pdata.sort('season','week').select('team').to_series()[-1]

    # Current season stats
    curr_data = pdata.filter(pl.col('season') == CURRENT_SEASON)
    hist_2025 = pdata.filter(pl.col('season') == 2025)
    hist_2024 = pdata.filter(pl.col('season') == 2024)
    hist_2023 = pdata.filter(pl.col('season') == 2023)

    # Use current season if available, else fall back to 2025
    base_data = curr_data if curr_data.shape[0] >= 2 else hist_2025

    def bavg(col):
        vals = base_data.select(col).to_series().drop_nulls().to_list()
        return round(sum(vals)/len(vals), 1) if vals else 0.0

    recent = pdata.sort('season','week').tail(4)
    def ravg(col):
        vals = recent.select(col).to_series().drop_nulls().to_list()
        return round(sum(vals)/len(vals), 1) if vals else 0.0

    # Last 10 games for sparkline
    last10 = pdata.sort('season','week').tail(10)
    weekly_fp = [
        [row['season'], row['week'], round(row['fantasy_points_ppr'] or 0, 1), row['opponent_team'] or '']
        for row in last10.iter_rows(named=True)
    ]

    # Fantasy point averages
    fp_curr = bavg('fantasy_points_ppr')
    fp_2025 = avg_col(pdata, 2025, 'fantasy_points_ppr')
    fp_2024 = avg_col(pdata, 2024, 'fantasy_points_ppr')
    fp_2023 = avg_col(pdata, 2023, 'fantasy_points_ppr')
    fp_rec  = ravg('fantasy_points_ppr')

    player_profiles[name] = {
        'pos':            pos,
        'team':           team,
        'games_curr':     curr_data.shape[0],
        'games_2025':     hist_2025.shape[0],
        'games_2024':     hist_2024.shape[0],
        'avg_fp_curr':    fp_curr,
        'avg_fp_2025':    fp_2025,
        'avg_fp_2024':    fp_2024,
        'avg_fp_2023':    fp_2023,
        'avg_fp_recent':  fp_rec,
        'avg_pass_yds':   bavg('passing_yards'),
        'avg_rush_yds':   bavg('rushing_yards'),
        'avg_rec_yds':    bavg('receiving_yards'),
        'avg_rec':        bavg('receptions'),
        'avg_targets':    bavg('targets'),
        'avg_tds':        round(bavg('passing_tds') + bavg('rushing_tds') + bavg('receiving_tds'), 1),
        'weekly_fp':      weekly_fp,
    }

    # Per-stat profiles for projection
    if pos == 'QB':
        stat_profiles[name] = {
            'pass_yds_base': bavg('passing_yards'),
            'pass_tds_base': bavg('passing_tds'),
            'ints_base':     bavg('passing_interceptions'),
            'rush_yds_base': bavg('rushing_yards'),
            'pass_yds_rec':  ravg('passing_yards'),
            'pass_tds_rec':  ravg('passing_tds'),
            'ints_rec':      ravg('passing_interceptions'),
            'rush_yds_rec':  ravg('rushing_yards'),
        }
    elif pos == 'RB':
        stat_profiles[name] = {
            'rush_yds_base': bavg('rushing_yards'),
            'rush_tds_base': bavg('rushing_tds'),
            'rush_att_base': bavg('carries'),
            'rec_base':      bavg('receptions'),
            'rec_yds_base':  bavg('receiving_yards'),
            'rec_tds_base':  bavg('receiving_tds'),
            'tgt_base':      bavg('targets'),
            'rush_yds_rec':  ravg('rushing_yards'),
            'rec_yds_rec':   ravg('receiving_yards'),
            'rec_rec':       ravg('receptions'),
        }
    else:
        stat_profiles[name] = {
            'rec_base':      bavg('receptions'),
            'rec_yds_base':  bavg('receiving_yards'),
            'rec_tds_base':  bavg('receiving_tds'),
            'tgt_base':      bavg('targets'),
            'rec_rec':       ravg('receptions'),
            'rec_yds_rec':   ravg('receiving_yards'),
            'tgt_rec':       ravg('targets'),
        }

print(f"   Player profiles built: {len(player_profiles)}")

# ── STEP 6: ACTUAL 2026 SCORES ───────────────────────────────────────────────
print("📥 Loading actual 2026 scores...")
actual_scores = {}
for row in ps_curr_reg.select([
    'player_display_name','position','team','opponent_team','week',
    'fantasy_points_ppr','passing_yards','passing_tds','passing_interceptions',
    'rushing_yards','rushing_tds','carries','receptions','receiving_yards','receiving_tds','targets'
]).iter_rows(named=True):
    fp = round(row['fantasy_points_ppr'] or 0, 1)
    if fp > 0:
        name = row['player_display_name']
        wk   = str(row['week'])
        if name not in actual_scores:
            actual_scores[name] = {}
        actual_scores[name][wk] = {
            'fp':       fp,
            'opp':      row['opponent_team'] or '',
            'pos':      row['position'],
            'team':     row['team'],
            'pass_yds': row['passing_yards'] or 0,
            'pass_tds': row['passing_tds'] or 0,
            'ints':     row['passing_interceptions'] or 0,
            'rush_yds': row['rushing_yards'] or 0,
            'rush_tds': row['rushing_tds'] or 0,
            'carries':  row['carries'] or 0,
            'rec':      row['receptions'] or 0,
            'rec_yds':  row['receiving_yards'] or 0,
            'rec_tds':  row['receiving_tds'] or 0,
            'targets':  row['targets'] or 0,
        }

# Keep only roster players + top scorers each week
filtered_actual = {}
for name, weeks in actual_scores.items():
    if name in player_profiles or any(w['fp'] >= 15 for w in weeks.values()):
        filtered_actual[name] = weeks

print(f"   Actual score entries: {len(filtered_actual)}")

# ── STEP 7: SERIALIZE ALL DATA ───────────────────────────────────────────────
print("📦 Serializing data...")
players_json  = json.dumps(player_profiles)
def_json      = json.dumps(def_strength)
ts_json       = json.dumps(team_schedule)
sched_json    = json.dumps(all_weeks)
inj_json      = json.dumps(injury_dict)
actual_json   = json.dumps(filtered_actual)
sp_json       = json.dumps(stat_profiles)

total_kb = (len(players_json)+len(def_json)+len(ts_json)+len(sched_json)+len(inj_json)+len(actual_json)+len(sp_json))//1024
print(f"   Total data size: {total_kb} KB")

# ── STEP 8: WRITE App.jsx ────────────────────────────────────────────────────
print(f"✍️  Writing {APP_JSX_PATH}...")

header = (
    'import { useState, useMemo } from "react";\n\n'
    f'// Auto-generated by update_nfl_data.py — Season {CURRENT_SEASON} Week {CURRENT_WEEK}\n'
    f'const TEAM_SCHEDULE = {ts_json};\n'
    f'const SCHEDULE = {sched_json};\n'
    f'const INJURIES = {inj_json};\n'
    f'const DEF_STRENGTH = {def_json};\n'
    f'const NFL_PLAYERS = {players_json};\n'
    f'const ACTUAL_{CURRENT_SEASON} = {actual_json};\n'
    f'const STAT_PROFILES = {sp_json};\n'
    f'const CURRENT_WEEK = {CURRENT_WEEK};\n'
    f'const CURRENT_SEASON = {CURRENT_SEASON};\n'
    f'const LAST_WEEK = {LAST_WEEK};\n'
)

# Read the JSX body from the existing App.jsx (everything after the data block)
existing_app = APP_JSX_PATH.read_text(encoding='utf-8') if APP_JSX_PATH.exists() else ""

# Find where the component logic starts (after the const declarations)
marker = '\nconst C = {'
if marker in existing_app:
    jsx_body = existing_app[existing_app.index(marker):]
    full_output = header + jsx_body
else:
    print("⚠️  Could not find component body in existing App.jsx.")
    print("   Make sure App.jsx exists in src/ before running this script.")
    sys.exit(1)

APP_JSX_PATH.write_text(full_output, encoding='utf-8')

print(f"\n✅ Done! App.jsx updated:")
print(f"   Players:    {len(player_profiles)}")
print(f"   Weeks data: {sorted(all_weeks.keys(), key=int)}")
print(f"   Injuries:   {len(injury_dict)}")
print(f"   Actual wks: {sorted(set(w for p in filtered_actual.values() for w in p.keys()), key=int)}")
print(f"\n   Vite will hot-reload automatically if npm run dev is running.")
print(f"   Then commit: git add . && git commit -m 'update week {LAST_WEEK} results'")
print(f"                git push\n")
