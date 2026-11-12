"""Frozen model settings.

These are fixed so that week to week changes in the odds come from football,
not from the model being re-tuned. Each value is backed by a test in
tests/test_model_evidence.py that fails if the data stops supporting it.
Re-tune only between seasons.
"""

from .elo import EloParams

ELO = EloParams(k=12.0, hfa=0.0, promoted=1400.0)
GOALS_WINDOW = 6  # seasons of history used to fit the goals model
SIGMA = 65.0  # Elo points of rating uncertainty, redrawn each simulated season
N_SIMS = 10_000
SEED = 2026
