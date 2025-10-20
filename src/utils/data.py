# ==============================================================================
# rl-finlab: Reinforcement Learning Finance Experimentation
# ==============================================================================
# Copyright (C) 2025  Harish Naik

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU LesserGeneral Public License along
# with this program. If not, see <https://www.gnu.org/licenses/lgpl-3.0.html>.
# ==============================================================================

import numpy as np
import typing as ty

import yfinance as yf


def load_prices_yf(cfg: ty.Dict[str, ty.Any]) -> np.ndarray:
    """Load prices from Yahoo Finance."""

    data = yf.download(
        cfg['tickers'], start=cfg['start'], end=cfg['end'], progress=False
    )['Adj Close']
    data = data.dropna()
    return data.values
