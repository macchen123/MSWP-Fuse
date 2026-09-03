#!/usr/bin/env bash
set -euo pipefail

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
THIRD_PARTY="$ROOT/third_party"
TSLIB="$THIRD_PARTY/Time-Series-Library"
PYTORCH_FORECASTING="$THIRD_PARTY/pytorch-forecasting"
TSLIB_COMMIT="4e938a1767106324dd753b2a44832bf870a0252e"
PYTORCH_FORECASTING_COMMIT="4d8d97cd3e85a15a9b90a38dfb0afc819d8e8aa4"

checkout_commit() {
    local directory="$1"
    local url="$2"
    local commit="$3"
    if [ ! -d "$directory/.git" ]; then
        git clone "$url" "$directory"
    fi
    git -C "$directory" fetch --depth 1 origin "$commit"
    git -C "$directory" checkout --detach "$commit"
    test "$(git -C "$directory" rev-parse HEAD)" = "$commit"
}

mkdir -p "$THIRD_PARTY"
checkout_commit "$TSLIB" "https://github.com/thuml/Time-Series-Library.git" "$TSLIB_COMMIT"
checkout_commit "$PYTORCH_FORECASTING" "https://github.com/sktime/pytorch-forecasting.git" "$PYTORCH_FORECASTING_COMMIT"
python -m pip install --editable "$PYTORCH_FORECASTING"
printf '%s\n' "Pinned upstream checkouts are ready."
