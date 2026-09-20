#!/usr/bin/env bash
# Clone the three comparison baselines next to our integration code.
# They are not vendored in this repository; see baselines/README.md for why.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

clone () {          # name url
  if [ -d "$1/.git" ]; then
    echo "== $1 already present"
  else
    echo "== cloning $1"
    git clone --depth 1 "$2" "$1"
  fi
  printf '   %s at commit %s\n' "$1" "$(git -C "$1" rev-parse --short HEAD)"
  if [ ! -f "$1/LICENSE" ] && [ ! -f "$1/LICENSE.md" ]; then
    echo "   NOTE: upstream ships no LICENSE file; check its terms before redistributing."
  fi
}

clone NSReg      https://github.com/mala-lab/NSReg.git
clone ConsisGAD  https://github.com/Xtra-Computing/ConsisGAD.git
clone GGAD       https://github.com/mala-lab/GGAD.git

cat <<'NOTE'

Record the commits printed above alongside any numbers you report. Upstream
baselines change, and a comparison you cannot re-derive is not a comparison.
NOTE
