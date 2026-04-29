#!/usr/bin/env bash
#
# release_to_rldx1.sh — publish a clean release of this repo to rlwrld/RLDX-1
#
# Strategy
#   1. Mirror-clone source repo (RLWRLD/RLDX) into a tmp dir — never touches local checkout.
#   2. Initialise submodules so gitlinks resolve.
#   3. Remove release-excluded paths (tests/, internal scripts, TODO file, etc).
#   4. Orphan branch + single squashed commit "RLDX-1 Release".
#   5. Force-push to rlwrld/RLDX-1 main (after explicit confirmation).
#
# Usage
#   bash scripts/release_to_rldx1.sh                        # default: source=main, target=main
#   bash scripts/release_to_rldx1.sh --source-branch BR     # release a different source branch
#   bash scripts/release_to_rldx1.sh --dry-run              # build the tree but skip push (workdir preserved)
#   bash scripts/release_to_rldx1.sh --yes                  # skip interactive confirmation
#
# Requirements: git, ssh access to both source and target repos.

set -euo pipefail

SOURCE_REMOTE="git@github.com:RLWRLD/RLDX.git"
TARGET_REMOTE="git@github.com:rlwrld/RLDX-1.git"
SOURCE_BRANCH="refactoring-for-release"
TARGET_BRANCH="main"
COMMIT_MSG="RLDX-1 Release"

DRY_RUN=0
ASSUME_YES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-branch) SOURCE_BRANCH="$2"; shift 2 ;;
    --target-branch) TARGET_BRANCH="$2"; shift 2 ;;
    --source-remote) SOURCE_REMOTE="$2"; shift 2 ;;
    --target-remote) TARGET_REMOTE="$2"; shift 2 ;;
    --dry-run)       DRY_RUN=1;          shift ;;
    --yes|-y)        ASSUME_YES=1;       shift ;;
    -h|--help)
      sed -n '2,22p' "$0"; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

# Paths that are tracked in source but must NOT ship in the release.
# Submodule entries (gitlinks) are intentionally preserved — they point at upstream commits.
EXCLUDE_PATHS=(
  "tests"
  "TODO_BEFORE_RELEASE.md"
  "scripts/smoke_rtc_real_obs.py"
  "scripts/verify_all_production_ckpts.py"
)

WORKDIR="$(mktemp -d -t rldx1-release-XXXXXX)"
KEEP_WORKDIR=0
cleanup() {
  if [[ "$KEEP_WORKDIR" == "1" ]]; then
    echo "Workdir preserved: $WORKDIR"
  else
    rm -rf "$WORKDIR"
  fi
}
trap cleanup EXIT

REPO="$WORKDIR/repo"

echo "[1/6] Cloning $SOURCE_REMOTE (branch: $SOURCE_BRANCH)"
git clone --branch "$SOURCE_BRANCH" --single-branch "$SOURCE_REMOTE" "$REPO"
cd "$REPO"

echo "[2/6] Initialising submodules (preserved as gitlinks in release)"
git submodule update --init --recursive

echo "[3/6] Removing release-excluded paths"
for p in "${EXCLUDE_PATHS[@]}"; do
  if [[ -e "$p" ]]; then
    echo "       - $p"
    rm -rf "$p"
  else
    echo "       . skip (absent): $p"
  fi
done

echo "[4/6] Defensive sweep for stray test_*.py / *_test.py outside submodules"
SUBMODULE_PATHS=()
if [[ -f .gitmodules ]]; then
  while read -r path; do
    [[ -n "$path" ]] && SUBMODULE_PATHS+=("./$path")
  done < <(git config --file .gitmodules --get-regexp '^submodule\..*\.path$' | awk '{print $2}')
fi
FIND_PRUNE=()
for sm in "${SUBMODULE_PATHS[@]}"; do
  FIND_PRUNE+=(-path "$sm" -prune -o)
done
# also skip .git
FIND_PRUNE+=(-path "./.git" -prune -o)
mapfile -t STRAYS < <(find . "${FIND_PRUNE[@]}" \
  \( -name 'test_*.py' -o -name '*_test.py' \) -type f -print)
if (( ${#STRAYS[@]} > 0 )); then
  printf '       removing stray: %s\n' "${STRAYS[@]}"
  rm -f "${STRAYS[@]}"
else
  echo "       (none)"
fi

echo "[5/6] Building single squashed orphan commit: \"$COMMIT_MSG\""
git checkout --orphan release-staging
git add -A
# git commit needs an identity in case the env has none configured
git -c user.name="rldx-release-bot" \
    -c user.email="release@rlwrld.ai" \
    commit -m "$COMMIT_MSG" >/dev/null
echo "       commit: $(git rev-parse --short HEAD)  tree-size: $(git ls-tree -r HEAD | wc -l) files"

echo "[6/6] Pushing to $TARGET_REMOTE ($TARGET_BRANCH)"
git remote add target "$TARGET_REMOTE"

if [[ "$DRY_RUN" == "1" ]]; then
  echo
  echo "[DRY RUN] would run: git push --force target HEAD:refs/heads/$TARGET_BRANCH"
  KEEP_WORKDIR=1
  exit 0
fi

if [[ "$ASSUME_YES" != "1" ]]; then
  echo
  echo "================================================================"
  echo "  WILL FORCE PUSH to $TARGET_REMOTE  branch=$TARGET_BRANCH"
  echo "  This OVERWRITES the target's history with a single commit."
  echo "================================================================"
  read -r -p "Type 'release' to continue (anything else aborts): " confirm
  [[ "$confirm" == "release" ]] || { echo "Aborted."; exit 1; }
fi

git push --force target "HEAD:refs/heads/$TARGET_BRANCH"
echo
echo "Done. Released $(git rev-parse --short HEAD) -> $TARGET_REMOTE ($TARGET_BRANCH)."
