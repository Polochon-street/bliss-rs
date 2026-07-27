#!/usr/bin/env bash
#
# ci_check2.sh - run the CI checks locally.
#
# The checks are extracted from .github/workflows/rust.yml at runtime, so this
# script cannot drift from what CI actually runs: add a step to the workflow and
# it shows up here automatically.
#
# Usage:
#   ./ci_check2.sh                  run every check, stop at the first failure
#   ./ci_check2.sh -k               run every check, report all failures at the end
#   ./ci_check2.sh -l               list the checks without running them
#   ./ci_check2.sh -f symphonia     only run checks whose name matches a regex
#   ./ci_check2.sh -j <job>         use a different job from the workflow
#
# Requires: yq (the jq wrapper - https://github.com/kislyuk/yq) and jq.

set -uo pipefail

WORKFLOW="${WORKFLOW:-.github/workflows/rust.yml}"
JOB="build-test-lint-linux"

# Steps that provision a CI machine rather than check the code. Matched against
# the command rather than the step name, so renaming a step in the workflow
# won't silently reintroduce `sudo apt-get` into your local run.
SKIP_RE='apt-get|emsdk|rustup '

keep_going=0
list_only=0
filter='.'

usage() {
    sed -n '3,15p' "$0" | sed 's/^#\ \?//'
}

while [ $# -gt 0 ]; do
    case "$1" in
        -k|--keep-going) keep_going=1 ;;
        -l|--list)       list_only=1 ;;
        -f|--filter)     filter="${2:?-f needs a regex}"; shift ;;
        -j|--job)        JOB="${2:?-j needs a job name}"; shift ;;
        -h|--help)       usage; exit 0 ;;
        *) printf 'unknown argument: %s\n\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

if [ -t 1 ]; then
    BOLD=$'\033[1m'; DIM=$'\033[2m'; RESET=$'\033[0m'
    RED=$'\033[31m'; GREEN=$'\033[32m'; BLUE=$'\033[34m'
else
    BOLD=''; DIM=''; RESET=''; RED=''; GREEN=''; BLUE=''
fi

die() { printf '%serror:%s %s\n' "$RED" "$RESET" "$1" >&2; exit 2; }

for tool in yq jq; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool is required but not installed"
done
[ -f "$WORKFLOW" ] || die "$WORKFLOW not found (run this from the repo root)"

if ! yq -e --arg job "$JOB" '.jobs | has($job)' "$WORKFLOW" >/dev/null 2>&1; then
    printf '%serror:%s no job %s in %s. Available jobs:\n' "$RED" "$RESET" "'$JOB'" "$WORKFLOW" >&2
    yq -r '.jobs | keys[]' "$WORKFLOW" | sed 's/^/  - /' >&2
    exit 2
fi

# One compact JSON object per step. Going through JSON rather than raw lines
# means a multi-line `run: |` block survives as an escaped \n instead of being
# split into broken fragments.
steps=$(yq -c -r --arg job "$JOB" --arg skip "$SKIP_RE" '
    .jobs[$job].steps[]
    | select(.run)
    | select(.run | test($skip) | not)
    | {name: (.name // "unnamed"), cmd: (.run | gsub(" --verbose"; "") | sub("\\s+$"; ""))}
' "$WORKFLOW") || die "could not parse $WORKFLOW"

[ -n "$steps" ] || die "no runnable steps found in job '$JOB'"

names=(); cmds=()
while IFS= read -r line; do
    name=$(jq -r '.name' <<<"$line")
    cmd=$(jq -r '.cmd' <<<"$line")
    [[ "$name" =~ $filter ]] || continue
    names+=("$name"); cmds+=("$cmd")
done <<<"$steps"

total=${#names[@]}
[ "$total" -gt 0 ] || die "no checks matched filter '$filter'"

if [ "$list_only" -eq 1 ]; then
    printf '%s%d checks from %s (job: %s)%s\n\n' "$BOLD" "$total" "$WORKFLOW" "$JOB" "$RESET"
    for i in "${!names[@]}"; do
        printf '%s%2d. %s%s\n' "$BOLD" $((i + 1)) "${names[$i]}" "$RESET"
        printf '%s    %s%s\n' "$DIM" "${cmds[$i]}" "$RESET"
    done
    exit 0
fi

failed=()
start_all=$SECONDS

for i in "${!names[@]}"; do
    printf '%s[%d/%d] %s%s\n' "$BOLD$BLUE" $((i + 1)) "$total" "${names[$i]}" "$RESET"
    printf '%s        %s%s\n' "$DIM" "${cmds[$i]}" "$RESET"

    start=$SECONDS
    if bash -c "${cmds[$i]}"; then
        printf '%s        passed%s (%ds)\n\n' "$GREEN" "$RESET" $((SECONDS - start))
    else
        printf '%s        FAILED%s (%ds)\n\n' "$RED" "$RESET" $((SECONDS - start))
        failed+=("${names[$i]}")
        if [ "$keep_going" -eq 0 ]; then
            printf '%sstopped at the first failure; pass -k to run the rest anyway%s\n' "$DIM" "$RESET"
            break
        fi
    fi
done

elapsed=$((SECONDS - start_all))
printf '%s---%s\n' "$DIM" "$RESET"

if [ ${#failed[@]} -eq 0 ]; then
    printf '%sall %d checks passed%s in %dm%02ds\n' \
        "$GREEN$BOLD" "$total" "$RESET" $((elapsed / 60)) $((elapsed % 60))
    exit 0
fi

printf '%s%d of %d checks failed%s (%dm%02ds):\n' \
    "$RED$BOLD" "${#failed[@]}" "$total" "$RESET" $((elapsed / 60)) $((elapsed % 60))
for f in "${failed[@]}"; do
    printf '  - %s\n' "$f"
done
exit 1
