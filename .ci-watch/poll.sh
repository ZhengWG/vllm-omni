#!/usr/bin/env bash
# Polls vllm-project/vllm-omni CI state and appends a structured log entry
# whenever something interesting changes. Designed to be run under tmux as a
# loop. Not committed to the repo (lives under .ci-watch which is gitignored
# implicitly via no add).
set -uo pipefail

REPO="vllm-project/vllm-omni"
LOG="/workspace/.ci-watch/events.log"
STATE_DIR="/workspace/.ci-watch/state"
mkdir -p "$STATE_DIR"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

snapshot_main_status() {
    # Combined commit status (Buildkite contexts) + GitHub Actions check runs.
    local sha state contexts actions
    sha=$(gh api "repos/$REPO/commits/main" --jq '.sha' 2>/dev/null) || return 1
    state=$(gh api "repos/$REPO/commits/$sha/status" --jq '.state' 2>/dev/null)
    contexts=$(gh api "repos/$REPO/commits/$sha/status" \
        --jq '[.statuses[] | "\(.context)=\(.state)"] | sort | join(",")' 2>/dev/null)
    actions=$(gh api "repos/$REPO/commits/$sha/check-runs" \
        --jq '[.check_runs[] | "\(.name)=\(.conclusion // .status)"] | sort | join(",")' 2>/dev/null)
    printf '%s\t%s\t%s\t%s\n' "$sha" "$state" "$contexts" "$actions"
}

snapshot_failures() {
    # Compact list of recent failed Actions runs (last 50)
    gh run list --repo "$REPO" --limit 50 \
        --json databaseId,conclusion,workflowName,headBranch,event,displayTitle,url \
        --jq '[.[] | select(.conclusion=="failure") | "\(.databaseId)|\(.workflowName)|\(.headBranch)|\(.displayTitle[:80])|\(.url)"] | join("\n")' 2>/dev/null
}

emit() {
    printf '[%s] %s\n' "$(ts)" "$*" >> "$LOG"
}

iter=0
while true; do
    iter=$((iter+1))

    main_now=$(snapshot_main_status || echo "")
    fails_now=$(snapshot_failures || echo "")

    main_prev_file="$STATE_DIR/main.txt"
    fails_prev_file="$STATE_DIR/failures.txt"

    main_prev=""
    [ -f "$main_prev_file" ] && main_prev=$(cat "$main_prev_file")

    fails_prev=""
    [ -f "$fails_prev_file" ] && fails_prev=$(cat "$fails_prev_file")

    if [ "$iter" -eq 1 ]; then
        emit "WATCH_START repo=$REPO"
        emit "MAIN_STATE_INITIAL $main_now"
        if [ -n "$fails_now" ]; then
            emit "RECENT_FAILURES_INITIAL"
            while IFS= read -r line; do
                [ -n "$line" ] && emit "  FAIL $line"
            done <<< "$fails_now"
        fi
    else
        if [ "$main_now" != "$main_prev" ]; then
            emit "MAIN_STATE_CHANGED prev=$main_prev"
            emit "MAIN_STATE_NOW     now=$main_now"
        fi

        # Detect new failures by id
        new_ids=$(comm -23 \
            <(printf '%s\n' "$fails_now" | awk -F'|' '$1!=""{print $1}' | sort) \
            <(printf '%s\n' "$fails_prev" | awk -F'|' '$1!=""{print $1}' | sort))
        if [ -n "$new_ids" ]; then
            while IFS= read -r id; do
                [ -z "$id" ] && continue
                line=$(printf '%s\n' "$fails_now" | awk -F'|' -v id="$id" '$1==id{print $0}')
                emit "NEW_FAILURE $line"
            done <<< "$new_ids"
        fi
    fi

    printf '%s' "$main_now" > "$main_prev_file"
    printf '%s' "$fails_now" > "$fails_prev_file"

    sleep 300
done
