#!/usr/bin/env bash
# Start the runner in a detached tmux session named `ltx-runner`.
#
# tmux rather than nohup so the process is attachable: a render is minutes long
# and when one fails you want to read the LTX traceback as it scrolls, not
# reconstruct it from a log tail afterwards.
#
#   ./start.sh              start (no-op if already running)
#   tmux attach -t ltx-runner   watch it
#   Ctrl-b d                    detach, leaving it running
#   ./start.sh --restart    pick up new code
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"

SESSION=ltx-runner
PORT="${PORT:-8190}"
PUBLIC_BASE="${PUBLIC_BASE:-http://$(hostname):${PORT}}"

[ "${1:-}" = "--restart" ] && tmux kill-session -t "$SESSION" 2>/dev/null

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "already running — tmux attach -t $SESSION  (or ./start.sh --restart)"
    exit 0
fi

if [ ! -x venv/bin/python ]; then
    echo "creating venv (separate from LTX-2's, which stays untouched)"
    python3 -m venv venv && ./venv/bin/pip -q install -r requirements.txt
fi

# tee, not a plain redirect: tmux scrollback is volatile and dies with the
# session, so a restart used to take every service-level log line with it. This
# keeps the session attachable AND leaves runner.log on disk.
tmux new-session -d -s "$SESSION" \
    "./venv/bin/python app.py --host 0.0.0.0 --port ${PORT} --public-base ${PUBLIC_BASE} 2>&1 \
       | tee -a runner.log; \
     echo; echo '--- runner exited, shell kept so the traceback stays readable ---'; exec bash"

sleep 3
if tmux has-session -t "$SESSION" 2>/dev/null && curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; then
    echo "ltx-runner up on :${PORT}  (tmux attach -t $SESSION)"
else
    echo "did not come up — tmux attach -t $SESSION to see why"; exit 1
fi
