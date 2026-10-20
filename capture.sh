#!/bin/bash
# Run a command and keep a copy of exactly what the terminal showed, for the video's screen captures.
# usage: ./capture.sh <name> <command...>
name="$1"; shift
out="media/terminal/$name.log"
{ echo "sreemanti@MacBook-Pro deploy-ai-app % $*"; "$@" 2>&1; } | tee "$out"
