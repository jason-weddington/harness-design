#!/usr/bin/env bash
if [ -f /app/hello.txt ] && [ "$(cat /app/hello.txt)" = "hello from talos" ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
