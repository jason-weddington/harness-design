#!/bin/sh
P=$(shuf -i 20000-60000 -n 1); echo "port $P" && printf 'a\\b\n' && true
