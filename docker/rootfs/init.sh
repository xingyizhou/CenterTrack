#!/bin/bash

export HASHED_PASSWORD=$(echo -n "$PASSWORD" | npx argon2-cli -e)

/startup.sh
