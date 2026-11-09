#!/usr/bin/env python3
"""Generate a bcrypt password hash for ADMIN_PASSWORD_HASH in .env"""
import getpass
import bcrypt

p = getpass.getpass("New admin password: ")
p2 = getpass.getpass("Confirm: ")
if p != p2:
    raise SystemExit("Passwords do not match")
print(f"\nADMIN_PASSWORD_HASH={bcrypt.hashpw(p.encode()[:72], bcrypt.gensalt(rounds=12)).decode()}")
print("\nPaste this line into your .env file.")
