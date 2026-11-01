#!/usr/bin/env python3
"""Generate a bcrypt password hash for ADMIN_PASSWORD_HASH in .env"""
import getpass, warnings
warnings.filterwarnings("ignore")
from passlib.context import CryptContext

pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
p = getpass.getpass("New admin password: ")
p2 = getpass.getpass("Confirm: ")
if p != p2:
    raise SystemExit("Passwords do not match")
print(f"\nADMIN_PASSWORD_HASH={pwd.hash(p)}")
print("\nPaste this line into your .env file.")
