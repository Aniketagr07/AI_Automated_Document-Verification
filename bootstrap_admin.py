"""
DocVerify - Secure Administrator Bootstrap Tool.
Protected administrative setup script for provisioning or resetting an initial
system administrator account without hardcoding credentials in source code.

Usage:
  Interactive mode:
    python bootstrap_admin.py

  Non-interactive / automated provisioning:
    python bootstrap_admin.py --email admin@docverify.gov.in --password YourSecurePassword123!
"""

import sys
import os
import argparse
import getpass
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import database
from auth_manager import hash_password, validate_password_strength, validate_email


def bootstrap_admin(email: str, password: str, full_name: str = "System Administrator") -> bool:
    """Provisions or updates a system administrator securely."""
    database.init_database()
    email_clean = email.strip().lower()

    if not validate_email(email_clean):
        print(f"[-] Error: '{email_clean}' is not a valid email address.")
        return False

    is_strong, msg = validate_password_strength(password)
    if not is_strong:
        print(f"[-] Password validation failed: {msg}")
        return False

    pwd_hash, salt = hash_password(password)
    now = datetime.now().isoformat()

    conn = database.get_db_connection()
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, role FROM users WHERE email = ?", (email_clean,))
            existing = cursor.fetchone()

            if existing:
                # Update existing user to ADMIN with new password
                user_id = existing["id"]
                cursor.execute("""
                    UPDATE users
                    SET password_hash = ?, salt = ?, role = 'ADMIN', full_name = ?, is_active = 1
                    WHERE id = ?
                """, (pwd_hash, salt, full_name, user_id))
                action_text = f"Administrator account '{email_clean}' updated."
            else:
                # Insert new ADMIN
                cursor.execute("""
                    INSERT INTO users (email, password_hash, salt, full_name, role, created_at, is_active)
                    VALUES (?, ?, ?, ?, 'ADMIN', ?, 1)
                """, (email_clean, pwd_hash, salt, full_name, now))
                user_id = cursor.lastrowid
                action_text = f"Administrator account '{email_clean}' provisioned."

            # Record in audit trail
            cursor.execute("""
                INSERT INTO audit_history (user_id, action, actor, timestamp, details)
                VALUES (?, 'ADMIN_BOOTSTRAPPED', 'CLI_SETUP', ?, ?)
            """, (user_id, now, action_text))

            print(f"[+] Success: {action_text}")
            print(f"[+] Role: ADMIN (Protected Route Access Enabled)")
            return True
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="DocVerify Administrator Account Bootstrap")
    parser.add_argument("--email", help="Administrator email address")
    parser.add_argument("--password", help="Administrator password")
    parser.add_argument("--name", default="System Administrator", help="Administrator full name")

    args = parser.parse_args()

    # Read from environment variables if not passed via CLI
    email = args.email or os.environ.get("DOCVERIFY_ADMIN_EMAIL")
    password = args.password or os.environ.get("DOCVERIFY_ADMIN_PASSWORD")
    name = args.name or os.environ.get("DOCVERIFY_ADMIN_NAME", "System Administrator")

    if not email:
        email = input("Enter Administrator Email: ").strip()

    if not password:
        password = getpass.getpass("Enter Secure Administrator Password: ").strip()
        confirm = getpass.getpass("Confirm Administrator Password: ").strip()
        if password != confirm:
            print("[-] Error: Passwords do not match.")
            sys.exit(1)

    success = bootstrap_admin(email, password, name)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
