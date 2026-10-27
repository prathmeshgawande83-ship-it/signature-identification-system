import sqlite3

db = sqlite3.connect("signatures.db")
email = input("Enter the email of your NEW account: ").strip()

row = db.execute(
    "SELECT id, name, role FROM users WHERE email = ? COLLATE NOCASE",
    (email,)
).fetchone()

if row is None:
    print("Account not found. No changes made.")
else:
    confirm = input(f"Make {row[1]} an admin? Type YES to confirm: ").strip()
    if confirm == "YES":
        db.execute(
            "UPDATE users SET role = 'admin' WHERE id = ?",
            (row[0],)
        )
        db.commit()
        print("Admin role updated successfully.")
    else:
        print("Cancelled. No changes made.")

db.close()
