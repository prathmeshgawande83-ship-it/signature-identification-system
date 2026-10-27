from getpass import getpass
from werkzeug.security import generate_password_hash
from main import init_db,db

init_db()
username=input('Admin username: ').strip()
password=getpass('Admin password (12+ characters): ')
if not username or len(password)<12: raise SystemExit('Username required and password must be at least 12 characters')
with db() as c:
    if c.execute('SELECT id FROM admins WHERE username=?',(username,)).fetchone():
        raise SystemExit('Admin already exists. No password changed.')
    c.execute('INSERT INTO admins(username,password_hash) VALUES(?,?)',(username,generate_password_hash(password)))
print('Admin created.')
