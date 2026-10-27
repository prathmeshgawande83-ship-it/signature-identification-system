import os, io, csv, uuid, sqlite3, secrets, hmac
from pathlib import Path
from datetime import datetime, timezone
from functools import wraps
from collections import Counter
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file, abort
from werkzeug.security import generate_password_hash, check_password_hash
from authlib.integrations.flask_client import OAuth
import re
from werkzeug.utils import secure_filename
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances, accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from skimage.feature import hog
import joblib
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

BASE = Path(__file__).resolve().parent
DB = BASE / 'signatures.db'
UPLOAD = BASE / 'static' / 'signatures'
MODELS = BASE / 'models'
UPLOAD.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)
app = Flask(__name__)
app.config.update(SECRET_KEY=os.environ.get('FLASK_SECRET_KEY') or secrets.token_hex(32), MAX_CONTENT_LENGTH=8*1024*1024, SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE='Lax', SESSION_COOKIE_SECURE=os.environ.get('COOKIE_SECURE') == '1')
oauth = OAuth(app)
GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID', '').strip()
GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET', '').strip()
if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    oauth.register(name='google', client_id=GOOGLE_CLIENT_ID, client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
        client_kwargs={'scope': 'openid email profile'})
ALLOWED = {'.png','.jpg','.jpeg','.bmp'}
NOW = lambda: datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

def audit(action, details=''):
    try:
        with db() as c:
            c.execute('INSERT INTO audit_log(admin_id,action,details,ip_address,created_at) VALUES(?,?,?,?,?)', (session.get('admin_id'), action, str(details)[:500], request.remote_addr or 'unknown', NOW()))
    except Exception:
        app.logger.exception('Could not write audit log')

def db():
    c=sqlite3.connect(DB); c.row_factory=sqlite3.Row; c.execute('PRAGMA foreign_keys=ON'); return c

def init_db():
    with db() as c:
        c.executescript('''
        CREATE TABLE IF NOT EXISTS students(id INTEGER PRIMARY KEY AUTOINCREMENT,name TEXT NOT NULL,roll_no TEXT NOT NULL UNIQUE,created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS signatures(id INTEGER PRIMARY KEY AUTOINCREMENT,student_id INTEGER NOT NULL,image_path TEXT NOT NULL,created_at TEXT NOT NULL,FOREIGN KEY(student_id) REFERENCES students(id));
        CREATE TABLE IF NOT EXISTS admins(id INTEGER PRIMARY KEY AUTOINCREMENT,username TEXT UNIQUE NOT NULL,password_hash TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, email TEXT UNIQUE NOT NULL COLLATE NOCASE, password_hash TEXT, google_sub TEXT UNIQUE, created_at TEXT NOT NULL, role TEXT NOT NULL DEFAULT 'user');
        CREATE TABLE IF NOT EXISTS identification_history(id INTEGER PRIMARY KEY AUTOINCREMENT,image_path TEXT,student_id INTEGER,score REAL,status TEXT NOT NULL,verification TEXT NOT NULL,created_at TEXT NOT NULL,FOREIGN KEY(student_id) REFERENCES students(id));
        CREATE TABLE IF NOT EXISTS audit_log(id INTEGER PRIMARY KEY AUTOINCREMENT,admin_id INTEGER,action TEXT NOT NULL,details TEXT,ip_address TEXT,created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS login_attempts(id INTEGER PRIMARY KEY AUTOINCREMENT,ip_address TEXT NOT NULL,success INTEGER NOT NULL,created_at TEXT NOT NULL);
        ''')
        columns={r['name'] for r in c.execute('PRAGMA table_info(signatures)')}
        if 'label' not in columns: c.execute("ALTER TABLE signatures ADD COLUMN label TEXT NOT NULL DEFAULT 'genuine'")
        # Never reset admin credentials automatically on startup.

def login_required(fn):
    @wraps(fn)
    def inner(*args, **kwargs):
        if not session.get('admin_id') and not session.get('user_id'):
            return redirect(url_for('login', next=request.path))
        return fn(*args, **kwargs)
    return inner

def admin_required(fn):
    @wraps(fn)
    def inner(*args,**kwargs):
        if not session.get('admin_id'): return redirect(url_for('login'))
        return fn(*args,**kwargs)
    return inner

@app.before_request
def csrf_guard():
    if 'csrf' not in session: session['csrf']=secrets.token_urlsafe(32)
    if request.method=='POST' and not hmac.compare_digest(request.form.get('csrf',''),session['csrf']): abort(400,'Invalid CSRF token')

@app.context_processor
def globals_(): return {'csrf':session.get('csrf','')}

def image_from_bytes(data):
    if not data or len(data)>8*1024*1024: raise ValueError('Empty or oversized image')
    arr=np.frombuffer(data,dtype=np.uint8)
    image=cv2.imdecode(arr,cv2.IMREAD_COLOR)
    if image is None: raise ValueError('Invalid image')
    if image.shape[0]*image.shape[1]>12_000_000: raise ValueError('Image dimensions too large')
    return image

def save_image(file):
    if not file or not file.filename: raise ValueError('Choose an image')
    suffix=Path(secure_filename(file.filename)).suffix.lower()
    if suffix not in ALLOWED: raise ValueError('Use JPG, PNG or BMP')
    image=image_from_bytes(file.read())
    filename=uuid.uuid4().hex+'.png'
    if not cv2.imwrite(str(UPLOAD/filename),image): raise ValueError('Could not save image')
    return 'static/signatures/'+filename

def remove_image(path):
    if not path: return
    try:
        target=(BASE/path).resolve()
        if target.parent==UPLOAD.resolve(): target.unlink(missing_ok=True)
    except OSError: pass

def features(path):
    im=cv2.imread(str(BASE/path),cv2.IMREAD_GRAYSCALE)
    if im is None: raise ValueError('Missing or invalid stored image')
    im=cv2.GaussianBlur(im,(3,3),0)
    _,bw=cv2.threshold(im,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    pts=cv2.findNonZero(bw)
    if pts is None: raise ValueError('Blank signature')
    x,y,w,h=cv2.boundingRect(pts)
    crop=bw[y:y+h,x:x+w]
    scale=min(240/w,100/h)
    resized=cv2.resize(crop,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_AREA)
    canvas=np.zeros((120,260),np.uint8)
    yy=(120-resized.shape[0])//2; xx=(260-resized.shape[1])//2
    canvas[yy:yy+resized.shape[0],xx:xx+resized.shape[1]]=resized
    return hog(canvas,pixels_per_cell=(16,16),cells_per_block=(2,2),feature_vector=True).astype(np.float32)

def data_rows():
    with db() as c: return c.execute('SELECT * FROM signatures ORDER BY id').fetchall()

def train_models():
    rows=data_rows(); valid=[]
    for r in rows:
        try: valid.append((r,features(r['image_path'])))
        except (ValueError,cv2.error): pass
    genuine=[(r,f) for r,f in valid if r['label']=='genuine']
    counts=Counter(r['student_id'] for r,f in genuine)
    # Require >=3 samples per writer, >=2 writers. Evaluation must use separate held-out data.
    eligible=[(r,f) for r,f in genuine if counts[r['student_id']]>=3]
    if len(set(r['student_id'] for r,f in eligible))<2: raise ValueError('Need at least 2 students with 3 genuine signatures each')
    X=np.array([f for r,f in eligible]); y=np.array([r['student_id'] for r,f in eligible])
    model=make_pipeline(StandardScaler(),SVC(kernel='rbf',probability=False,class_weight='balanced'))
    model.fit(X,y)
    # Open-set rejection: distance to same-writer training examples, conservative percentile.
    dist=pairwise_distances(X,metric='cosine')
    own=[min(dist[i,j] for j in range(len(y)) if i!=j and y[j]==y[i]) for i in range(len(y))]
    threshold=min(0.45,max(0.08,float(np.quantile(own,0.9))*1.2))
    joblib.dump({'model':model,'X':X,'y':y,'threshold':threshold},MODELS/'identification.joblib')
    # Separate verification model: genuine reference-vs-genuine query (positive) and
    # explicitly labeled imitation-vs-genuine reference (negative).
    genuine_by_id={sid:[f for r,f in genuine if r['student_id']==sid] for sid in counts}
    pairs=[]; labels=[]
    for sid,vecs in genuine_by_id.items():
        for i in range(len(vecs)):
            for j in range(i+1,len(vecs)):
                pairs.append(np.abs(vecs[i]-vecs[j])); labels.append(1)
    for r,f in valid:
        if r['label']=='forged' and r['student_id'] in genuine_by_id:
            for ref in genuine_by_id[r['student_id']]:
                pairs.append(np.abs(f-ref)); labels.append(0)
    if len(set(labels))==2 and labels.count(0)>=3 and labels.count(1)>=3:
        verifier=make_pipeline(StandardScaler(),SVC(kernel='linear',class_weight='balanced'))
        verifier.fit(np.array(pairs),np.array(labels))
        joblib.dump(verifier,MODELS/'verification.joblib')
        verify_status='Trained (experimental)'
    else:
        (MODELS/'verification.joblib').unlink(missing_ok=True)
        verify_status='Not trained: add at least 3 labeled forged comparisons'
    return f'Writer model trained on {len(eligible)} genuine images; verification: {verify_status}. Models require held-out evaluation.'

def invalidate_models():
    # Stored models contain writer IDs and feature vectors; force retraining after dataset changes.
    for filename in ('identification.joblib', 'verification.joblib'):
        (MODELS / filename).unlink(missing_ok=True)

def identify_image(path):
    model_file=MODELS/'identification.joblib'
    if not model_file.exists(): return None,0,'Model not trained','Not available',None
    artifact=joblib.load(model_file)
    q=features(path)
    pred=int(artifact['model'].predict([q])[0])
    ix=np.where(artifact['y']==pred)[0]
    distances=pairwise_distances(q.reshape(1,-1),artifact['X'][ix],metric='cosine')[0]
    closest=float(np.min(distances))
    score=round(max(0,100*(1-closest)),1) # similarity, not probability
    if closest>artifact['threshold']: return None,score,'No reliable match','Not available',None
    with db() as c:
        student=c.execute('SELECT * FROM students WHERE id=?',(pred,)).fetchone()
        refs=c.execute("SELECT image_path FROM signatures WHERE student_id=? AND label='genuine'",(pred,)).fetchall()
    best_path=None; best_distance=float('inf'); best_feat=None
    for r in refs:
        try:
            feat=features(r['image_path'])
            d=float(pairwise_distances(q.reshape(1,-1),feat.reshape(1,-1),metric='cosine')[0,0])
            if d<best_distance: best_distance=d; best_path=r['image_path']; best_feat=feat
        except ValueError: continue
    verification='Not available (verification model not trained)'
    verifier_file=MODELS/'verification.joblib'
    if verifier_file.exists() and best_feat is not None:
        v=joblib.load(verifier_file)
        verification='Genuine-like (experimental)' if int(v.predict([np.abs(q-best_feat)])[0])==1 else 'Imitation-like (experimental)'
    return student,score,'Matched (experimental)',verification,best_path

@app.route('/login',methods=['GET','POST'])
def login():
    ip = request.remote_addr or 'unknown'
    if request.method=='POST':
        username=request.form.get('username','').strip()
        password=request.form.get('password','')
        with db() as c:
            recent = c.execute("SELECT COUNT(*) FROM login_attempts WHERE ip_address=? AND success=0 AND created_at >= strftime('%Y-%m-%d %H:%M:%S UTC','now','-15 minutes')", (ip,)).fetchone()[0]
            admin=c.execute('SELECT * FROM admins WHERE username=?',(username,)).fetchone()
            user=c.execute('SELECT * FROM users WHERE email=? COLLATE NOCASE',(username,)).fetchone()
        if recent >= 5:
            flash('Too many failed attempts. Wait 15 minutes before trying again.','error')
            return render_template('login.html'), 429
        if admin and check_password_hash(admin['password_hash'],password):
            with db() as c: c.execute('INSERT INTO login_attempts(ip_address,success,created_at) VALUES(?,?,?)',(ip,1,NOW()))
            session.clear(); session['admin_id']=admin['id']; session['role']='admin'; session['display_name']=admin['username']; session['csrf']=secrets.token_urlsafe(32)
            audit('login_success', f'admin={admin["username"]}')
            return redirect(url_for('index'))
        if user and user['password_hash'] and check_password_hash(user['password_hash'],password):
            with db() as c: c.execute('INSERT INTO login_attempts(ip_address,success,created_at) VALUES(?,?,?)',(ip,1,NOW()))
            session.clear()
            session['user_id']=user['id']
            session['role']=user['role'] or 'user'
            session['display_name']=user['name']
            session['csrf']=secrets.token_urlsafe(32)
            # Accounts explicitly promoted by an existing admin can access the full admin workspace.
            if session['role'] == 'admin':
                session['admin_id']=user['id']
                audit('admin_user_login_success', f'user_id={user["id"]}')
                return redirect(url_for('index'))
            audit('user_login_success', f'user_id={user["id"]}')
            return redirect(url_for('identify'))
        with db() as c: c.execute('INSERT INTO login_attempts(ip_address,success,created_at) VALUES(?,?,?)',(ip,0,NOW()))
        audit('login_failed', f'identifier={username[:80]}')
        flash('Invalid email/username or password.','error')
    return render_template('login.html', google_enabled=bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET))

@app.route('/signup', methods=['GET','POST'])
def signup():
    if request.method == 'POST':
        name=request.form.get('name','').strip()
        email=request.form.get('email','').strip().lower()
        password=request.form.get('password','')
        if len(name)<2 or len(name)>100 or not re.fullmatch(r'[^@\s]+@[^@\s]+\.[^@\s]+', email):
            flash('Enter a valid name and email address.','error')
            return render_template('signup.html')
        if len(password)<12:
            flash('Use a password with at least 12 characters.','error')
            return render_template('signup.html')
        try:
            with db() as c:
                c.execute('INSERT INTO users(name,email,password_hash,created_at,role) VALUES(?,?,?,?,?)',(name,email,generate_password_hash(password),NOW(),'user'))
            flash('Account created. You can now sign in.','success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('An account with that email already exists. Try logging in.','error')
    return render_template('signup.html')

@app.route('/auth/google')
def google_login():
    if not (GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET):
        flash('Google login is not configured yet. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET first.','error')
        return redirect(url_for('login'))
    redirect_uri=url_for('google_callback', _external=True)
    return oauth.google.authorize_redirect(redirect_uri, prompt='select_account')

@app.route('/auth/google/callback')
def google_callback():
    if not (GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET):
        return redirect(url_for('login'))
    try:
        token=oauth.google.authorize_access_token()
        info=token.get('userinfo') or oauth.google.userinfo()
        email=(info.get('email') or '').strip().lower()
        subject=info.get('sub')
        name=(info.get('name') or email.split('@')[0] or 'SignaLab user').strip()
        if not email or not subject or info.get('email_verified') is not True:
            flash('Google did not provide a verified email address.','error')
            return redirect(url_for('login'))
        with db() as c:
            user=c.execute('SELECT * FROM users WHERE google_sub=? OR email=? COLLATE NOCASE',(subject,email)).fetchone()
            if user:
                # Link an existing email account only when Google verifies the same email.
                c.execute('UPDATE users SET google_sub=? WHERE id=? AND (google_sub IS NULL OR google_sub=?)',(subject,user['id'],subject))
                user_id=user['id']
                display=user['name']
            else:
                cur=c.execute('INSERT INTO users(name,email,google_sub,created_at,role) VALUES(?,?,?,?,?)',(name,email,subject,NOW(),'user'))
                user_id=cur.lastrowid; display=name
        with db() as c:
            role_row = c.execute('SELECT role FROM users WHERE id=?', (user_id,)).fetchone()
        role = (role_row['role'] if role_row else 'user') or 'user'
        session.clear(); session['user_id']=user_id; session['role']=role; session['display_name']=display; session['csrf']=secrets.token_urlsafe(32)
        if role == 'admin':
            session['admin_id']=user_id
            audit('google_admin_login_success', f'user_id={user_id}')
            return redirect(url_for('index'))
        audit('google_login_success',f'user_id={user_id}')
        return redirect(url_for('identify'))
    except Exception:
        app.logger.exception('Google OAuth login failed')
        flash('Google sign-in failed. Please try again or use email and password.','error')
        return redirect(url_for('login'))

@app.route('/logout',methods=['POST'])
@login_required
def logout():
    audit('logout')
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
@admin_required
def index():
    with db() as c:
        stats={'students':c.execute('SELECT COUNT(*) FROM students').fetchone()[0], 'signatures':c.execute('SELECT COUNT(*) FROM signatures').fetchone()[0], 'checks':c.execute('SELECT COUNT(*) FROM identification_history').fetchone()[0]}
        students=c.execute('SELECT s.*,COUNT(g.id) signature_count FROM students s LEFT JOIN signatures g ON s.id=g.student_id GROUP BY s.id ORDER BY s.id DESC').fetchall()
        history=c.execute('SELECT h.*,s.name FROM identification_history h LEFT JOIN students s ON h.student_id=s.id ORDER BY h.id DESC LIMIT 10').fetchall()
    return render_template('dashboard.html',stats=stats,students=students,history=history)

@app.route('/register',methods=['GET','POST'])
@admin_required
def register():
    if request.method=='POST':
        name=request.form.get('name','').strip(); roll=request.form.get('roll_no','').strip()
        if not name or not roll: flash('Name and roll number required','error'); return redirect(url_for('register'))
        path=None
        try:
            path=save_image(request.files.get('signature'))
            features(path)
            with db() as c:
                cur=c.execute('INSERT INTO students(name,roll_no,created_at) VALUES(?,?,?)',(name,roll,NOW()))
                c.execute('INSERT INTO signatures(student_id,image_path,created_at,label) VALUES(?,?,?,?)',(cur.lastrowid,path,NOW(),'genuine'))
            audit('student_registered', f'roll_no={roll}')
            flash('Student registered','success'); return redirect(url_for('index'))
        except (ValueError,sqlite3.IntegrityError) as e:
            remove_image(path); flash(str(e),'error')
    return render_template('register.html')

@app.route('/student/<int:student_id>/edit', methods=['GET', 'POST'])
@admin_required
def edit_student(student_id):
    with db() as c:
        student = c.execute('SELECT * FROM students WHERE id=?', (student_id,)).fetchone()
    if student is None:
        abort(404)
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        roll_no = request.form.get('roll_no', '').strip()
        if not name or not roll_no:
            flash('Name and roll number are required.', 'error')
            return redirect(url_for('edit_student', student_id=student_id))
        try:
            with db() as c:
                c.execute('UPDATE students SET name=?, roll_no=? WHERE id=?', (name, roll_no, student_id))
            audit('student_updated', f'id={student_id}; roll_no={roll_no}')
            flash('Student details updated.', 'success')
            return redirect(url_for('student_detail', student_id=student_id))
        except sqlite3.IntegrityError:
            flash('That roll number is already assigned to another student.', 'error')
    return render_template('edit_student.html', student=student)

@app.route('/student/<int:student_id>/delete', methods=['POST'])
@admin_required
def delete_student(student_id):
    with db() as c:
        student = c.execute('SELECT id FROM students WHERE id=?', (student_id,)).fetchone()
        if student is None:
            abort(404)
        paths = [r['image_path'] for r in c.execute('SELECT image_path FROM signatures WHERE student_id=?', (student_id,)).fetchall()]
        # Preserve audit history while unlinking it from the deleted student.
        c.execute('UPDATE identification_history SET student_id=NULL WHERE student_id=?', (student_id,))
        c.execute('DELETE FROM signatures WHERE student_id=?', (student_id,))
        c.execute('DELETE FROM students WHERE id=?', (student_id,))
    for image_path in paths:
        remove_image(image_path)
    audit('student_deleted', f'id={student_id}; samples={len(paths)}')
    invalidate_models()
    flash('Student and their stored signature samples were deleted. Retrain models before identifying signatures.', 'success')
    return redirect(url_for('index'))

@app.route('/student/<int:student_id>')
@admin_required
def student_detail(student_id):
    with db() as c:
        student=c.execute('SELECT * FROM students WHERE id=?',(student_id,)).fetchone()
        rows=c.execute('SELECT * FROM signatures WHERE student_id=? ORDER BY id DESC',(student_id,)).fetchall()
    if not student: abort(404)
    return render_template('student.html',student=student,rows=rows)

@app.route('/signature/<int:signature_id>/delete', methods=['POST'])
@admin_required
def delete_signature(signature_id):
    with db() as c:
        row = c.execute('SELECT id, student_id, image_path FROM signatures WHERE id=?', (signature_id,)).fetchone()
        if row is None:
            abort(404)
        student_id = row['student_id']
        c.execute('DELETE FROM signatures WHERE id=?', (signature_id,))
    remove_image(row['image_path'])
    audit('signature_deleted', f'id={signature_id}; student_id={student_id}')
    invalidate_models()
    flash('Signature sample deleted. Retrain models before the next identification.', 'success')
    return redirect(url_for('student_detail', student_id=student_id))

@app.route('/student/<int:student_id>/add-signature',methods=['POST'])
@admin_required
def add_signature(student_id):
    with db() as c: exists=c.execute('SELECT id FROM students WHERE id=?',(student_id,)).fetchone()
    if not exists: abort(404)
    path=None
    try:
        path=save_image(request.files.get('signature')); features(path)
        label=request.form.get('label','genuine')
        if label not in ('genuine','forged'): raise ValueError('Invalid label')
        with db() as c: c.execute('INSERT INTO signatures(student_id,image_path,created_at,label) VALUES(?,?,?,?)',(student_id,path,NOW(),label))
        audit('signature_added', f'student_id={student_id}; label={label}')
        flash('Sample saved. Retrain models to include it.','success')
    except ValueError as e: remove_image(path); flash(str(e),'error')
    return redirect(url_for('student_detail',student_id=student_id))

@app.route('/dataset',methods=['GET','POST'])
@admin_required
def dataset():
    if request.method=='POST':
        roll=request.form.get('roll_no','').strip(); label=request.form.get('label','genuine')
        if label not in ('genuine','forged'): abort(400)
        with db() as c: student=c.execute('SELECT * FROM students WHERE roll_no=?',(roll,)).fetchone()
        if not student: flash('Register this student first','error'); return redirect(url_for('dataset'))
        uploaded=0; errors=0
        for file in request.files.getlist('images')[:100]:
            path=None
            try:
                path=save_image(file); features(path)
                with db() as c: c.execute('INSERT INTO signatures(student_id,image_path,created_at,label) VALUES(?,?,?,?)',(student['id'],path,NOW(),label))
                uploaded+=1
            except ValueError: remove_image(path); errors+=1
        audit('dataset_import', f'roll_no={roll}; label={label}; uploaded={uploaded}; skipped={errors}')
        flash(f'Imported {uploaded} images; skipped {errors}. Retrain models.','success')
        return redirect(url_for('dataset'))
    return render_template('dataset.html')

@app.route('/train',methods=['POST'])
@admin_required
def train():
    try:
        message=train_models()
        audit('models_trained', message)
        flash(message,'success')
    except ValueError as e:
        flash(str(e),'error')
    return redirect(url_for('index'))

@app.route('/admin/users', methods=['GET', 'POST'])
@admin_required
def manage_users():
    if request.method == 'POST':
        try:
            user_id = int(request.form.get('user_id', ''))
            action = request.form.get('action', '')
        except ValueError:
            abort(400)
        if action not in ('promote', 'demote'):
            abort(400)
        with db() as c:
            user = c.execute('SELECT id, name, email, role FROM users WHERE id=?', (user_id,)).fetchone()
            if user is None:
                abort(404)
            # Do not let an admin accidentally demote their own active account.
            if action == 'demote' and session.get('user_id') == user_id:
                flash('You cannot demote the account you are currently using.', 'error')
                return redirect(url_for('manage_users'))
            new_role = 'admin' if action == 'promote' else 'user'
            c.execute('UPDATE users SET role=? WHERE id=?', (new_role, user_id))
        audit('user_role_changed', f'user_id={user_id}; role={new_role}')
        flash(f'{user["name"]} now has {new_role} access. They must log out and log in again for the change to apply.', 'success')
        return redirect(url_for('manage_users'))
    with db() as c:
        users = c.execute('SELECT id, name, email, role, created_at FROM users ORDER BY id DESC').fetchall()
    return render_template('admin_users.html', users=users)


@app.route('/identify',methods=['GET','POST'])
@login_required
def identify():
    result=None
    if request.method=='POST':
        path=None
        try:
            path=save_image(request.files.get('signature'))
            student,score,status,verification,reference=identify_image(path)
            with db() as c: c.execute('INSERT INTO identification_history(image_path,student_id,score,status,verification,created_at) VALUES(?,?,?,?,?,?)',(path,student['id'] if student else None,score,status,verification,NOW()))
            audit('identification_run', f'status={status}; score={score}')
            result={'path':path,'student':student,'score':score,'status':status,'verification':verification,'reference':reference}
        except ValueError as e: remove_image(path); flash(str(e),'error')
    return render_template('identify.html',result=result)

@app.route('/history')
@admin_required
def history():
    with db() as c: rows=c.execute('SELECT h.*,s.name,s.roll_no FROM identification_history h LEFT JOIN students s ON s.id=h.student_id ORDER BY h.id DESC LIMIT 500').fetchall()
    return render_template('history.html',rows=rows)

@app.route('/export/<kind>/<format>')
@admin_required
def export(kind,format):
    if kind not in ('students','history') or format not in ('csv','pdf'): abort(404)
    with db() as c:
        if kind=='students':
            rows=c.execute('SELECT id,name,roll_no,created_at FROM students ORDER BY id').fetchall()
            columns=['id','name','roll_no','created_at']
        else:
            rows=c.execute('SELECT id,student_id,score,status,verification,created_at FROM identification_history ORDER BY id').fetchall()
            columns=['id','student_id','score','status','verification','created_at']
    records=[[str(r[k] if r[k] is not None else '') for k in columns] for r in rows]
    output=io.BytesIO()
    if format=='csv':
        text=io.StringIO(); writer=csv.writer(text); writer.writerow(columns)
        for row in records: writer.writerow(["'"+v if v.startswith(('=','+','-','@')) else v for v in row])
        output.write(text.getvalue().encode('utf-8-sig')); mime='text/csv'
    else:
        doc=SimpleDocTemplate(output,pagesize=(842,595),leftMargin=28,rightMargin=28)
        styles=getSampleStyleSheet(); body=[Paragraph(f'Signature System — {kind.title()}',styles['Title']),Spacer(1,12)]
        table=Table([columns]+[r for r in records[:300]],repeatRows=1,colWidths=[(842-56)/len(columns)]*len(columns))
        table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dce6f7')),('FONTSIZE',(0,0),(-1,-1),7),('GRID',(0,0),(-1,-1),.25,colors.grey),('VALIGN',(0,0),(-1,-1),'TOP')]))
        body.append(table); doc.build(body); mime='application/pdf'
    output.seek(0)
    return send_file(output,mimetype=mime,as_attachment=True,download_name=f'{kind}.{format}')


@app.route('/analytics')
@admin_required
def analytics():
    with db() as c:
        daily=c.execute("SELECT substr(created_at,1,10) day, COUNT(*) count FROM identification_history GROUP BY day ORDER BY day DESC LIMIT 14").fetchall()
        outcomes=c.execute("SELECT status, COUNT(*) count FROM identification_history GROUP BY status ORDER BY count DESC").fetchall()
        labels=c.execute("SELECT label,COUNT(*) count FROM signatures GROUP BY label").fetchall()
        recent=c.execute('SELECT created_at,action,details,ip_address FROM audit_log ORDER BY id DESC LIMIT 30').fetchall()
    daily=list(reversed(daily))
    return render_template('analytics.html', daily=daily, outcomes=outcomes, labels=labels, recent=recent)

@app.route('/model-performance')
@admin_required
def model_performance():
    report=None; error=None
    try:
        rows=data_rows(); valid=[]
        for r in rows:
            if r['label'] != 'genuine': continue
            try: valid.append((r,features(r['image_path'])))
            except (ValueError,cv2.error): pass
        counts=Counter(r['student_id'] for r,f in valid)
        valid=[(r,f) for r,f in valid if counts[r['student_id']]>=3]
        ids=sorted(set(r['student_id'] for r,f in valid))
        if len(ids)<2: raise ValueError('Need at least 2 writers with 3 or more genuine signatures each to calculate cross-validated metrics.')
        X=np.array([f for r,f in valid]); y=np.array([r['student_id'] for r,f in valid])
        folds=min(5, min(Counter(y).values()))
        if folds<2: raise ValueError('Not enough samples per writer for cross-validation.')
        estimator=make_pipeline(StandardScaler(),SVC(kernel='rbf',class_weight='balanced'))
        cv=StratifiedKFold(n_splits=folds,shuffle=True,random_state=42)
        predicted=cross_val_predict(estimator,X,y,cv=cv)
        precision,recall,f1,_=precision_recall_fscore_support(y,predicted,average='macro',zero_division=0)
        cm=confusion_matrix(y,predicted,labels=ids)
        with db() as c:
            names={r['id']:f"{r['name']} ({r['roll_no']})" for r in c.execute('SELECT id,name,roll_no FROM students').fetchall()}
        report={'accuracy':round(float(accuracy_score(y,predicted))*100,2),'precision':round(float(precision)*100,2),'recall':round(float(recall)*100,2),'f1':round(float(f1)*100,2),'folds':folds,'samples':len(y),'labels':[names.get(i,str(i)) for i in ids],'matrix':cm.tolist(),'details':classification_report(y,predicted,labels=ids,target_names=[names.get(i,str(i)) for i in ids],zero_division=0,output_dict=True)}
    except ValueError as e: error=str(e)
    model_exists=(MODELS/'identification.joblib').exists()
    return render_template('model_performance.html',report=report,error=error,model_exists=model_exists)

@app.route('/audit-log')
@admin_required
def audit_log_page():
    with db() as c: rows=c.execute('SELECT * FROM audit_log ORDER BY id DESC LIMIT 300').fetchall()
    return render_template('audit_log.html',rows=rows)

@app.route('/about')
@login_required
def about(): return render_template('about.html')

@app.errorhandler(413)
def too_large(e): return 'Upload exceeds 8 MB',413

if __name__=='__main__':
    init_db()
    app.run(debug=os.environ.get('FLASK_DEBUG')=='1',host='127.0.0.1')
