# SignaLab — Signature Identification Research Prototype

## Features
Admin login, SQLite storage, writer enrollment, bulk upload, HOG+SVM writer classifier, separate experimental genuine/imitation classifier, unknown-writer heuristic, comparison display, history, dashboard, CSV/PDF exports, responsive professional UI, light/dark theme toggle, and live student search.

## Windows PowerShell setup (Python 3.12 recommended)

```powershell
cd "C:\Users\user\Downloads\signature_identification_project\signature_identification_project"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python create_admin.py
$env:FLASK_SECRET_KEY = python -c "import secrets; print(secrets.token_hex(32))"
python app.py
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` instead of `python` without activating. Keep `FLASK_SECRET_KEY` consistent across restarts; store it securely as an environment variable. Never commit secrets.

Open http://127.0.0.1:5000 and log in.

## Migrating your existing project

1. **Back up** the existing `app.py`, `signatures.db`, `static/signatures/`, and `templates/` first.
2. Copy this project's `app.py`, `create_admin.py`, `requirements.txt`, `templates/`, and `static/style.css` into your existing project folder. **Do not delete** the existing `signatures.db` or image folder.
3. Run `python create_admin.py` to initialize new tables and add an admin. The existing `students` and `signatures` tables are preserved, and `signatures.label` is added with default `genuine`.
4. Start `python app.py` and verify existing student records and images.
5. Add at least 3 **genuine** images for each of at least 2 writers, then click **Train / retrain models**.
6. For experimental forgery verification, also add clearly labeled imitations **of the target writer** (at least 3 negative comparisons). Retrain.

## Dataset without collecting your own

Look into public research datasets such as **CEDAR** (English signatures) and **BHSig260** (Bengali/Hindi signatures). Check their actual download access, licensing, and usage terms before using. This project does not bundle or download them. Keep the real writer identities consistent; never assign arbitrary generated signatures to one person. If using public data, register a writer ID as the roll number and bulk upload each writer's genuine and known imitation samples separately.

## Important model limitations

- The writer classifier is HOG + SVM. The rejection threshold is estimated from training samples, **not** a validated operating point. A prediction is not proof of identity.
- The verification classifier uses the absolute HOG difference between a genuine reference and a query, trained with same-writer genuine pairs (positive) and explicitly labeled imitation-vs-genuine pairs (negative). It is a classroom baseline, **not** a validated forgery detector.
- Train/test separation, per-writer holdout, dataset audits, precision/recall, false-accept and false-reject rates, and threshold calibration are required before presenting performance claims.
- Model files created by joblib must be treated as trusted local files; never load model files from untrusted users.
- Signature images are sensitive personal data. Obtain consent, secure backups, and restrict access. Production deployment needs HTTPS, rate-limited login, CSRF coverage for all routes, stronger access controls, and data retention policies.
- Training is synchronous and may take time on larger datasets. PDF export is capped at 300 rows, history page at 500 rows.

## Notes

The old `/student/<id>` and `/student/<id>/add-signature` routes are preserved. `init_db()` preserves existing tables. Use the included modern templates to match the new routes.


## UI improvements in this build
- Responsive navigation and card/table layouts for desktop and mobile.
- Professional SignaLab visual system with clearer hierarchy, status chips, and empty states.
- Light/dark theme toggle (preference saved in the browser).
- Dashboard student search filters by name or roll number without changing database records.
- Existing Flask routes and database schema are preserved.

## Analytics, model metrics, and security additions

This build adds three administrator pages to the navigation:

- **Analytics** (`/analytics`): daily recorded identification counts, outcomes, dataset label composition, and recent audit events. Charts use Chart.js from a CDN, so the charts require an internet connection; the underlying tables remain available if the CDN is blocked.
- **Model metrics** (`/model-performance`): calculates accuracy, macro precision/recall/F1, and a confusion matrix using stratified cross-validation on genuine signature samples currently stored in SQLite. It requires at least two writers with three genuine samples each. These are cross-validation estimates on the current dataset, not an independent benchmark and not evidence of forensic reliability.
- **Audit log** (`/audit-log`): records login successes/failures, logout, student registration/edit/deletion, signature deletion, model training, and identification runs. It shows recent events only.

Login rate limiting blocks further attempts from an IP after five recorded failures in a rolling 15-minute window. This is a lightweight single-instance safeguard, not a replacement for a production reverse proxy or centralized rate limiter. The app should remain bound to localhost for classroom demos. Use a strong persistent `FLASK_SECRET_KEY`; changing it logs users out.

### Upgrade notes

Back up `signatures.db`, `static/signatures/`, and `models/` before copying this version. Existing databases are migrated at startup with `CREATE TABLE IF NOT EXISTS` for the audit and login-attempt tables. No student records or administrator passwords are intentionally reset. The new navigation requires the matching `analytics.html`, `model_performance.html`, and `audit_log.html` templates from this package.


## Account registration and Google sign-in

- Visit `/signup` to create a regular user account with name, email, and a password (minimum 12 characters).
- Regular users can use the Identify page. Student management, dataset management, exports, training, analytics, and audit logs remain administrator-only.
- Admin accounts are still created with `python create_admin.py`; public registration never grants admin rights.
- Google sign-in uses Authlib and OpenID Connect. It is optional until configured.

### Configure Google OAuth

1. In Google Cloud Console, create/select a project and configure the OAuth consent screen.
2. Create an OAuth Client ID of type **Web application**.
3. Add this exact authorized redirect URI for local testing: `http://127.0.0.1:5000/auth/google/callback`. If using `http://localhost:5000`, add that callback separately and consistently use the same hostname.
4. Set environment variables in the same PowerShell window before starting Flask:

```powershell
$env:GOOGLE_CLIENT_ID = "your-client-id.apps.googleusercontent.com"
$env:GOOGLE_CLIENT_SECRET = "your-client-secret"
$env:FLASK_SECRET_KEY = "a-long-random-secret-kept-private"
.\.venv\Scripts\python.exe app.py
```

5. Open the site and click **Continue with Google**. Without both Google environment variables, the Google button is hidden and email/password login still works.

For deployment, register your production HTTPS callback URI in Google Cloud Console, use a persistent secret key, set `COOKIE_SECURE=1`, and never commit OAuth secrets to GitHub. Google sign-in creates only a regular user account; it cannot create or promote administrators.


## Admin access for a new signup

New email/password and Google accounts are regular users by default. Sign in using the existing admin account, open **Accounts** in the navigation (or visit `/admin/users`), and choose **Make admin** for the trusted account. The account owner must log out and log in again. Do not promote accounts you do not trust; admins can manage students, datasets, exports, model training, and other accounts.

Before replacing an existing local project, back up `signatures.db` and `static/signatures/`. Do not copy the sample ZIP over an existing database. Google sign-in is optional and requires `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET`.
