# 🛒 GROCERA

## 📌 Project Overview

- **GROCERA** is a smart grocery inventory manager designed to track items, prices, quantity, and expiry dates.
- Allows users to upload and process `.csv` grocery data with predefined column formats.
- Offers a web-based UI using **Flask** for seamless inventory interaction.
- Integrates **Google Generative AI** for enhanced analytics and smart assistance.
- Aims to **minimize grocery wastage** and **optimize stock management**.
- ✅ **Live Demo:** [https://grocera-2.onrender.com](https://grocera-2.onrender.com)

---

## ▲ Deploy on Vercel

This project is now configured for Vercel with:
- [api/index.py](api/index.py) as the serverless entry point
- [vercel.json](vercel.json) route/build configuration

### Steps

1. Push the repository to GitHub.
2. Import the repository in Vercel.
3. Set these environment variables in Vercel Project Settings:
        - `SECRET_KEY` = strong random value
        - `FLASK_ENV` = `production`
        - `DEBUG` = `False`
        - `DATABASE_URL` = your production DB URL (recommended: managed PostgreSQL/MySQL)
        - `GROK_API_KEY` (if AI features are required)
        - `STRIPE_SECRET_KEY` (if payment features are required)
4. Deploy.

### Notes

- On Vercel, temporary writable storage is `/tmp`.
- If `DATABASE_URL` is missing, the app falls back to `sqlite:////tmp/grocera.db` for basic boot.
- For production, always use an external persistent database.

---

## 🧰 Prerequisites

- Python 3.9 must be installed on your machine. (https://www.python.org/downloads/release/python-390/)
- Ensure `pip` is available and functional (`pip --version` to check).
- PowerShell users (Windows) must bypass execution policy to activate virtualenv:

```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process

Clone the repository:
git clone <your-repo-url>
cd GROCERA-main

Create virtual environment using Python 3.9:
python3.9 -m venv myenv39

Activate the virtual environment:
myenv39\Scripts\activate
 for mac: source myenv39/bin/activate

Install required packages from requirements.txt:
pip install -r requirements.txt

If protobuf version causes errors, downgrade it:
pip install protobuf==3.20.3

Run the Flask application:
python app.py
# OR
flask run


Open the app in your browser:
http://127.0.0.1:xxxx/

                                            🤝 Credits
                                    Built by the GROCERA Team.


