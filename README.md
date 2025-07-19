# 🛒 GROCERA

## 📌 Project Overview

- **GROCERA** is a smart grocery inventory manager designed to track items, prices, quantity, and expiry dates.
- Allows users to upload and process `.csv` grocery data with predefined column formats.
- Offers a web-based UI using **Flask** for seamless inventory interaction.
- Integrates **Google Generative AI** for enhanced analytics and smart assistance.
- Aims to **minimize grocery wastage** and **optimize stock management**.
- ✅ **Live Demo:** [https://grocera-2.onrender.com](https://grocera-2.onrender.com)

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


