# 🛒 GROCERA

## 📌 Project Overview

- GROCERA is a smart grocery inventory manager for tracking items, prices, quantity, and expiry dates.
- Uploads and processes `.csv` grocery data with predefined column formats.
- Provides a web interface using Flask for inventory interaction.
- Uses Google Generative AI for enhanced analysis or assistance features.
- Aims to minimize grocery wastage and improve stock efficiency.

---

## 🧰 Prerequisites

- Python 3.9 must be installed on your machine.
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


