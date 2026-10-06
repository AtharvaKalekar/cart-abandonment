# 🚀 Streamlit App Deployment Guide (Case Study 59)

This guide provides step-by-step instructions for deploying your Cart Abandonment Intelligence Platform both **locally** (for presentation) and **live on Streamlit Community Cloud** (for sharing via link).

---

## Option 1: Live Cloud Deployment (Streamlit Community Cloud - FREE)

Streamlit Community Cloud allows you to host your app online for free with a public `https://share.streamlit.io` URL.

### Step 1: Push Project to GitHub
1. Make sure your repository contains the following files:
   - `cart-abandonment/app/streamlit_app.py`
   - `cart-abandonment/requirements.txt`
   - `cart-abandonment/models/` (pickle files `.joblib`)
   - `cart-abandonment/reports/` (CSV and JSON files)
   - `cart-abandonment/data/raw/online_shoppers_intention.csv`

2. Open terminal in project root and push to GitHub:
   ```bash
   git add .
   git commit -m "Deploy Cart Abandonment Streamlit App"
   git push origin main
   ```

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your **GitHub account**.
2. Click the **"New app"** button.
3. Fill in the deployment details:
   - **Repository:** Select your GitHub repo (e.g. `your-username/cart-abandonment`)
   - **Branch:** `main`
   - **Main file path:** `cart-abandonment/app/streamlit_app.py`
   - **App URL:** (Optional) Customize your custom sub-domain name.
4. Click **"Deploy!"**.
5. Streamlit will install dependencies from `requirements.txt` and launch your live app in ~2 minutes!

---

## Option 2: Local & Local Network Deployment (For Faculty Presentation)

If you are presenting live in class to your faculty, running locally or over local Wi-Fi is instant and requires no internet hosting.

### 1. Run Locally (Your Laptop)
Open terminal in your project directory and run:
```bash
source venv/bin/activate
streamlit run cart-abandonment/app/streamlit_app.py
```
* **URL:** `http://localhost:8501`

### 2. Share on Same Wi-Fi Network (Faculty/Classmates' Devices)
To let your faculty access the app on their own laptop/tablet connected to the same Wi-Fi network:
```bash
source venv/bin/activate
streamlit run cart-abandonment/app/streamlit_app.py --server.address 0.0.0.0
```
* **Network URL:** Look at the terminal output for `Network URL: http://<YOUR_IP_ADDRESS>:8501`. Share that link with your faculty!

---

## 🛠️ Requirements File Checklist
Ensure `cart-abandonment/requirements.txt` contains all required libraries:
```text
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
imbalanced-learn>=0.11.0
shap>=0.42.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0
streamlit>=1.25.0
joblib>=1.3.0
```
