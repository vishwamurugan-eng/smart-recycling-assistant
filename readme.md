# Smart Recycling Assistant ♻️

A supervised machine learning web application that classifies waste images into four categories (**Glass**, **Metal**, **Paper**, and **Plastic**) using **TensorFlow (MobileNetV2 Transfer Learning)**, evaluates model performance with **scikit-learn**, and logs dataset catalogs and live user predictions into a **MySQL** database via an interactive **Streamlit** dashboard.

## Project Architecture & Files

| File / Folder | Description |
| :--- | :--- |
| `app.py` | Interactive Streamlit web application with 3 tabs: Live Image Classification, `scikit-learn` Evaluation Metrics, and MySQL Database Explorer |
| `train_classifier.py` | Supervised training pipeline using TensorFlow/Keras (`MobileNetV2`) and `scikit-learn` (`classification_report`, `confusion_matrix`, `accuracy_score`) |
| `db_manager.py` | MySQL database connector that initializes `smart_recycling_db` and manages the `waste_images_catalog` and `prediction_history` tables |
| `artifacts/` | Saved TensorFlow model (`recycling_cnn_model.keras`), `scikit-learn` metrics (`metrics.json`), `confusion_matrix.png`, and `training_curves.png` |
| `requirements.txt` | Python package dependencies required to run the project |

## Technologies Used

* **Programming Language:** Python 3
* **Machine Learning & Deep Learning:** `tensorflow`, `scikit-learn`, `numpy`, `pandas`, `Pillow`, `matplotlib`
* **Database:** MySQL (`mysql-connector-python`)
* **Web Framework:** Streamlit (`streamlit`)

## How to Run the Project

### 1. Configure MySQL Database
Log in to MySQL (`mysql -u root -p`) and run:

```sql
CREATE DATABASE IF NOT EXISTS smart_recycling_db;
CREATE USER IF NOT EXISTS 'recycling_user'@'localhost' IDENTIFIED BY 'recycling_pass123';
GRANT ALL PRIVILEGES ON smart_recycling_db.* TO 'recycling_user'@'localhost';
FLUSH PRIVILEGES;
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Initialize Database & Run the Streamlit App
Since the pre-trained model and evaluation plots are already saved inside `artifacts/`, you can launch the Streamlit app directly:

```bash
python db_manager.py
streamlit run app.py
```

*(Optional)* To retrain the model from scratch on a `dataset/` folder containing `glass/`, `metal/`, `paper/`, and `plastic/` subfolders:

```bash
python train_classifier.py --dataset-dir dataset --epochs 10
```
