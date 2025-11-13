import time
import csv
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

url = "http://localhost:8501"  # Streamlit app
csv_file = "corrected_positions.csv"
updated_file = "images_metadata_with_labels.csv"

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)
driver.get(url)
wait = WebDriverWait(driver, 30)

updated_rows = []

with open(csv_file, "r") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames + ["predicted_label", "predicted_label_with_conf"]

    for row in reader:
        image_path = os.path.abspath(os.path.join("flight_data", row['image']))



        # Upload image
        upload = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']")))
        upload.send_keys(image_path)

        predictions, labels = [], []

        try:
            # Wait 2 seconds for final combined prediction to appear
            time.sleep(2)
            result_element = driver.find_element(By.ID, "prediction-text")
            result_text = result_element.text.strip()
            print("Raw result:", result_text)

            # If unknown → mark directly
            if "unknown" in result_text.lower():
                row["predicted_label"] = "unknown"
                row["predicted_label_with_conf"] = "unknown (0.00)"
                updated_rows.append(row)
                continue

            # Parse final combined prediction
            if "Predicted Labels:" in result_text:
                preds_text = result_text.split("Predicted Labels:")[-1].strip()
                predictions = [p.strip() for p in preds_text.split(";") if p.strip()]
                labels = [p.split()[0] for p in predictions]

        except Exception as e:
            print("❌ Could not get prediction:", e)
            row["predicted_label"] = "unknown"
            row["predicted_label_with_conf"] = "unknown (0.00)"
            updated_rows.append(row)
            continue

        # Save predictions
        row["predicted_label"] = ";".join(labels) if labels else "unknown"
        row["predicted_label_with_conf"] = ";".join(predictions) if predictions else "unknown (0.00)"
        updated_rows.append(row)

# Save updated CSV
with open(updated_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(updated_rows)

driver.quit()
print(f"✅ Updated CSV saved as {updated_file}")
