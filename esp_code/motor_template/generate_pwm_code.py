import numpy as np
import csv
import subprocess

# Your measurements
pwm_data = np.array([255, 230])
time_data = np.array([20, 29])

# Linear regression: time = m * PWM + c
m, c = np.polyfit(pwm_data, time_data, 1)
print(f"Linear regression: time = {m:.4f}*PWM + {c:.4f}")

def pwm_for_flow(volume_ml, desired_time_s):
    scaled_time = desired_time_s * 200 / volume_ml
    pwm = (scaled_time - c) / m
    return int(max(0, min(255, round(pwm))))

# Set your desired flow
desired_ml = 200
desired_time = 60  # seconds
pwm_value = pwm_for_flow(desired_ml, desired_time)
print(f"Calculated PWM: {pwm_value}")

# Generate CSV
with open("pwm_flow.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Volume_ml","PWM_value","Time_s"])
    for vol in range(50, 251, 10):
        scaled_time = desired_time * 200 / vol
        pwm = pwm_for_flow(vol, desired_time)
        writer.writerow([vol, pwm, round(scaled_time, 2)])

# Replace PWM in template and save new file
template_file = "motor_template.ino"
output_file = "motor_auto.ino"
with open(template_file, "r") as f:
    code = f.read()
code = code.replace("PWM_PLACEHOLDER", str(pwm_value))
with open(output_file, "w") as f:
    f.write(code)
print(f"Updated ESP32 code saved as {output_file}")

# Compile and upload via Arduino CLI (update COM port)
fqbn = "esp32:esp32:esp32dev"
port = "COM10"  # replace with your ESP32 port

arduino_cli_path=r"C:\Users\Public\Desktop\Arduino IDE.lnk"

subprocess.run([arduino_cli_path, "compile", "--fqbn", fqbn, output_file])
subprocess.run([arduino_cli_path, "upload", "-p", port, "--fqbn", fqbn, output_file])
print("Code compiled and uploaded successfully!")
