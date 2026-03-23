from __future__ import annotations

import threading
from datetime import datetime

import customtkinter as ctk
import cv2
import requests
from PIL import Image, ImageTk

API_BASE = "http://127.0.0.1:8000/api"
REFRESH_INTERVAL_MS = 3000
ENROLLMENT_FACE_TARGET = 12

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class OptaraApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Optara Facial Recognition")
        self.geometry("1100x680")

        self.left = ctk.CTkFrame(self, width=340)
        self.left.pack(side="left", fill="y", padx=12, pady=12)

        self.start_btn = ctk.CTkButton(self.left, text="Start Camera", command=self.start_camera)
        self.start_btn.pack(pady=(12, 8), padx=12, fill="x")

        self.stop_btn = ctk.CTkButton(self.left, text="Stop Camera", command=self.stop_camera)
        self.stop_btn.pack(pady=8, padx=12, fill="x")

        self.map_btn = ctk.CTkButton(self.left, text="Map Face", command=self.map_face)
        self.map_btn.pack(pady=8, padx=12, fill="x")

        self.api_btn = ctk.CTkButton(self.left, text="Ping API", command=self.ping_api)
        self.api_btn.pack(pady=8, padx=12, fill="x")

        self.status_label = ctk.CTkLabel(self.left, text="API: checking...")
        self.status_label.pack(pady=8, padx=12, anchor="w")

        self.enroll_label = ctk.CTkLabel(self.left, text=f"Enrollment faces: 0/{ENROLLMENT_FACE_TARGET}")
        self.enroll_label.pack(pady=8, padx=12, anchor="w")

        self.log_box = ctk.CTkTextbox(self.left, width=320, height=420)
        self.log_box.pack(pady=(8, 12), padx=12, fill="both", expand=True)

        self.video_frame = ctk.CTkLabel(self, text="Camera feed will appear here")
        self.video_frame.pack(side="right", expand=True, fill="both", padx=(0, 12), pady=12)

        self.cap: cv2.VideoCapture | None = None
        self.running = False
        self.last_frame = None
        self.enrollment_mode = False
        self.enrollment_samples: list[bytes] = []

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.refresh_logs()

    def start_camera(self) -> None:
        if self.running:
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.append_log("Failed to open camera.")
            self.cap.release()
            self.cap = None
            return

        self.running = True
        self.append_log("Camera started.")
        threading.Thread(target=self.update_frame, daemon=True).start()

    def stop_camera(self) -> None:
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.append_log("Camera stopped.")

    def update_frame(self) -> None:
        while self.running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                continue

            self.last_frame = frame.copy()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
            self.video_frame.configure(image=imgtk, text="")
            self.video_frame.image = imgtk

            if self.enrollment_mode:
                self.collect_face_sample(frame)

    def map_face(self) -> None:
        if not self.running or self.last_frame is None:
            self.append_log("Start the camera before mapping a face.")
            return

        self.enrollment_samples = []
        self.enrollment_mode = True
        self.enroll_label.configure(text=f"Enrollment faces: 0/{ENROLLMENT_FACE_TARGET}")
        self.append_log("Mapping started. Look at camera until capture is complete.")

    def collect_face_sample(self, frame) -> None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
        if len(faces) == 0:
            return

        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        crop = frame[y : y + h, x : x + w]
        ok, encoded = cv2.imencode(".jpg", crop)
        if not ok:
            return

        if len(self.enrollment_samples) < ENROLLMENT_FACE_TARGET:
            self.enrollment_samples.append(encoded.tobytes())
            self.enroll_label.configure(text=f"Enrollment faces: {len(self.enrollment_samples)}/{ENROLLMENT_FACE_TARGET}")

        if len(self.enrollment_samples) >= ENROLLMENT_FACE_TARGET:
            self.enrollment_mode = False
            self.prompt_and_register_user()

    def prompt_and_register_user(self) -> None:
        dialog = ctk.CTkInputDialog(text="Enter name for this mapped face", title="Save Identity")
        name = dialog.get_input()
        if not name:
            self.append_log("Enrollment cancelled: no name provided.")
            return

        files = []
        for i, blob in enumerate(self.enrollment_samples):
            files.append(("images", (f"face_{i}.jpg", blob, "image/jpeg")))

        try:
            response = requests.post(
                f"{API_BASE}/register/",
                data={"name": name.strip()},
                files=files,
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
            self.append_log(
                f"Stored identity '{name}' with {payload.get('embedding_count', 0)} embeddings (user_id={payload.get('user_id')})."
            )
        except Exception as exc:
            self.append_log(f"Registration failed: {exc}")

    def ping_api(self) -> None:
        try:
            response = requests.get(f"{API_BASE}/logs/", timeout=3)
            response.raise_for_status()
            self.status_label.configure(text="API: online")
        except Exception as exc:
            self.status_label.configure(text="API: offline")
            self.append_log(f"API Error: {exc}")

    def refresh_logs(self) -> None:
        try:
            response = requests.get(f"{API_BASE}/logs/", timeout=5)
            response.raise_for_status()
            logs = response.json()
            self.status_label.configure(text="API: online")

            self.log_box.delete("1.0", "end")
            for log in logs[:20]:
                user_name = log.get("user_name") or "UNKNOWN"
                confidence = log.get("confidence", 0)
                timestamp = log.get("timestamp", "")
                self.log_box.insert("end", f"{timestamp} | {user_name} | conf={confidence:.3f}\n")
        except Exception as exc:
            self.status_label.configure(text="API: offline")
            self.append_log(f"API Error: {exc}")

        self.after(REFRESH_INTERVAL_MS, self.refresh_logs)

    def append_log(self, message: str) -> None:
        time_str = datetime.utcnow().strftime("%H:%M:%S")
        self.log_box.insert("end", f"[{time_str}] {message}\n")
        self.log_box.see("end")

    def on_close(self) -> None:
        self.stop_camera()
        self.destroy()


if __name__ == "__main__":
    app = OptaraApp()
    app.mainloop()
