from __future__ import annotations

import threading
import time
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

        self.recognize_btn = ctk.CTkButton(self.left, text="Recognize Face", command=self.toggle_recognize)
        self.recognize_btn.pack(pady=8, padx=12, fill="x")

        self.api_btn = ctk.CTkButton(self.left, text="Ping API", command=self.ping_api)
        self.api_btn.pack(pady=8, padx=12, fill="x")

        self.status_label = ctk.CTkLabel(self.left, text="API: checking...")
        self.status_label.pack(pady=8, padx=12, anchor="w")

        self.enroll_label = ctk.CTkLabel(self.left, text=f"Enrollment faces: 0/{ENROLLMENT_FACE_TARGET}")
        self.enroll_label.pack(pady=8, padx=12, anchor="w")

        self.log_box = ctk.CTkTextbox(self.left, width=320, height=420)
        self.log_box.pack(pady=(8, 12), padx=12, fill="both", expand=True)

        import tkinter as tk
        self.video_frame = tk.Label(self, text="Camera feed will appear here", bg="#212121", fg="white", font=("Helvetica", 16))
        self.video_frame.pack(side="right", expand=True, fill="both", padx=(0, 12), pady=12)

        self.cap: cv2.VideoCapture | None = None
        self.running = False
        self.last_frame = None
        self.enrollment_mode = False
        self.enrollment_samples: list[bytes] = []

        self.recognize_mode = False
        self.recognize_in_progress = False
        self.last_recognized_name = ""
        self.last_recognized_box = None

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

    def detect_face(self, gray):
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
        if len(faces) > 0:
            return faces
        
        # Check for side profiles (OpenCV profile cascade only looks left)
        faces = self.profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(80, 80))
        if len(faces) > 0:
            return faces
        
        # Check flipped image for right side profiles
        gray_flipped = cv2.flip(gray, 1)
        faces_flipped = self.profile_cascade.detectMultiScale(gray_flipped, scaleFactor=1.1, minNeighbors=4, minSize=(80, 80))
        if len(faces_flipped) > 0:
            w_img = gray.shape[1]
            return [(w_img - x - w, y, w, h) for (x, y, w, h) in faces_flipped]
            
        return []

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.refresh_logs()

    def start_camera(self) -> None:
        if self.running or getattr(self, "camera_starting", False):
            return

        self.camera_starting = True
        self.start_btn.configure(text="Starting...", state="disabled")
        self.append_log("Initializing camera hardware...")
        threading.Thread(target=self._init_camera, daemon=True).start()

    def _init_camera(self) -> None:
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            
        if not self.cap.isOpened():
            self.after(0, lambda: self.append_log("Failed to open camera."))
            self.camera_starting = False
            self.after(0, lambda: self.start_btn.configure(text="Start Camera", state="normal"))
            return

        self.running = True
        self.camera_starting = False
        self.after(0, lambda: self.start_btn.configure(text="Start Camera", state="normal"))
        self.after(0, lambda: self.append_log("Camera started successfully."))
        self.after(0, self.update_frame)

    def stop_camera(self) -> None:
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.video_frame.configure(image="", text="Camera feed will appear here")
        self.append_log("Camera stopped.")

    def update_frame(self) -> None:
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if ret:
            if self.enrollment_mode:
                self.collect_face_sample(frame)
            elif self.recognize_mode:
                self.process_recognition(frame.copy())

            if self.recognize_mode and self.last_recognized_box is not None:
                x, y, w, h = self.last_recognized_box
                color = (0, 255, 120)
                length = int(w * 0.2)
                t = 2

                # Corner Brackets
                cv2.line(frame, (x, y), (x + length, y), color, t)
                cv2.line(frame, (x, y), (x, y + length), color, t)
                cv2.line(frame, (x + w, y), (x + w - length, y), color, t)
                cv2.line(frame, (x + w, y), (x + w, y + length), color, t)
                cv2.line(frame, (x, y + h), (x + length, y + h), color, t)
                cv2.line(frame, (x, y + h), (x, y + h - length), color, t)
                cv2.line(frame, (x + w, y + h), (x + w - length, y + h), color, t)
                cv2.line(frame, (x + w, y + h), (x + w, y + h - length), color, t)

                # Crosshair and Pulsing Target Ring
                cx, cy = x + w // 2, y + h // 2
                cv2.line(frame, (cx - 15, cy), (cx + 15, cy), color, 1)
                cv2.line(frame, (cx, cy - 15), (cx, cy + 15), color, 1)
                
                import math
                pulse = int(math.sin(time.time() * 5) * 8)
                cv2.circle(frame, (cx, cy), int(w * 0.35) + pulse, (0, 180, 50), 1)

                # Sci-Fi Data Readouts
                cv2.putText(frame, f"SYS.TRACK.ID: {id(self.last_recognized_box)%10000}", (x + w + 15, y + 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 200, 255), 1)
                cv2.putText(frame, f"W:{w} H:{h} T:{int(time.time()%100)}", (x + w + 15, y + 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 200, 255), 1)

                # Sweeping Scanner Line (Removed)

                if self.last_recognized_name:
                    is_unknown = "Unknown" in self.last_recognized_name
                    txt_color = (0, 0, 255) if is_unknown else color
                    bg_color = (30, 0, 0) if is_unknown else (0, 30, 0)
                    
                    text = f"TARGET: {self.last_recognized_name}"
                    txt_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)[0]
                    cv2.rectangle(frame, (x, y - 35), (x + txt_size[0] + 15, y - 5), bg_color, cv2.FILLED)
                    cv2.rectangle(frame, (x, y - 35), (x + txt_size[0] + 15, y - 5), txt_color, 1)
                    cv2.putText(frame, text, (x + 5, y - 12), cv2.FONT_HERSHEY_DUPLEX, 0.6, txt_color, 1)

            self.last_frame = frame.copy()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
            
            self.video_frame.configure(image=imgtk, text="")
            self.video_frame.image = imgtk

        self.after(30, self.update_frame)

    def map_face(self) -> None:
        if not self.running or self.last_frame is None:
            self.append_log("Start the camera before mapping a face.")
            return

        self.enrollment_samples = []
        self.enrollment_mode = True
        self.last_capture_time = 0
        self.enroll_label.configure(text=f"Enrollment faces: 0/{ENROLLMENT_FACE_TARGET}")
        self.append_log("Mapping started. Follow on-screen instructions.")

    def collect_face_sample(self, frame) -> None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detect_face(gray)
        if len(faces) == 0:
            return

        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        
        progress = len(self.enrollment_samples)
        if progress < 4:
            instruction = "LOOK STRAIGHT"
            color = (0, 150, 255)
        elif progress < 8:
            instruction = "TURN HEAD LEFT & RIGHT"
            color = (0, 255, 255)
        else:
            instruction = "TILT HEAD UP & DOWN"
            color = (0, 255, 150)

        cv2.circle(frame, (x + w // 2, y + h // 2), int(w * 0.6), color, 2)
        cv2.putText(frame, instruction, (x - 20, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"Capturing: {progress}/12", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        import time
        if time.time() - getattr(self, "last_capture_time", 0) > 0.4:
            self.last_capture_time = time.time()
            crop = frame[y : y + h, x : x + w]
            ok, encoded = cv2.imencode(".jpg", crop)
            if not ok:
                return

            if len(self.enrollment_samples) < ENROLLMENT_FACE_TARGET:
                self.enrollment_samples.append(encoded.tobytes())
                self.enroll_label.configure(text=f"Enrollment faces: {len(self.enrollment_samples)}/{ENROLLMENT_FACE_TARGET}")

            if len(self.enrollment_samples) >= ENROLLMENT_FACE_TARGET:
                self.enrollment_mode = False
                self.after(100, self.prompt_and_register_user)

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
        threading.Thread(target=self._fetch_logs_thread, daemon=True).start()
        self.after(REFRESH_INTERVAL_MS, self.refresh_logs)

    def _fetch_logs_thread(self) -> None:
        try:
            response = requests.get(f"{API_BASE}/logs/", timeout=3)
            response.raise_for_status()
            logs = response.json()
            self.after(0, lambda: self._update_logs_ui(logs))
        except Exception as exc:
            self.after(0, lambda: self._update_logs_ui_error(str(exc)))

    def _update_logs_ui(self, logs):
        self.status_label.configure(text="API: online")
        self.log_box.delete("1.0", "end")
        for log in logs[:20]:
            user_name = log.get("user_name") or "UNKNOWN"
            confidence = log.get("confidence", 0)
            timestamp = log.get("timestamp", "")
            self.log_box.insert("end", f"{timestamp} | {user_name} | conf={confidence:.3f}\n")

    def _update_logs_ui_error(self, err):
        self.status_label.configure(text="API: offline")
        self.append_log(f"API Error: {err}")

    def append_log(self, message: str) -> None:
        time_str = datetime.utcnow().strftime("%H:%M:%S")
        self.log_box.insert("end", f"[{time_str}] {message}\n")
        self.log_box.see("end")

    def toggle_recognize(self) -> None:
        if not self.running or self.last_frame is None:
            self.append_log("Start the camera before recognizing faces.")
            return

        self.recognize_mode = not self.recognize_mode
        if self.recognize_mode:
            self.recognize_btn.configure(text="Stop Recognition")
            self.last_recognized_name = ""
            self.last_recognized_box = None
            self.append_log("Real-time facial recognition started.")
        else:
            self.recognize_btn.configure(text="Recognize Face")
            self.last_recognized_box = None
            self.append_log("Real-time facial recognition stopped.")

    def process_recognition(self, frame) -> None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detect_face(gray)
        if len(faces) == 0:
            self.missed_frames = getattr(self, "missed_frames", 0) + 1
            if self.missed_frames > 20:
                self.last_recognized_box = None
            return

        self.missed_frames = 0

        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        self.last_recognized_box = (x, y, w, h)

        if getattr(self, "recognize_in_progress", False):
            return

        crop = frame[y : y + h, x : x + w]
        ok, encoded = cv2.imencode(".jpg", crop)
        if not ok:
            return

        self.recognize_in_progress = True
        threading.Thread(target=self._api_recognize, args=(encoded.tobytes(),), daemon=True).start()

    def _api_recognize(self, blob: bytes) -> None:
        try:
            response = requests.post(f"{API_BASE}/recognize/", files={"image": ("face.jpg", blob, "image/jpeg")}, timeout=3)
            response.raise_for_status()
            data = response.json()
            if data.get("label") == "MATCH":
                self.last_recognized_name = f"{data.get('name')} ({data.get('confidence'):.2f})"
            else:
                self.last_recognized_name = "Unknown"
        except Exception:
            self.last_recognized_name = ""
        finally:
            self.recognize_in_progress = False

    def on_close(self) -> None:
        self.stop_camera()
        self.destroy()


if __name__ == "__main__":
    app = OptaraApp()
    app.mainloop()
