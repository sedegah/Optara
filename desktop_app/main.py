from __future__ import annotations

import threading
from datetime import datetime

import customtkinter as ctk
import cv2
import requests
from PIL import Image, ImageTk

API_BASE = "http://127.0.0.1:8000/api"
REFRESH_INTERVAL_MS = 3000

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class OptaraApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Optara Facial Recognition")
        self.geometry("1000x600")

        self.left = ctk.CTkFrame(self, width=320)
        self.left.pack(side="left", fill="y", padx=12, pady=12)

        self.start_btn = ctk.CTkButton(self.left, text="Start Camera", command=self.start_camera)
        self.start_btn.pack(pady=(12, 8), padx=12, fill="x")

        self.stop_btn = ctk.CTkButton(self.left, text="Stop Camera", command=self.stop_camera)
        self.stop_btn.pack(pady=8, padx=12, fill="x")

        self.api_btn = ctk.CTkButton(self.left, text="Ping API", command=self.ping_api)
        self.api_btn.pack(pady=8, padx=12, fill="x")

        self.status_label = ctk.CTkLabel(self.left, text="API: checking...")
        self.status_label.pack(pady=8, padx=12, anchor="w")

        self.log_box = ctk.CTkTextbox(self.left, width=300, height=380)
        self.log_box.pack(pady=(8, 12), padx=12, fill="both", expand=True)

        self.video_frame = ctk.CTkLabel(self, text="Camera feed will appear here")
        self.video_frame.pack(side="right", expand=True, fill="both", padx=(0, 12), pady=12)

        self.cap: cv2.VideoCapture | None = None
        self.running = False

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.refresh_logs()

    # ---------------- Camera ----------------
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

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_frame.configure(image=imgtk, text="")
            self.video_frame.image = imgtk

    # ---------------- API ----------------
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
