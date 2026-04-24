import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox, simpledialog

try:
    import face_recognition
except ModuleNotFoundError:
    face_recognition = None

try:
    from backend.recommendation_engine import generate_recommendation
    from backend.weather_api import get_weather
    from core.accessory_engine import AccessoryDetector
except ModuleNotFoundError:
    import sys

    ROOT_DIR = Path(__file__).resolve().parents[1]
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from backend.recommendation_engine import generate_recommendation
    from backend.weather_api import get_weather
    from core.accessory_engine import AccessoryDetector


APP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = APP_DIR / "data"
ENCODINGS_DIR = DATA_DIR / "encodings"
USERS_FILE = DATA_DIR / "registered_users.json"
ENCODINGS_DIR.mkdir(parents=True, exist_ok=True)
USERS_FILE.parent.mkdir(parents=True, exist_ok=True)

BG_PRIMARY = "#07111f"
BG_SIDEBAR = "#081427"
BG_PANEL = "#0f1d33"
BG_CARD = "#152846"
BG_SOFT = "#1b3559"
BG_BUTTON = "#203b5f"
ACCENT = "#5cd6ff"
ACCENT_STRONG = "#00a9e0"
SUCCESS = "#78f0b1"
WARNING = "#f7c97c"
DANGER = "#ff8e93"
TEXT_PRIMARY = "#e8f0fb"
TEXT_SECONDARY = "#9fb2c8"
TEXT_DIM = "#7f97b1"
CAMERA_SIZE = (960, 600)
SCAN_WINDOW_MS = 7000
RECOGNITION_STREAK = 6


def require_face_recognition():
    if face_recognition is None:
        raise ModuleNotFoundError(
            "face_recognition is not installed. Install it with 'pip install face-recognition' before running the app."
        )


def load_json_file(path, default):
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return default


def save_json_file(path, payload):
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


class FaceRegistry:
    def __init__(self):
        require_face_recognition()
        self.known_encodings = []
        self.known_names = []
        self.user_profiles = {}
        self.reload()

    def reload(self):
        self.known_encodings = []
        self.known_names = []
        self.user_profiles = load_json_file(USERS_FILE, {})

        for encoding_file in sorted(ENCODINGS_DIR.glob("*.npy")):
            try:
                encoding = np.load(encoding_file)
            except OSError:
                continue
            self.known_encodings.append(encoding)
            self.known_names.append(encoding_file.stem)

    def recognize(self, face_encoding):
        if not self.known_encodings:
            return "Unknown", None

        matches = face_recognition.compare_faces(self.known_encodings, face_encoding, tolerance=0.48)
        distances = face_recognition.face_distance(self.known_encodings, face_encoding)
        if len(distances) == 0:
            return "Unknown", None

        best_index = int(np.argmin(distances))
        if matches[best_index]:
            name = self.known_names[best_index]
            return name, self.user_profiles.get(name, {})
        return "Unknown", None

    def register(self, name, face_encoding):
        cleaned_name = name.strip()
        if not cleaned_name:
            raise ValueError("Name is required for registration.")

        np.save(ENCODINGS_DIR / f"{cleaned_name}.npy", face_encoding)
        profiles = load_json_file(USERS_FILE, {})
        profiles[cleaned_name] = {
            "name": cleaned_name,
            "registered_on": datetime.now().strftime("%d %b %Y, %I:%M %p"),
            "notes": "Registered from the desktop SmartWear camera with a clear face capture.",
        }
        save_json_file(USERS_FILE, profiles)
        self.reload()


class SmartWearApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SmartWear Advisor Desktop")
        self.root.configure(bg=BG_PRIMARY)
        self.root.geometry("1460x880")
        self.root.minsize(1280, 780)

        self.registry = FaceRegistry()
        self.accessory_detector = AccessoryDetector()

        self.cap = None
        self.running = False
        self.paused = False
        self.recognition_started_at = None
        self.detected_name = None
        self.recognition_streak = 0
        self.details_window = None
        self.latest_frame = None
        self.latest_face_box = None
        self.latest_face_encoding = None
        self.last_detection = None

        self.weather_city = tk.StringVar(value="Delhi")
        self.status_text = tk.StringVar(value="Ready when you are.")
        self.location_text = tk.StringVar(value="Using your chosen city for weather-aware guidance.")
        self.camera_state_text = tk.StringVar(value="Idle")
        self.identity_text = tk.StringVar(value="Awaiting face")
        self.accessory_text = tk.StringVar(value="Mask and glasses detection will appear here.")
        self.model_text = tk.StringVar(value=self.accessory_detector.status_summary())
        self.user_text = tk.StringVar(value="No registered user in focus.")
        self.weather_status_text = tk.StringVar(value="Weather pending")
        self.weather_meta_text = tk.StringVar(value="Humidity -- | Condition --")

        self.build_layout()
        self.refresh_weather()
        self.update_button_state()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def build_layout(self):
        shell = tk.Frame(self.root, bg=BG_PRIMARY)
        shell.pack(fill="both", expand=True)
        shell.grid_columnconfigure(0, weight=0)
        shell.grid_columnconfigure(1, weight=1)
        shell.grid_rowconfigure(0, weight=1)

        self.build_sidebar(shell)
        self.build_main(shell)

    def build_sidebar(self, parent):
        sidebar = tk.Frame(parent, bg=BG_SIDEBAR, width=290)
        sidebar.grid(row=0, column=0, sticky="nsw")
        sidebar.grid_propagate(False)

        brand = tk.Frame(sidebar, bg=BG_PANEL, highlightbackground="#233c61", highlightthickness=1)
        brand.pack(fill="x", padx=18, pady=(18, 14))
        tk.Label(brand, text="SMARTWEAR", font=("Segoe UI", 11, "bold"), fg=TEXT_DIM, bg=BG_PANEL).pack(anchor="w", padx=18, pady=(16, 2))
        tk.Label(brand, text="Advisor Desktop", font=("Segoe UI Semibold", 21), fg=TEXT_PRIMARY, bg=BG_PANEL).pack(anchor="w", padx=18, pady=(0, 16))

        for label in ("Scanner", "Desktop dashboard", "Weather-aware mode"):
            tk.Label(sidebar, text=label, font=("Segoe UI", 11), fg=TEXT_SECONDARY, bg=BG_SIDEBAR).pack(anchor="w", padx=24, pady=(4, 6))

        weather = tk.Frame(sidebar, bg=BG_PANEL, highlightbackground="#233c61", highlightthickness=1)
        weather.pack(fill="x", padx=18, pady=(18, 14))
        tk.Label(weather, text="Local weather", font=("Segoe UI", 10, "bold"), fg=TEXT_DIM, bg=BG_PANEL).pack(anchor="w", padx=18, pady=(16, 6))
        tk.Label(weather, textvariable=self.weather_status_text, font=("Segoe UI Semibold", 16), fg=TEXT_PRIMARY, bg=BG_PANEL, wraplength=220, justify="left").pack(anchor="w", padx=18)
        tk.Label(weather, textvariable=self.weather_meta_text, font=("Segoe UI", 10), fg=TEXT_SECONDARY, bg=BG_PANEL, wraplength=220, justify="left").pack(anchor="w", padx=18, pady=(8, 16))

        controls = tk.Frame(sidebar, bg=BG_PANEL, highlightbackground="#233c61", highlightthickness=1)
        controls.pack(fill="x", padx=18, pady=(0, 14))
        tk.Label(controls, text="Session controls", font=("Segoe UI Semibold", 14), fg=TEXT_PRIMARY, bg=BG_PANEL).pack(anchor="w", padx=18, pady=(16, 10))
        tk.Label(controls, text="Weather city", font=("Segoe UI", 10), fg=TEXT_SECONDARY, bg=BG_PANEL).pack(anchor="w", padx=18)
        self.city_entry = tk.Entry(
            controls,
            textvariable=self.weather_city,
            font=("Segoe UI", 11),
            bg=BG_SOFT,
            fg=TEXT_PRIMARY,
            insertbackground=TEXT_PRIMARY,
            relief="flat",
        )
        self.city_entry.pack(fill="x", padx=18, pady=(6, 10), ipady=8)
        self.city_entry.bind("<Return>", lambda _event: self.refresh_weather())

        self.start_button = self.make_button(controls, "Start scan", self.start_camera, fill=ACCENT)
        self.start_button.pack(fill="x", padx=18, pady=(0, 10))
        self.stop_button = self.make_button(controls, "Stop scan", self.stop_camera)
        self.stop_button.pack(fill="x", padx=18, pady=(0, 10))
        self.register_button = self.make_button(controls, "Register face", self.register_current_face, fill=SUCCESS)
        self.register_button.pack(fill="x", padx=18, pady=(0, 18))

    def build_main(self, parent):
        page = tk.Frame(parent, bg=BG_PRIMARY)
        page.grid(row=0, column=1, sticky="nsew", padx=(18, 20), pady=18)
        page.grid_columnconfigure(0, weight=5)
        page.grid_columnconfigure(1, weight=3)
        page.grid_rowconfigure(1, weight=1)

        header = tk.Frame(page, bg=BG_PRIMARY)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 16))
        tk.Label(header, text="LIVE SCANNER", font=("Segoe UI", 10, "bold"), fg=TEXT_DIM, bg=BG_PRIMARY).pack(anchor="w")
        tk.Label(header, text="Weather-aware face and accessory recognition", font=("Segoe UI Semibold", 28), fg=TEXT_PRIMARY, bg=BG_PRIMARY).pack(anchor="w", pady=(4, 0))

        hero = tk.Frame(page, bg=BG_PANEL, highlightbackground="#233c61", highlightthickness=1)
        hero.grid(row=1, column=0, sticky="nsew", padx=(0, 16))
        hero.grid_columnconfigure(0, weight=1)
        hero.grid_rowconfigure(1, weight=1)

        topcopy = tk.Frame(hero, bg=BG_PANEL)
        topcopy.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 8))
        tk.Label(topcopy, text="Desktop aligned with the browser flow", font=("Segoe UI", 10, "bold"), fg=ACCENT, bg=BG_PANEL).pack(anchor="w")
        tk.Label(topcopy, text="Keep your face inside the guide. If there is no match after a short scan, the app pauses safely and waits for registration.", font=("Segoe UI Semibold", 22), fg=TEXT_PRIMARY, bg=BG_PANEL, wraplength=700, justify="left").pack(anchor="w", pady=(12, 10))
        tk.Label(topcopy, text="Use the live camera to recognize registered users, check mask and glasses status, and open a weather-aware detail panel without leaving the desktop app.", font=("Segoe UI", 11), fg=TEXT_SECONDARY, bg=BG_PANEL, wraplength=720, justify="left").pack(anchor="w")

        status_grid = tk.Frame(hero, bg=BG_PANEL)
        status_grid.grid(row=1, column=0, sticky="nsew", padx=20, pady=(6, 20))
        for column in range(2):
            status_grid.grid_columnconfigure(column, weight=1)

        self.build_status_card(status_grid, "Recognition status", self.status_text, 0, 0)
        self.build_status_card(status_grid, "Location status", self.location_text, 0, 1)
        self.build_status_card(status_grid, "Camera state", self.camera_state_text, 1, 0)
        self.build_status_card(status_grid, "Model status", self.model_text, 1, 1)

        camera_panel = tk.Frame(page, bg=BG_PANEL, highlightbackground="#233c61", highlightthickness=1)
        camera_panel.grid(row=1, column=1, sticky="nsew")
        camera_panel.grid_rowconfigure(0, weight=1)
        camera_panel.grid_columnconfigure(0, weight=1)

        self.camera_label = tk.Label(
            camera_panel,
            bg="#040a12",
            fg=TEXT_PRIMARY,
            text="Camera preview will appear here",
            font=("Segoe UI", 16),
        )
        self.camera_label.grid(row=0, column=0, sticky="nsew", padx=18, pady=18)

        caption = tk.Frame(camera_panel, bg=BG_CARD, highlightbackground="#2b476f", highlightthickness=1)
        caption.grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 12))
        tk.Label(caption, textvariable=self.identity_text, font=("Segoe UI Semibold", 18), fg=TEXT_PRIMARY, bg=BG_CARD).pack(anchor="w", padx=16, pady=(14, 4))
        tk.Label(caption, textvariable=self.accessory_text, font=("Segoe UI", 10), fg=TEXT_SECONDARY, bg=BG_CARD, wraplength=360, justify="left").pack(anchor="w", padx=16, pady=(0, 14))

        notes = tk.Frame(camera_panel, bg=BG_PANEL)
        notes.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 18))
        tk.Label(notes, text="Register only when the current face is centered and visible without extra motion. The app pauses safely instead of scanning forever.", font=("Segoe UI", 10), fg=TEXT_SECONDARY, bg=BG_PANEL, wraplength=360, justify="left").pack(anchor="w")
        tk.Label(notes, textvariable=self.user_text, font=("Segoe UI", 10), fg=TEXT_PRIMARY, bg=BG_PANEL, wraplength=360, justify="left").pack(anchor="w", pady=(10, 0))

    def build_status_card(self, parent, title, text_var, row, column):
        card = tk.Frame(parent, bg=BG_CARD, highlightbackground="#2b476f", highlightthickness=1)
        card.grid(row=row, column=column, sticky="nsew", padx=(0, 12) if column == 0 else (0, 0), pady=(0, 12))
        tk.Label(card, text=title, font=("Segoe UI", 10), fg=TEXT_DIM, bg=BG_CARD).pack(anchor="w", padx=16, pady=(14, 4))
        tk.Label(card, textvariable=text_var, font=("Segoe UI Semibold", 14), fg=TEXT_PRIMARY, bg=BG_CARD, wraplength=320, justify="left").pack(anchor="w", padx=16, pady=(0, 14))

    def make_button(self, parent, label, command, fill=None):
        button_fill = fill or BG_BUTTON
        button_fg = "#04111d" if fill in (ACCENT, SUCCESS) else TEXT_PRIMARY
        return tk.Button(
            parent,
            text=label,
            command=command,
            font=("Segoe UI Semibold", 11),
            bg=button_fill,
            fg=button_fg,
            activebackground=ACCENT_STRONG if fill == ACCENT else button_fill,
            activeforeground=button_fg,
            relief="flat",
            bd=0,
            padx=12,
            pady=10,
            cursor="hand2",
        )

    def update_button_state(self):
        self.start_button.configure(state="disabled" if self.running and not self.paused else "normal")
        self.stop_button.configure(state="normal" if self.running else "disabled")
        self.register_button.configure(state="normal" if self.running or self.latest_frame is not None else "disabled")

    def refresh_weather(self):
        city = self.weather_city.get().strip() or "Delhi"
        weather = get_weather(city)
        self.weather_status_text.set(f"{weather.get('city', city)} | {weather.get('temp', '--')} C")
        self.weather_meta_text.set(f"Humidity {weather.get('humidity', '--')}% | {weather.get('description', weather.get('condition', 'Unavailable'))}")

    def start_camera(self):
        if self.running and not self.paused:
            return

        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Unable to access the webcam. Please check camera permissions.")
            self.cap = None
            return

        self.running = True
        self.paused = False
        self.recognition_started_at = datetime.now()
        self.detected_name = None
        self.recognition_streak = 0
        self.latest_face_box = None
        self.latest_face_encoding = None
        self.camera_state_text.set("Scanning")
        self.status_text.set("Camera live. Hold your face inside the guide.")
        self.location_text.set(f"Weather locked to {self.weather_city.get().strip() or 'Delhi'} for recommendations.")
        self.update_button_state()
        self.update_frame()

    def stop_camera(self):
        self.running = False
        self.paused = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.camera_state_text.set("Stopped")
        self.status_text.set("Camera stopped. Nothing is being processed right now.")
        self.update_button_state()

    def pause_scan(self):
        self.running = False
        self.paused = True
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.camera_state_text.set("Paused")
        self.status_text.set("No registered match found after the scan window. Register this face or start again.")
        self.update_button_state()

    def update_frame(self):
        if not self.running or self.cap is None:
            return

        ok, frame = self.cap.read()
        if not ok:
            self.status_text.set("Camera frame unavailable. Check the device and retry.")
            self.stop_camera()
            return

        frame = cv2.flip(frame, 1)
        self.latest_frame = frame.copy()
        display_frame = frame.copy()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        face_locations = sorted(
            face_locations,
            key=lambda face: (face[2] - face[0]) * (face[1] - face[3]),
            reverse=True,
        )

        if face_locations:
            primary_face = face_locations[0]
            encodings = face_recognition.face_encodings(rgb_frame, [primary_face])
            if encodings:
                face_encoding = encodings[0]
                self.latest_face_box = primary_face
                self.latest_face_encoding = face_encoding

                name, profile = self.registry.recognize(face_encoding)
                accessory_status = self.accessory_detector.analyze(frame, primary_face)
                self.last_detection = {
                    "name": name,
                    "profile": profile or {},
                    "accessories": accessory_status,
                }

                self.identity_text.set(name if name != "Unknown" else "Unknown")
                self.accessory_text.set(
                    f"Mask: {accessory_status['mask']} | Glasses: {accessory_status['glasses']} | "
                    f"Glasses confidence: {accessory_status.get('glasses_confidence', 0.0):.2f}"
                )

                if name != "Unknown":
                    self.user_text.set(
                        f"{name}\nRegistered on: {profile.get('registered_on', 'Unknown')}\n"
                        f"{profile.get('notes', 'No notes available')}"
                    )
                    self.camera_state_text.set("Recognized")
                    self.status_text.set("Registered face recognized. Opening the detail view shortly.")
                    self.recognition_streak = self.recognition_streak + 1 if self.detected_name == name else 1
                    self.detected_name = name
                    self.draw_detection_overlay(display_frame, primary_face, name, True)
                    if self.recognition_streak >= RECOGNITION_STREAK:
                        self.show_details_screen(name, profile or {}, accessory_status)
                        return
                else:
                    self.detected_name = None
                    self.recognition_streak = 0
                    self.camera_state_text.set("Scanning")
                    self.status_text.set("Face detected, but it is not registered yet.")
                    self.user_text.set("Unknown user. Use Register once to save this face for future scans.")
                    self.draw_detection_overlay(display_frame, primary_face, "Unknown", False)
            else:
                self.handle_no_face_state("Face detected, but the encoding could not be created.")
        else:
            self.handle_no_face_state("No face detected. Center your face inside the guide.")

        if self.recognition_started_at is not None:
            elapsed_ms = int((datetime.now() - self.recognition_started_at).total_seconds() * 1000)
            if self.detected_name is None and elapsed_ms >= SCAN_WINDOW_MS:
                self.render_frame(display_frame)
                self.pause_scan()
                return

        self.draw_ellipse_guide(display_frame)
        self.render_frame(display_frame)
        self.root.after(15, self.update_frame)

    def draw_detection_overlay(self, frame, face_box, label, recognized):
        top, right, bottom, left = face_box
        color = (120, 240, 177) if recognized else (247, 201, 124)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, max(0, top - 34)), (right, top), color, -1)
        cv2.putText(frame, label, (left + 8, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 15, 22), 2)

    def draw_ellipse_guide(self, frame):
        height, width = frame.shape[:2]
        overlay = frame.copy()
        center = (width // 2, height // 2)
        axes = (int(width * 0.18), int(height * 0.34))
        cv2.ellipse(overlay, center, axes, 0, 0, 360, (137, 160, 183), 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.98, frame, 0.02, 0, frame)

    def handle_no_face_state(self, status_message):
        self.detected_name = None
        self.recognition_streak = 0
        self.latest_face_box = None
        self.latest_face_encoding = None
        self.status_text.set(status_message)
        self.camera_state_text.set("Scanning" if self.running else "Idle")
        self.identity_text.set("Awaiting face")
        self.accessory_text.set("Mask and glasses detection will appear here.")
        self.user_text.set("No registered user in focus.")

    def render_frame(self, frame):
        display_frame = cv2.resize(frame, CAMERA_SIZE, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(image=image)
        self.camera_label.configure(image=photo, text="")
        self.camera_label.image = photo

    def register_current_face(self):
        if self.latest_frame is None or self.latest_face_box is None:
            if not self.running:
                self.start_camera()
            messagebox.showinfo("Register face", "Hold one clear face inside the guide, then press Register again.")
            return

        if self.latest_face_encoding is None:
            messagebox.showwarning("Register face", "A clear face encoding is not ready yet. Try again with better lighting.")
            return

        name = simpledialog.askstring("Register Face", "Enter the user's name:", parent=self.root)
        if not name:
            return

        accessories = self.accessory_detector.analyze(self.latest_frame, self.latest_face_box)
        if accessories.get("mask") != "No Mask" or accessories.get("glasses") != "No Glasses":
            messagebox.showwarning(
                "Register Face",
                "Registration works best with no mask or glasses. Please remove accessories and try again.",
            )
            return

        self.registry.register(name.strip(), self.latest_face_encoding)
        self.status_text.set(f"{name.strip()} registered successfully.")
        self.identity_text.set(name.strip())
        self.user_text.set(
            f"{name.strip()}\nRegistered on: {datetime.now().strftime('%d %b %Y, %I:%M %p')}\n"
            "Profile captured from the desktop live scanner."
        )
        messagebox.showinfo("Registration complete", f"{name.strip()} has been registered successfully.")
        self.paused = True
        self.running = False
        self.camera_state_text.set("Stopped")
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.update_button_state()

    def show_details_screen(self, name, profile, accessories):
        self.stop_camera()

        if self.details_window is not None and self.details_window.winfo_exists():
            self.details_window.destroy()

        weather = get_weather(self.weather_city.get().strip() or "Delhi")
        recommendations = generate_recommendation(
            weather=weather,
            is_mask=accessories.get("mask") == "Mask",
            is_glasses=accessories.get("glasses") == "Glasses",
        )

        detail_window = tk.Toplevel(self.root, bg=BG_PRIMARY)
        detail_window.title(f"{name} | SmartWear detail")
        detail_window.geometry("1080x760")
        detail_window.minsize(980, 700)
        self.details_window = detail_window

        header = tk.Frame(detail_window, bg=BG_PRIMARY)
        header.pack(fill="x", padx=24, pady=(22, 16))
        tk.Label(header, text="Identity snapshot", font=("Segoe UI", 10, "bold"), fg=TEXT_DIM, bg=BG_PRIMARY).pack(anchor="w")
        tk.Label(header, text=f"{name}", font=("Segoe UI Semibold", 30), fg=TEXT_PRIMARY, bg=BG_PRIMARY).pack(anchor="w", pady=(4, 0))
        tk.Label(header, text="Desktop detail view with accessory readout and weather-aware recommendations.", font=("Segoe UI", 11), fg=TEXT_SECONDARY, bg=BG_PRIMARY).pack(anchor="w", pady=(6, 0))

        grid = tk.Frame(detail_window, bg=BG_PRIMARY)
        grid.pack(fill="both", expand=True, padx=24, pady=(0, 20))
        for column in range(2):
            grid.grid_columnconfigure(column, weight=1)

        self.detail_card(grid, "User details", [
            f"Name: {name}",
            f"Registered on: {profile.get('registered_on', 'Unknown')}",
            f"Profile notes: {profile.get('notes', 'No notes saved')}",
        ], 0, 0)
        self.detail_card(grid, "Accessory readout", [
            f"Mask: {accessories.get('mask', 'Unknown')}",
            f"Glasses: {accessories.get('glasses', 'Unknown')}",
            f"Mask confidence: {accessories.get('mask_confidence', 0.0):.2f}",
            f"Glasses confidence: {accessories.get('glasses_confidence', 0.0):.2f}",
        ], 0, 1)
        self.detail_card(grid, "Weather snapshot", [
            f"Location: {weather.get('city', 'Unknown')}",
            f"Temperature: {weather.get('temp', '--')} C",
            f"Condition: {weather.get('description', weather.get('condition', 'Unknown'))}",
            f"Humidity: {weather.get('humidity', '--')}%",
        ], 1, 0)
        self.detail_card(grid, "Recommendations", recommendations, 1, 1)

        footer = tk.Frame(detail_window, bg=BG_PRIMARY)
        footer.pack(fill="x", padx=24, pady=(0, 24))
        self.make_button(footer, "Return to live scanner", self.restart_session, fill=ACCENT).pack(side="left")

    def detail_card(self, parent, title, lines, row, column):
        card = tk.Frame(parent, bg=BG_CARD, highlightbackground="#2b476f", highlightthickness=1)
        card.grid(row=row, column=column, sticky="nsew", padx=(0, 16) if column == 0 else (0, 0), pady=(0, 16))
        tk.Label(card, text=title, font=("Segoe UI Semibold", 15), fg=TEXT_PRIMARY, bg=BG_CARD).pack(anchor="w", padx=18, pady=(18, 12))
        for line in lines:
            tk.Label(card, text=line, font=("Segoe UI", 11), fg=TEXT_SECONDARY if title != "Recommendations" else TEXT_PRIMARY, bg=BG_CARD, wraplength=420, justify="left").pack(anchor="w", padx=18, pady=(0, 10))

    def restart_session(self):
        if self.details_window is not None and self.details_window.winfo_exists():
            self.details_window.destroy()
        self.details_window = None
        self.refresh_weather()
        self.start_camera()

    def on_close(self):
        self.stop_camera()
        try:
            cv2.destroyAllWindows()
        finally:
            self.root.destroy()


def run():
    require_face_recognition()
    root = tk.Tk()
    SmartWearApp(root)
    root.mainloop()


if __name__ == "__main__":
    run()
