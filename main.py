from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.spinner import Spinner
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.uix.textinput import TextInput
from kivy.uix.checkbox import CheckBox, ToggleButtonBehavior
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView

import pyaudio
import wave
import threading
import numpy as np
import soundfile as sf
import os
import json
from pydub import AudioSegment
import keyboard
from collections import deque
from datetime import datetime
import psutil
from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
import time
import win32gui
import win32process
import locale
import chardet
import ctypes
import logging

logging.basicConfig(level=logging.WARNING)
logging.getLogger('pycaw').setLevel(logging.WARNING)

def get_app_name_from_pid(pid):
    try:
        process = psutil.Process(pid)
        return process.name()
    except psutil.NoSuchProcess:
        return "Unknown"

class CheckBoxLabel(ToggleButtonBehavior, Label):
    pass

class AudioRecorderApp(App):
    CHUNK = 1024
    GAIN = 1.0

    def build(self):
        self.title = 'Audio Recorder'
        self.settings_file = 'recorder_settings.json'

        self.is_recording = False
        self.frames = {}
        self.p = pyaudio.PyAudio()
        self.streams = {}
        self.recording_threads = {}
        self.key_thread = None
        self.stop_recording = False
        self.devices = []
        self.selected_apps = []

        self.settings = self.load_settings()

        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        self.apps_grid = GridLayout(cols=1, size_hint_y=None)
        self.apps_grid.bind(minimum_height=self.apps_grid.setter('height'))

        self.refresh_apps_button = Button(text="Refresh Apps", size_hint=(1, None), height=44)
        self.refresh_apps_button.bind(on_press=self.refresh_applications)
        
        apps_layout = BoxLayout(orientation='vertical', size_hint=(1, 1))
        apps_scrollview = ScrollView(size_hint=(1, 1))
        apps_scrollview.add_widget(self.apps_grid)
        apps_layout.add_widget(apps_scrollview)
        apps_layout.add_widget(self.refresh_apps_button)

        self.devices = self.get_input_devices()
        self.device_spinner = Spinner(
            text=self.settings.get('device', 'Select Input Device'),
            values=[f"{index}: {name}" for index, name in self.devices],
            size_hint=(1, None),
            height=44
        )
        self.device_spinner.bind(text=self.on_setting_change)

        duration_layout = BoxLayout(size_hint=(1, None), height=44)
        duration_layout.add_widget(Label(text='Buffer Duration (sec):'))
        self.duration_input = TextInput(
            text=str(self.settings.get('duration', '10')),
            multiline=False,
            input_filter='float',
            size_hint=(0.7, None),
            height=44
        )
        self.duration_input.bind(text=self.on_setting_change)
        duration_layout.add_widget(self.duration_input)

        self.format_spinner = Spinner(
            text=self.settings.get('format', 'WAV'),
            values=('WAV', 'FLAC', 'OGG', 'MP3'),
            size_hint=(1, None),
            height=44
        )
        self.format_spinner.bind(text=self.on_setting_change)

        self.bitrate_spinner = Spinner(
            text=self.settings.get('bitrate', '192k'),
            values=('128k', '192k', '256k', '320k'),
            size_hint=(1, None),
            height=44
        )
        self.bitrate_spinner.bind(text=self.on_setting_change)

        self.record_button = Button(
            text='Start Recording',
            size_hint=(1, None),
            height=50
        )
        self.record_button.bind(on_press=self.toggle_recording)

        self.separate_audio_checkbox = CheckBox(active=self.settings.get('separate_audio', False), size_hint=(None, None), size=(44, 44))
        self.separate_audio_checkbox.bind(active=self.on_setting_change)

        separate_audio_layout = BoxLayout(size_hint=(1, None), height=44)
        separate_audio_layout.add_widget(Label(text='Separate Audio: '))
        separate_audio_layout.add_widget(self.separate_audio_checkbox)

        layout.add_widget(apps_layout)
        layout.add_widget(self.device_spinner)
        layout.add_widget(duration_layout)
        layout.add_widget(self.format_spinner)
        layout.add_widget(self.bitrate_spinner)
        layout.add_widget(separate_audio_layout)
        layout.add_widget(self.record_button)

        Clock.schedule_interval(self.update_apps_list, 5)

        return layout

    def load_settings(self):
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading settings: {e}")
        return {}

    def save_settings(self):
        settings = {
            'device': self.device_spinner.text,
            'duration': self.duration_input.text,
            'format': self.format_spinner.text,
            'bitrate': self.bitrate_spinner.text,
            'separate_audio': self.separate_audio_checkbox.active
        }
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def on_setting_change(self, *args):
        self.save_settings()

    def refresh_applications(self, instance):
        self.update_apps_list()

    def update_apps_list(self, dt=0):
        original_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.WARNING)

        try:
            current_selected_apps = set(self.selected_apps)

            apps = self.get_running_applications_with_audio()
            self.apps_grid.clear_widgets()

            if not apps or apps == ['No Audio Applications Found']:
                self.apps_grid.add_widget(Label(text="No Audio Applications Found", size_hint_y=None, height=44))
                self.selected_apps = ['Microphone']
            else:
                for app in apps:
                    layout = BoxLayout(size_hint_y=None, height=44)
                    label = Label(text=app, size_hint_x=0.8)
                    checkbox = CheckBox(size_hint_x=0.2)
                    checkbox.bind(active=self.on_app_checkbox_update)

                    if app in current_selected_apps:
                        checkbox.active = True

                    layout.add_widget(label)
                    layout.add_widget(checkbox)
                    self.apps_grid.add_widget(layout)
        finally:
            logging.getLogger().setLevel(original_level)

    def on_app_checkbox_update(self, checkbox, value):
        if checkbox.parent:
            app_name = checkbox.parent.children[1].text
            if value:
                if app_name not in self.selected_apps:
                    self.selected_apps.append(app_name)
            else:
                if app_name in self.selected_apps:
                    self.selected_apps.remove(app_name)

    def get_running_applications_with_audio(self):
        apps = set()
        try:
            sessions = AudioUtilities.GetAllSessions()
            for session in sessions:
                if session.Process and session.Process.name():
                    app_name = session.Process.name()
                    pid = session.ProcessId
                    apps.add(f"{app_name}:{pid}")
        except Exception as e:
            print(f"Error getting audio sessions: {e}")

        return sorted(list(apps)) if apps else ['No Audio Applications Found']

    def get_input_devices(self):
        devices = []
        self.device_indices = {}
        try:
            info = self.p.get_host_api_info_by_index(0)
            num_devices = info.get('deviceCount')

            for i in range(num_devices):
                device_info = self.p.get_device_info_by_host_api_device_index(0, i)
                if device_info.get('maxInputChannels') > 0:
                    buffer = ctypes.create_string_buffer(256)
                    ctypes.windll.kernel32.WideCharToMultiByte(
                        0, 0, device_info.get('name'), -1, buffer, 256, None, None
                    )
                    device_name = buffer.value.decode('utf-8', 'ignore')
                    devices.append((i + 1, device_name))
                    self.device_indices[i + 1] = i

        except Exception as e:
            print(f"Error getting input devices: {e}")
            return ["No input devices found"]

        return devices

    def toggle_recording(self, instance):
        if not self.is_recording:
            if not self.selected_apps or self.selected_apps == ['Microphone']:
                self.selected_apps = ['Microphone']

            if not self.device_spinner.text or self.device_spinner.text == "No input devices found":
                print("Please select a valid input device.")
                return

            self.frames = {}
            self.start_recording()
        else:
            self.stop_recording_threads()

    def stop_recording_threads(self):
        self.stop_recording = True
        self.is_recording = False
        self.record_button.text = 'Start Recording'

        for app, stream in self.streams.items():
            if stream:
                stream.stop_stream()
                stream.close()
        self.streams = {}

    def start_recording(self):
        try:
            self.stop_recording = False
            self.is_recording = True
            self.record_button.text = 'Stop Recording'
            buffer_duration = float(self.duration_input.text)

            selected_number = int(self.device_spinner.text.split(':')[0])
            device_index = self.device_indices[selected_number]

            apps_to_record = self.selected_apps

            for app in apps_to_record:
                if app != 'No Audio Applications Found':
                    self.frames[app] = deque(maxlen=int(44100 * buffer_duration / self.CHUNK))


            self.recording_threads = {}
            for app in self.frames.keys():
                thread = threading.Thread(target=self.record_audio, args=(device_index, app))
                thread.daemon = True
                self.recording_threads[app] = thread
                thread.start()

            self.key_thread = threading.Thread(target=self.check_key)
            self.key_thread.daemon = True
            self.key_thread.start()

        except Exception as e:
            print(f"Error starting recording: {e}")

    def check_key(self):
        last_save_time = 0
        min_interval = 0.1

        while self.is_recording:
            if keyboard.is_pressed('k'):
                current_time = time.time()
                if current_time - last_save_time >= min_interval:
                    self.save_current_buffer()
                    last_save_time = current_time
            time.sleep(0.05)

    def save_current_buffer(self):
        if not self.frames:
            return

        buffer_duration = float(self.duration_input.text)
        recordings_folder = "Recordings"
        os.makedirs(recordings_folder, exist_ok=True)

        for app, frames in self.frames.items():
            current_frames = list(frames)
            
            # Проверка наличия данных в буфере
            if not current_frames:
                print(f"No data to save for {app}")
                continue

            audio_data = np.concatenate(current_frames)
            format_type = self.format_spinner.text
            bitrate = self.bitrate_spinner.text
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if app == 'Microphone':
                filename = os.path.join(recordings_folder, f"microphone_recording_{timestamp}.{format_type.lower()}")
            elif app == 'combined':
                filename = os.path.join(recordings_folder, f"combined_recording_{timestamp}.{format_type.lower()}")
            else:
                app_name = app.split(":")[0]
                filename = os.path.join(recordings_folder, f"{app_name}_recording_{timestamp}.{format_type.lower()}")

            temp_wav = os.path.join(recordings_folder, f"temp_recording_{timestamp}_{app}.wav")

            try:
                with wave.open(temp_wav, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(44100)
                    wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

                if format_type == 'WAV':
                    os.replace(temp_wav, filename)
                elif format_type == 'MP3':
                    try:
                        audio = AudioSegment.from_wav(temp_wav)
                        audio.export(
                            filename,
                            format="mp3",
                            bitrate=bitrate,
                            parameters=["-q:a", "0"]
                        )
                    except Exception as e:
                        print(f"Error converting to MP3: {e}")
                        try:
                            from pydub.utils import which
                            AudioSegment.converter = which("ffmpeg")
                            audio = AudioSegment.from_wav(temp_wav)
                            audio.export(filename, format="mp3", bitrate=bitrate)
                        except Exception as e2:
                            print(f"Alternative method also failed: {e2}")
                elif format_type == 'FLAC':
                    sf.write(filename, audio_data, 44100, format='FLAC')
                elif format_type == 'OGG':
                    sf.write(filename, audio_data, 44100, format='OGG')

                if format_type != 'WAV' and os.path.exists(temp_wav):
                    os.remove(temp_wav)

                print(f"Saved {app} to {filename}")
                duration = len(audio_data) / 44100
                print(f"Recording duration: {duration:.2f} seconds")

                # Очистка буфера после успешного сохранения
                if self.separate_audio_checkbox.active or app == 'combined':
                  self.frames[app] = deque(maxlen=int(44100 * buffer_duration / self.CHUNK))


            except Exception as e:
                print(f"Error saving audio for {app}: {e}")
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)

    def record_audio(self, device_index, app_name):
        FORMAT = pyaudio.paFloat32
        CHANNELS = 1
        RATE = 44100

        stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.CHUNK
        )

        self.streams[app_name] = stream

        print(f"Recording started for {app_name}. Press 'k' to save buffer.")

        while not self.stop_recording:
            try:
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.float32)

                audio_data = np.array(audio_data, copy=True)

                audio_data *= self.GAIN

                if self.is_recording:
                    if self.separate_audio_checkbox.active:
                        if app_name == "Microphone":
                            self.frames[app_name].append(audio_data)
                        else:
                            pid = int(app_name.split(":")[1])
                            hwnd = win32gui.GetForegroundWindow()
                            _, foreground_pid = win32process.GetWindowThreadProcessId(hwnd)
                            if pid == foreground_pid:
                                self.frames[app_name].append(audio_data)
                    else:
                        if 'combined' not in self.frames:
                            self.frames['combined'] = deque(maxlen=int(44100 * float(self.duration_input.text) / self.CHUNK))
                        self.frames['combined'].append(audio_data)

            except Exception as e:
                print(f"Error during recording for {app_name}: {e}")
                break

        print(f"Recording stopped for {app_name}")

    def on_stop(self):
        self.stop_recording_threads()
        self.p.terminate()

if __name__ == '__main__':
    AudioRecorderApp().run()