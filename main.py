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
from kivy.core.window import Window

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
import sounddevice as sd

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
        Window.size = (500, 400)  # Set initial window size
        Window.minimum_width = 500  # Set minimum width
        Window.minimum_height = 400  # Set minimum height

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

        apps_layout = BoxLayout(orientation='vertical', size_hint=(1, 1))
        apps_scrollview = ScrollView(size_hint=(1, 1))
        apps_scrollview.add_widget(self.apps_grid)
        apps_layout.add_widget(apps_scrollview)

        self.refresh_apps_button = Button(text="Refresh Apps", size_hint=(1, None), height=44)
        self.refresh_apps_button.bind(on_press=self.refresh_applications)
        apps_layout.add_widget(self.refresh_apps_button)

        self.devices = self.get_input_devices()
        self.device_spinner = Spinner(
            text=self.settings.get('device1', 'Select Input Device 1'),
            values=[f"{index}: {name}" for index, name in self.devices],
            size_hint=(1, None),
            height=44
        )
        self.device_spinner.bind(text=self.on_setting_change)

        self.device_spinner2 = Spinner(
            text=self.settings.get('device2', 'Select Input Device 2'),
            values=[f"{index}: {name}" for index, name in self.devices],
            size_hint=(1, None),
            height=44
        )
        self.device_spinner2.bind(text=self.on_setting_change)

        format_bitrate_layout = BoxLayout(size_hint=(1, None), height=44, spacing=10)

        self.format_spinner = Spinner(
            text=self.settings.get('format', 'WAV'),
            values=('WAV', 'FLAC', 'OGG', 'MP3'),
            size_hint=(0.5, None),
            height=44
        )
        self.format_spinner.bind(text=self.on_setting_change)
        format_bitrate_layout.add_widget(self.format_spinner)

        self.bitrate_spinner = Spinner(
            text=self.settings.get('bitrate', '192k'),
            values=('128k', '192k', '256k', '320k'),
            size_hint=(0.5, None),
            height=44
        )
        self.bitrate_spinner.bind(text=self.on_setting_change)
        format_bitrate_layout.add_widget(self.bitrate_spinner)

        self.recording_mode_button = Button(
            text='Mode: Combined',
            size_hint=(1, None),
            height=44
        )
        self.recording_mode_button.bind(on_press=self.toggle_recording_mode)

        record_duration_layout = BoxLayout(size_hint=(1, None), height=44, spacing=10)

        duration_label = Label(
            text='Buffer Duration (sec):',
            size_hint=(0.5, 1),
            halign='left'
        )
        record_duration_layout.add_widget(duration_label)

        self.duration_input = TextInput(
            text=str(self.settings.get('duration', '10')),
            multiline=False,
            input_filter='float',
            size_hint=(0.5, None),
            height=44
        )
        self.duration_input.bind(text=self.on_setting_change)
        record_duration_layout.add_widget(self.duration_input)

        self.record_button = Button(
            text='Start Recording',
            size_hint=(0.5, None),
            height=44
        )
        self.record_button.bind(on_press=self.toggle_recording)
        record_duration_layout.add_widget(self.record_button)

        layout.add_widget(apps_layout)
        layout.add_widget(self.device_spinner)
        layout.add_widget(self.device_spinner2)
        layout.add_widget(format_bitrate_layout)
        layout.add_widget(self.recording_mode_button)
        layout.add_widget(record_duration_layout)

        Clock.schedule_interval(self.update_apps_list, 5)

        # Initialize apps visibility based on current mode
        is_separate = self.settings.get('separate_audio', False)
        self.recording_mode_button.text = 'Mode: Separate' if is_separate else 'Mode: Combined'
        self.apps_grid.opacity = 1 if is_separate else 0
        self.apps_grid.disabled = not is_separate
        self.refresh_apps_button.opacity = 1 if is_separate else 0
        self.refresh_apps_button.disabled = not is_separate

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
            'device1': self.device_spinner.text,
            'device2': self.device_spinner2.text,
            'duration': self.duration_input.text,
            'format': self.format_spinner.text,
            'bitrate': self.bitrate_spinner.text,
            'separate_audio': self.settings['separate_audio']
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

            # Add microphone as a separate option
            apps.insert(0, 'Microphone')

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
    
    def get_wasapi_loopback_devices(self):
        try:
            devices = sd.query_devices()
            loopback_devices = []
            for idx, dev in enumerate(devices):
                if 'hostapi' in dev and dev.get('max_input_channels', 0) > 0:
                    loopback_devices.append((idx, dev['name']))
            return loopback_devices
        except Exception as e:
            print(f"Error getting WASAPI loopback devices: {e}")
            return []

    def get_input_devices(self):
        devices = []
        self.device_indices = {}
        try:
            # Добавляем опцию "Отключено" как первое устройство
            devices.append((0, "Отключено"))
            
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

            # Получаем индексы обоих устройств
            device_index1 = None
            device_index2 = None

            if not self.device_spinner.text.startswith("0:"):
                try:
                    selected_number = int(self.device_spinner.text.split(':')[0])
                    device_index1 = self.device_indices.get(selected_number)
                except:
                    device_index1 = None

            if not self.device_spinner2.text.startswith("0:"):
                try:
                    selected_number = int(self.device_spinner2.text.split(':')[0])
                    device_index2 = self.device_indices.get(selected_number)
                except:
                    device_index2 = None

            apps_to_record = self.selected_apps

            for app in apps_to_record:
                if app != 'No Audio Applications Found':
                    self.frames[app] = deque(maxlen=int(44100 * buffer_duration / self.CHUNK))

            self.recording_threads = {}
            
            # Запускаем потоки для каждого устройства
            if device_index1 is not None:
                thread1 = threading.Thread(target=self.record_audio, args=(device_index1, 'device1'))
                thread1.daemon = True
                self.recording_threads['device1'] = thread1
                thread1.start()

            if device_index2 is not None:
                thread2 = threading.Thread(target=self.record_audio, args=(device_index2, 'device2'))
                thread2.daemon = True
                self.recording_threads['device2'] = thread2
                thread2.start()

            self.key_thread = threading.Thread(target=self.check_key)
            self.key_thread.daemon = True
            self.key_thread.start()

        except Exception as e:
            print(f"Error starting recording: {str(e)}")
            import traceback
            traceback.print_exc()

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

    def sanitize_filename(self, filename):
        # Remove or replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        return filename

    def save_current_buffer(self):
        if not self.frames:
            return

        buffer_duration = float(self.duration_input.text)
        recordings_folder = "Recordings"
        os.makedirs(recordings_folder, exist_ok=True)

        for app, frames in self.frames.items():
            current_frames = list(frames)
            
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
                app_name = app.split(":")[0]  # Get only the application name part
                safe_app_name = self.sanitize_filename(app_name)
                filename = os.path.join(recordings_folder, f"{safe_app_name}_recording_{timestamp}.{format_type.lower()}")

            # Also sanitize the temporary filename
            temp_wav = os.path.join(recordings_folder, f"temp_recording_{timestamp}_{self.sanitize_filename(app)}.wav")

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

                # Очистка бфера после успешного сохранения
                if self.settings['separate_audio'] or app == 'combined':
                  self.frames[app] = deque(maxlen=int(44100 * buffer_duration / self.CHUNK))


            except Exception as e:
                print(f"Error saving audio for {app}: {e}")
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)

    def record_audio(self, device_index, device_name):
        FORMAT = pyaudio.paFloat32
        CHANNELS = 1
        RATE = 44100

        try:
            stream = self.p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.CHUNK
            )
            self.streams[device_name] = stream

            print(f"Recording started for {device_name}. Press 'k' to save buffer.")

            while not self.stop_recording:
                try:
                    data = stream.read(self.CHUNK, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=np.float32)
                    audio_data = np.array(audio_data, copy=True)
                    audio_data *= self.GAIN

                    if device_name not in self.frames:
                        self.frames[device_name] = deque(maxlen=int(44100 * float(self.duration_input.text) / self.CHUNK))
                    self.frames[device_name].append(audio_data)

                except Exception as e:
                    print(f"Error during recording: {e}")
                    break

        except Exception as e:
            print(f"Error setting up stream for {device_name}: {e}")

        print(f"Recording stopped for {device_name}")

    def get_wasapi_loopback_device(self):
        """Get the WASAPI loopback device index for system audio capture"""
        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if ('WASAPI' in device['name'] and 
                    device.get('max_input_channels', 0) > 0 and 
                    'loopback' in device['name'].lower()):
                    return i
                
            # Fallback: try to find any stereo mix device
            for i, device in enumerate(devices):
                if any(name in device['name'] for name in ['Stereo Mix', 'What U Hear', 'Stereo Mixer']):
                    return i
                
            # If still not found, try to find default Windows WASAPI device
            default_device = sd.default.device[0]  # Get default input device
            if default_device is not None:
                return default_device
                
        except Exception as e:
            print(f"Error finding WASAPI loopback device: {e}")
            
        print("Available audio devices:")
        print(sd.query_devices())
        return None

    def toggle_recording_mode(self, instance):
        if self.recording_mode_button.text == 'Mode: Combined':
            self.recording_mode_button.text = 'Mode: Separate'
            self.settings['separate_audio'] = True
            # Show apps selection
            self.apps_grid.opacity = 1
            self.apps_grid.disabled = False
            self.refresh_apps_button.opacity = 1
            self.refresh_apps_button.disabled = False
        else:
            self.recording_mode_button.text = 'Mode: Combined'
            self.settings['separate_audio'] = False
            # Hide apps selection
            self.apps_grid.opacity = 0
            self.apps_grid.disabled = True
            self.refresh_apps_button.opacity = 0
            self.refresh_apps_button.disabled = True
        self.save_settings()

    def on_stop(self):
        self.stop_recording_threads()
        self.p.terminate()

if __name__ == '__main__':
    AudioRecorderApp().run()