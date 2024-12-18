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

# Configure logging
logging.basicConfig(level=logging.WARNING)
logging.getLogger('pycaw').setLevel(logging.WARNING)

# Helper functions
def get_app_name_from_pid(pid):
    try:
        process = psutil.Process(pid)
        return process.name()
    except psutil.NoSuchProcess:
        return "Unknown"

# UI Classes
class CheckBoxLabel(ToggleButtonBehavior, Label):
    pass

# Main Application Class
class AudioRecorderApp(App):
    # Constants
    CHUNK = 1024
    GAIN = 1.0

    # Initialization Methods
    def build(self):
        self._init_window()
        self._init_variables()
        self._init_audio()
        layout = self._create_layout()
        return layout

    def _init_window(self):
        Window.size = (500, 400)
        Window.minimum_width = 500
        Window.minimum_height = 400
        self.title = 'Запись Аудио'
        self.settings_file = 'recorder_settings.json'

    def _init_variables(self):
        self.is_recording = False
        self.frames = {}
        self.streams = {}
        self.recording_threads = {}
        self.key_thread = None
        self.stop_recording = False
        self.devices = []
        self.settings = self.load_settings()

    def _init_audio(self):
        self.p = pyaudio.PyAudio()
        self.devices = self.get_input_devices()

    # UI Creation Methods
    def _create_layout(self):
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Create device spinners
        self._create_device_spinners()
        
        # Create format and bitrate controls
        format_bitrate_layout = self._create_format_bitrate_layout()
        
        # Create recording mode button
        self._create_recording_mode_button()
        
        # Create duration controls
        record_duration_layout = self._create_duration_layout()
        
        # Create record button
        self._create_record_button()
        
        # Add all widgets to layout
        self._add_widgets_to_layout(layout, format_bitrate_layout, record_duration_layout)
        
        # Initialize recording mode
        self._init_recording_mode()
        
        return layout

    def _create_device_spinners(self):
        self.device_spinner = Spinner(
            text=self.settings.get('device1', 'Выберите устройство ввода 1'),
            values=[f"{index}: {name}" for index, name in self.devices],
            size_hint=(1, None),
            height=44
        )
        self.device_spinner.bind(text=self.on_setting_change)

        self.device_spinner2 = Spinner(
            text=self.settings.get('device2', 'Выберите устройство ввода 2'),
            values=[f"{index}: {name}" for index, name in self.devices],
            size_hint=(1, None),
            height=44
        )
        self.device_spinner2.bind(text=self.on_setting_change)

    def _create_format_bitrate_layout(self):
        layout = BoxLayout(size_hint=(1, None), height=44, spacing=10)
        
        self.format_spinner = Spinner(
            text=self.settings.get('format', 'WAV'),
            values=('WAV', 'FLAC', 'OGG', 'MP3'),
            size_hint=(0.5, None),
            height=44
        )
        self.format_spinner.bind(text=self.on_setting_change)
        
        self.bitrate_spinner = Spinner(
            text=self.settings.get('bitrate', '192k'),
            values=('128k', '192k', '256k', '320k'),
            size_hint=(0.5, None),
            height=44
        )
        self.bitrate_spinner.bind(text=self.on_setting_change)
        
        layout.add_widget(self.format_spinner)
        layout.add_widget(self.bitrate_spinner)
        
        return layout

    def _create_recording_mode_button(self):
        self.recording_mode_button = Button(
            text='Режим: Совместный',
            size_hint=(1, None),
            height=44
        )
        self.recording_mode_button.bind(on_press=self.toggle_recording_mode)

    def _create_duration_layout(self):
        layout = BoxLayout(size_hint=(1, None), height=44, spacing=10)
        
        duration_label = Label(
            text='Длительность:',
            size_hint=(0.3, 1),
            halign='left'
        )
        
        self.duration_input = TextInput(
            text=str(self.settings.get('duration', '10')),
            multiline=False,
            input_filter='float',
            size_hint=(0.3, None),
            height=44
        )
        self.duration_input.bind(text=self.on_setting_change)
        
        self.time_unit_spinner = Spinner(
            text=self.settings.get('time_unit', 'секунд'),
            values=('секунд', 'минут', 'часов'),
            size_hint=(0.3, None),
            height=44
        )
        self.time_unit_spinner.bind(text=self.on_setting_change)
        
        layout.add_widget(duration_label)
        layout.add_widget(self.duration_input)
        layout.add_widget(self.time_unit_spinner)
        
        return layout

    def get_buffer_duration_seconds(self):
        try:
            duration = float(self.duration_input.text)
            unit = self.time_unit_spinner.text
            
            # Convert to seconds based on unit
            if unit == 'минут':
                duration *= 60
            elif unit == 'часов':
                duration *= 3600
            
            # Ensure duration is not zero or negative
            if duration <= 0:
                return 10  # default value
            
            return duration
        except (ValueError, AttributeError):
            return 10  # default value if there's any error

    def _create_record_button(self):
        self.record_button = Button(
            text='Начать запись',
            size_hint=(1, None),
            height=44
        )
        self.record_button.bind(on_press=self.toggle_recording)

    def _add_widgets_to_layout(self, layout, format_bitrate_layout, record_duration_layout):
        layout.add_widget(self.device_spinner)
        layout.add_widget(self.device_spinner2)
        layout.add_widget(format_bitrate_layout)
        layout.add_widget(self.recording_mode_button)
        layout.add_widget(record_duration_layout)
        layout.add_widget(self.record_button)

    def _init_recording_mode(self):
        is_separate = self.settings.get('separate_audio', False)
        self.recording_mode_button.text = 'Режим: Раздельный' if is_separate else 'Режим: Совместный'

    # Settings Management
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
            'time_unit': self.time_unit_spinner.text,
            'format': self.format_spinner.text,
            'bitrate': self.bitrate_spinner.text,
            'separate_audio': self.settings['separate_audio']
        }
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Ошибка сохранения настроек: {e}")

    def on_setting_change(self, *args):
        self.save_settings()

    # Device Management
    def get_input_devices(self):
        devices = []
        self.device_indices = {}
        try:
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

    # Recording Control
    def toggle_recording(self, instance):
        if not self.is_recording:
            self.frames = {}
            self.start_recording()
        else:
            self.stop_recording_threads()

    def stop_recording_threads(self):
        self.stop_recording = True
        self.is_recording = False
        self.record_button.text = 'Начать запись'

        for app, stream in self.streams.items():
            if stream:
                stream.stop_stream()
                stream.close()
        self.streams = {}

    def start_recording(self):
        try:
            self.stop_recording = False
            self.is_recording = True
            self.record_button.text = 'Остановить запись'
            buffer_duration = float(self.duration_input.text)

            device_index1 = self._get_device_index(self.device_spinner.text)
            device_index2 = self._get_device_index(self.device_spinner2.text)

            self.recording_threads = {}
            
            if device_index1 is not None:
                self._start_recording_thread(device_index1, 'device1')

            if device_index2 is not None:
                self._start_recording_thread(device_index2, 'device2')

            self._start_key_thread()

        except Exception as e:
            print(f"Error starting recording: {str(e)}")
            import traceback
            traceback.print_exc()

    def _get_device_index(self, spinner_text):
        if not spinner_text.startswith("0:"):
            try:
                selected_number = int(spinner_text.split(':')[0])
                return self.device_indices.get(selected_number)
            except:
                return None
        return None

    def _start_recording_thread(self, device_index, device_name):
        thread = threading.Thread(target=self.record_audio, args=(device_index, device_name))
        thread.daemon = True
        self.recording_threads[device_name] = thread
        thread.start()

    def _start_key_thread(self):
        self.key_thread = threading.Thread(target=self.check_key)
        self.key_thread.daemon = True
        self.key_thread.start()

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

    # File Management
    def sanitize_filename(self, filename):
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

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        format_type = self.format_spinner.text
        bitrate = self.bitrate_spinner.text

        if self.settings['separate_audio']:
            self._save_separate_audio(recordings_folder, timestamp, format_type, bitrate)
        else:
            self._save_combined_audio(recordings_folder, timestamp, format_type, bitrate)

    def _save_separate_audio(self, recordings_folder, timestamp, format_type, bitrate):
        for device_name, frames in self.frames.items():
            current_frames = list(frames)
            if not current_frames:
                continue

            audio_data = np.concatenate(current_frames)
            filename = os.path.join(recordings_folder, f"{device_name}_{timestamp}.{format_type.lower()}")
            temp_wav = os.path.join(recordings_folder, f"temp_{device_name}_{timestamp}.wav")

            try:
                self.save_audio_file(audio_data, temp_wav, filename, format_type, bitrate)
                print(f"Saved {device_name} to {filename}")
            except Exception as e:
                print(f"Error saving audio for {device_name}: {e}")

    def _save_combined_audio(self, recordings_folder, timestamp, format_type, bitrate):
        if len(self.frames) == 2:
            frames1 = list(self.frames['device1'])
            frames2 = list(self.frames['device2'])
            
            if frames1 and frames2:
                audio_data1 = np.concatenate(frames1)
                audio_data2 = np.concatenate(frames2)
                
                min_length = min(len(audio_data1), len(audio_data2))
                audio_data1 = audio_data1[:min_length]
                audio_data2 = audio_data2[:min_length]
                
                combined_audio = (audio_data1 + audio_data2) / 2.0
                
                filename = os.path.join(recordings_folder, f"combined_{timestamp}.{format_type.lower()}")
                temp_wav = os.path.join(recordings_folder, f"temp_combined_{timestamp}.wav")
                
                try:
                    self.save_audio_file(combined_audio, temp_wav, filename, format_type, bitrate)
                    print(f"Saved combined audio to {filename}")
                except Exception as e:
                    print(f"Error saving combined audio: {e}")

    def save_audio_file(self, audio_data, temp_wav, filename, format_type, bitrate):
        try:
            with wave.open(temp_wav, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(44100)
                wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

            if format_type == 'WAV':
                os.replace(temp_wav, filename)
            elif format_type == 'MP3':
                audio = AudioSegment.from_wav(temp_wav)
                audio.export(filename, format="mp3", bitrate=bitrate)
            elif format_type == 'FLAC':
                sf.write(filename, audio_data, 44100, format='FLAC')
            elif format_type == 'OGG':
                sf.write(filename, audio_data, 44100, format='OGG')

            if format_type != 'WAV' and os.path.exists(temp_wav):
                os.remove(temp_wav)

        except Exception as e:
            raise Exception(f"Error in save_audio_file: {e}")

    # Audio Recording
    def record_audio(self, device_index, device_name):
        FORMAT = pyaudio.paFloat32
        CHANNELS = 1
        RATE = 44100

        try:
            stream = self._setup_audio_stream(device_index, FORMAT, CHANNELS, RATE)
            self.streams[device_name] = stream

            print(f"Recording started for {device_name}. Press 'k' to save buffer.")

            self._record_audio_loop(stream, device_name)

        except Exception as e:
            print(f"Error setting up stream for {device_name}: {e}")

        print(f"Recording stopped for {device_name}")

    def _setup_audio_stream(self, device_index, FORMAT, CHANNELS, RATE):
        return self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.CHUNK
        )

    def _record_audio_loop(self, stream, device_name):
        while not self.stop_recording:
            try:
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.float32)
                audio_data = np.array(audio_data, copy=True)
                audio_data *= self.GAIN

                if device_name not in self.frames:
                    buffer_duration = self.get_buffer_duration_seconds()
                    self.frames[device_name] = deque(maxlen=int(44100 * buffer_duration / self.CHUNK))
                self.frames[device_name].append(audio_data)

            except Exception as e:
                print(f"Error during recording: {e}")
                break

    # WASAPI Device Management
    def get_wasapi_loopback_device(self):
        try:
            devices = sd.query_devices()
            
            # Try to find WASAPI loopback device
            wasapi_device = self._find_wasapi_device(devices)
            if wasapi_device is not None:
                return wasapi_device
            
            # Try to find stereo mix device
            stereo_mix_device = self._find_stereo_mix_device(devices)
            if stereo_mix_device is not None:
                return stereo_mix_device
            
            # Fallback to default device
            return sd.default.device[0]
                
        except Exception as e:
            print(f"Error finding WASAPI loopback device: {e}")
            print("Available audio devices:")
            print(sd.query_devices())
            return None

    def _find_wasapi_device(self, devices):
        for i, device in enumerate(devices):
            if ('WASAPI' in device['name'] and 
                device.get('max_input_channels', 0) > 0 and 
                'loopback' in device['name'].lower()):
                return i
        return None

    def _find_stereo_mix_device(self, devices):
        for i, device in enumerate(devices):
            if any(name in device['name'] for name in ['Stereo Mix', 'What U Hear', 'Stereo Mixer']):
                return i
        return None

    # Mode Management
    def toggle_recording_mode(self, instance):
        if self.recording_mode_button.text == 'Режим: Совместный':
            self.recording_mode_button.text = 'Режим: Раздельный'
            self.settings['separate_audio'] = True
        else:
            self.recording_mode_button.text = 'Режим: Совместный'
            self.settings['separate_audio'] = False
        self.save_settings()

    # Cleanup
    def on_stop(self):
        self.stop_recording_threads()
        self.p.terminate()

if __name__ == '__main__':
    AudioRecorderApp().run()