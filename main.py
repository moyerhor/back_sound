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
        Window.size = (500, 400)
        Window.minimum_width = 500
        Window.minimum_height = 400

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

        self.settings = self.load_settings()

        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

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

        layout.add_widget(self.device_spinner)
        layout.add_widget(self.device_spinner2)
        layout.add_widget(format_bitrate_layout)
        layout.add_widget(self.recording_mode_button)
        layout.add_widget(record_duration_layout)

        # Initialize recording mode
        is_separate = self.settings.get('separate_audio', False)
        self.recording_mode_button.text = 'Mode: Separate' if is_separate else 'Mode: Combined'

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

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        format_type = self.format_spinner.text
        bitrate = self.bitrate_spinner.text

        if self.settings['separate_audio']:
            # Separate mode: save each device separately
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

        else:
            # Combined mode: mix and save as one file
            if len(self.frames) == 2:
                frames1 = list(self.frames['device1'])
                frames2 = list(self.frames['device2'])
                
                if frames1 and frames2:
                    audio_data1 = np.concatenate(frames1)
                    audio_data2 = np.concatenate(frames2)
                    
                    # Ensure both arrays are the same length
                    min_length = min(len(audio_data1), len(audio_data2))
                    audio_data1 = audio_data1[:min_length]
                    audio_data2 = audio_data2[:min_length]
                    
                    # Mix the audio
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
        else:
            self.recording_mode_button.text = 'Mode: Combined'
            self.settings['separate_audio'] = False
        self.save_settings()

    def on_stop(self):
        self.stop_recording_threads()
        self.p.terminate()

if __name__ == '__main__':
    AudioRecorderApp().run()