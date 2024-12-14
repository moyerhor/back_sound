from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.spinner import Spinner
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.uix.textinput import TextInput
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
from pycaw.pycaw import AudioUtilities
import time

class AudioRecorderApp(App):
    CHUNK = 1024  # Определяем CHUNK как константу класса
    
    def build(self):
        self.title = 'Audio Recorder'
        self.settings_file = 'recorder_settings.json'
        
        # Initialize variables
        self.is_recording = False
        self.frames = deque(maxlen=int(44100 * 10 / 1024))  # Buffer for 10 seconds
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.recording_thread = None
        self.key_thread = None
        self.stop_recording = False
        
        # Load saved settings
        self.settings = self.load_settings()
        
        # Main layout
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Get list of running applications
        apps = self.get_running_applications()
        
        # Spinner for selecting application
        self.app_spinner = Spinner(
            text=self.settings.get('application', 'Select Application'),
            values=apps,
            size_hint=(1, None),
            height=44
        )
        self.app_spinner.bind(text=self.on_setting_change)
        
        # Spinner for selecting device
        devices = self.get_input_devices()
        self.device_spinner = Spinner(
            text=self.settings.get('device', 'Select Input Device'),
            values=devices,
            size_hint=(1, None),
            height=44
        )
        self.device_spinner.bind(text=self.on_setting_change)
        
        # Buffer duration selection
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
        
        # Format selection
        self.format_spinner = Spinner(
            text=self.settings.get('format', 'WAV'),
            values=('WAV', 'FLAC', 'OGG', 'MP3'),
            size_hint=(1, None),
            height=44
        )
        self.format_spinner.bind(text=self.on_setting_change)
        
        # Bitrate selection
        self.bitrate_spinner = Spinner(
            text=self.settings.get('bitrate', '192k'),
            values=('128k', '192k', '256k', '320k'),
            size_hint=(1, None),
            height=44
        )
        self.bitrate_spinner.bind(text=self.on_setting_change)
        
        # Record button
        self.record_button = Button(
            text='Start Recording',
            size_hint=(1, None),
            height=50
        )
        self.record_button.bind(on_press=self.toggle_recording)
        
        # Add widgets to layout
        layout.add_widget(self.app_spinner)
        layout.add_widget(self.device_spinner)
        layout.add_widget(duration_layout)
        layout.add_widget(self.format_spinner)
        layout.add_widget(self.bitrate_spinner)
        layout.add_widget(self.record_button)
        
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
            'application': self.app_spinner.text,
            'device': self.device_spinner.text,
            'duration': self.duration_input.text,
            'format': self.format_spinner.text,
            'bitrate': self.bitrate_spinner.text
        }
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Error saving settings: {e}")
    
    def on_setting_change(self, *args):
        self.save_settings()
    
    def get_running_applications(self):
        apps = []
        try:
            # Получаем все активные аудио сессии
            sessions = AudioUtilities.GetAllSessions()
            for session in sessions:
                if session.Process and session.Process.name():
                    # Добавляем только приложения с активными аудио сессиями
                    app_name = session.Process.name()
                    if not app_name.lower().endswith('.exe'):
                        apps.append(app_name)
        except Exception as e:
            print(f"Error getting audio sessions: {e}")
            return ['No Audio Applications Found']
        
        unique_apps = sorted(list(set(apps)))
        return unique_apps if unique_apps else ['No Audio Applications Found']
    
    def get_input_devices(self):
        devices = []
        try:
            # Создаем словарь для хранения реальных индексов устройств
            real_indices = {}
            counter = 1  # Начинаем с 1
            
            # Получаем устройства через PyAudio
            for i in range(self.p.get_device_count()):
                device = self.p.get_device_info_by_index(i)
                if device['maxInputChannels'] > 0:
                    name = device['name']
                    # Пропускаем устройства с исключительно некорректными символами, если необходимо
                    # Уберите или измените следующую строку
                    # if any(char in name for char in 'РЂЃЉЊЋЌЍЎЏђѓљњћќѝўџ'):
                    #     continue
                    
                    # Проверяем на дубликаты
                    if not any(n == name for _, n in devices):
                        real_indices[counter] = i
                        devices.append((counter, name))
                        counter += 1

        except Exception as e:
            print(f"Error getting input devices: {e}")
            return ["No input devices found"]

        # Сохраняем словарь соответствия индексов как атрибут класса
        self.device_indices = real_indices
        
        # Форматируем список устройств
        return [f"{index}: {name}" for index, name in devices] if devices else ["No input devices found"]
    
    def toggle_recording(self, instance):
        if not self.is_recording:
            selected_app = self.app_spinner.text
            if selected_app == 'Select Application' or selected_app == 'No Applications Found':
                print("Пожалуйста, выберите действительное приложение для записи.")
                return
            # Полностью очищаем буфер перед началом новой записи
            buffer_duration = float(self.duration_input.text)
            self.frames = deque(maxlen=int(44100 * buffer_duration / self.CHUNK))
            self.start_recording()
        else:
            self.stop_recording = True
            self.is_recording = False
            self.record_button.text = 'Start Recording'
            # Очищаем буфер при остановке
            buffer_duration = float(self.duration_input.text)
            self.frames = deque(maxlen=int(44100 * buffer_duration / self.CHUNK))
    
    def start_recording(self):
        try:
            self.stop_recording = False
            self.is_recording = True
            self.record_button.text = 'Stop Recording'
            # Создаем новый пустой буфер
            buffer_duration = float(self.duration_input.text)
            self.frames = deque(maxlen=int(44100 * buffer_duration / self.CHUNK))
            
            # Получаем выбранный номер из спиннера
            selected_number = int(self.device_spinner.text.split(':')[0])
            device_index = self.device_indices[selected_number]
            
            # Запускаем поток записи
            self.recording_thread = threading.Thread(target=self.record_audio, args=(device_index,))
            self.recording_thread.start()
            
            # Запускаем поток прослушивания клавиш
            self.key_thread = threading.Thread(target=self.check_key)
            self.key_thread.daemon = True
            self.key_thread.start()
            
        except Exception as e:
            print(f"Ошибка при запуске записи: {e}")
    
    def check_key(self):
        last_save_time = 0
        min_interval = 0.1  # Минимальный интервал между сохранениями в секундах
        
        while self.is_recording:
            if keyboard.is_pressed('k'):
                current_time = time.time()
                if current_time - last_save_time >= min_interval:
                    self.save_current_buffer()
                    last_save_time = current_time
                    time.sleep(0.1)  # Небольшая задержка после сохранения
    
    def save_current_buffer(self):
        if not self.frames:
            return
        
        # Копируем текущий буфер
        current_frames = list(self.frames)
        
        buffer_duration = float(self.duration_input.text)
        self.frames = deque(maxlen=int(44100 * buffer_duration / self.CHUNK))
        
        if current_frames:
            audio_data = np.concatenate(current_frames)
            format_type = self.format_spinner.text
            bitrate = self.bitrate_spinner.text  # Получаем выбранный битрейт
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            recordings_folder = "Recordings"
            filename = f"{recordings_folder}/recording_{timestamp}.{format_type.lower()}"
            
            # Убеждаемся, что папка существует
            os.makedirs(recordings_folder, exist_ok=True)
            
            # Временный WAV файл
            temp_wav = f"{recordings_folder}/temp_recording_{timestamp}.wav"

            try:
                # Сохраняем временный WAV файл в 16-битном формате
                with wave.open(temp_wav, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 2 байта для int16
                    wf.setframerate(44100)
                    wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

                if format_type == 'WAV':
                    os.replace(temp_wav, filename)
                elif format_type == 'MP3':
                    try:
                        # Используем pydub для конвертации с выбранным битрейтом
                        audio = AudioSegment.from_wav(temp_wav)
                        # Добавляем проверку наличия ffmpeg
                        if not AudioSegment.converter:
                            print("FFmpeg не найден. Убедитесь, что FFmpeg установлен и добавлен в PATH")
                            return
                            
                        audio.export(
                            filename, 
                            format="mp3",
                            bitrate=bitrate,
                            parameters=["-q:a", "0"]
                        )
                    except Exception as e:
                        print(f"Ошибка при конвертации в MP3: {e}")
                        # Попробуем альтернативный метод сохранения, если первый не удался
                        try:
                            from pydub.utils import which
                            AudioSegment.converter = which("ffmpeg")
                            audio = AudioSegment.from_wav(temp_wav)
                            audio.export(filename, format="mp3", bitrate=bitrate)
                        except Exception as e2:
                            print(f"Альтернативный метод также не удался: {e2}")
                elif format_type == 'FLAC':
                    sf.write(filename, audio_data, 44100, format='FLAC')
                elif format_type == 'OGG':
                    sf.write(filename, audio_data, 44100, format='OGG')

                # Удаляем временный WAV файл, если это не WAV формат
                if format_type != 'WAV' and os.path.exists(temp_wav):
                    os.remove(temp_wav)

                print(f"Сохранено в {filename}")
                # Выводим информацию о длительности
                duration = len(audio_data) / 44100
                print(f"Длительность записи: {duration:.2f} секунд")

            except Exception as e:
                print(f"Ошибка при сохранении аудио: {e}")
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)
    
    def record_audio(self, device_index):
        FORMAT = pyaudio.paFloat32
        CHANNELS = 1
        RATE = 44100
        
        self.stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.CHUNK
        )
        
        print("Recording started. Press 'k' to save buffer.")
        
        while not self.stop_recording:
            try:
                data = self.stream.read(self.CHUNK)
                if self.is_recording:
                    audio_data = np.frombuffer(data, dtype=np.float32)
                    self.frames.append(audio_data)
            except Exception as e:
                print(f"Error during recording: {e}")
                break
        
        print("Recording stopped")
        self.stream.stop_stream()
        self.stream.close()
    
    def on_stop(self):
        self.stop_recording = True
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

if __name__ == '__main__':
    AudioRecorderApp().run()