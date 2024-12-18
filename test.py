import sounddevice as sd
import soundfile as sf
import numpy as np

# Выводим список всех доступных аудио устройств
print("Доступные устройства:")
print(sd.query_devices())

def record_system_audio(duration=5, samplerate=44100, device=2):
    try:
        # Записываем аудио с указанного устройства
        recording = sd.rec(int(duration * samplerate),
                         samplerate=samplerate,
                         channels=2,
                         dtype=np.float32,
                         device=device)
        
        print("Запись началась...")
        sd.wait()
        print("Запись завершена!")

        # Сохраняем запись в файл
        sf.write('system_audio.wav', recording, samplerate)
        print("Файл сохранен как 'system_audio.wav'")
        
    except Exception as e:
        print(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    record_system_audio()
