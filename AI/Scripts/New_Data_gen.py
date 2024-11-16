import os
import librosa
import soundfile as sf


def split_audio(file_path, sample_duration=4.0, output_folder="WAV/bright"):

    audio, sr = librosa.load(file_path, sr=None)


    samples_per_segment = int(sample_duration * sr)


    os.makedirs(output_folder, exist_ok=True)


    total_samples = len(audio)
    num_segments = total_samples // samples_per_segment

    print(f"Rozbijam plik na {num_segments} próbek po {sample_duration} sekundy")

    for i in range(num_segments):

        start_sample = i * samples_per_segment
        end_sample = start_sample + samples_per_segment
        segment = audio[start_sample:end_sample]


        output_filename = os.path.join(output_folder, f"sample_{i + 1}.wav")
        sf.write(output_filename, segment, sr)
        print(f"Zapisano próbkę: {output_filename}")


    if total_samples % samples_per_segment != 0:
        remainder_segment = audio[num_segments * samples_per_segment:]
        output_filename = os.path.join(output_folder, f"sample_{num_segments + 1}.wav")
        sf.write(output_filename, remainder_segment, sr)
        print(f"Zapisano ostatnią próbkę (mniejsza niż 4 sekundy): {output_filename}")



audio_file = 'Bright.wav'
split_audio(audio_file, sample_duration=4.0, output_folder="WAV/bright")
