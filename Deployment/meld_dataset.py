from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer
import os
import cv2
import numpy as np
import torch
import subprocess
import torchaudio

# Disable parallelism warnings & SSL verification
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'


class MELDDataset(Dataset):
    def __init__(self, csv_path, video_dir):
        self.data = pd.read_csv(csv_path)
        self.video_dir = video_dir  # Directory containing video files

        # Refers to bert-base-uncased model in local system
        self.tokenizer = AutoTokenizer.from_pretrained(
            r"C:\MP\models\bert-base-uncased"
        )

        # Mapping emotions to integers
        self.emotion_map = {
            'anger': 0,
            'disgust': 1,
            'fear': 2,
            'joy': 3,
            'neutral': 4,
            'sadness': 5,
            'surprise': 6
        }

        # Mapping sentiments to integers
        self.sentiment_map = {
            'negative': 0,
            'neutral': 1,
            'positive': 2
        }

    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        try:
            if not cap.isOpened():
                raise ValueError(f"Error opening video file: {video_path}")

            # Try and read the first frame to validate video 
            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f"Error reading frame from video file: {video_path}")

            # Reset index to start
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            while len(frames) < 30 and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (224, 224))
                frame = frame / 255.0  # Normalize pixel values to [0, 1]
                frames.append(frame)

        except Exception as e:
            raise ValueError(f"Video error: {str(e)}")
        finally:
            cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames extracted from video: {video_path}")

        # Ensure at least 30 frames are present else add dummy frames
        if len(frames) < 30:
            frames += [np.zeros_like(frames[0])] * (30 - len(frames))
        else:
            frames = frames[:30]

        # Convert to tensor and permute to (T, C, H, W)
        return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)

    # def _extract_audio_features(self, video_path):
    #     audio_path = video_path.replace('.mp4', '.wav')

    #     try:
    #         subprocess.run([
    #             'ffmpeg',
    #             '-i', video_path,
    #             '-vn',
    #             '-acodec', 'pcm_s16le',
    #             '-ar', '16000',
    #             '-ac', '1',
    #             audio_path
    #         ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    #         waveform, sample_rate = torchaudio.load(audio_path)

    #         if sample_rate != 16000:
    #             resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    #             waveform = resampler(waveform)

    #         mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    #             sample_rate=16000,
    #             n_mels=64,
    #             n_fft=1024,
    #             hop_length=512
    #         )
    #         mel_spec = mel_spectrogram(waveform)

    #         # Normalize
    #         mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std())
    #         if mel_spec.size(2) < 300:
    #             pad_amount = 300 - mel_spec.size(2)
    #             mel_spec = torch.nn.functional.pad(mel_spec, (0, pad_amount))
    #         else:
    #             mel_spec = mel_spec[:, :, :300]  # Truncate to 300 frames

    #         return mel_spec  # Shape: (1, n_mels, time)

    #     except subprocess.CalledProcessError as e:
    #         raise RuntimeError(f"FFmpeg error: {str(e)}")
    #     except Exception as e:
    #         raise RuntimeError(f"Audio processing error: {str(e)}")
    #     finally:
    #         if os.path.exists(audio_path):
    #             os.remove(audio_path)

    def _extract_audio_features(self, video_path):
        audio_path = video_path.replace('.mp4', '.wav')

        try:
            # --- Step 1: Extract audio from video using ffmpeg ---
            subprocess.run([
                'ffmpeg',
                '-i', video_path,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                audio_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # --- Step 2: Load audio with torchaudio ---
            waveform, sample_rate = None, None
            try:
                # Try with sox_io (default backend)
                torchaudio.set_audio_backend("sox_io")
                waveform, sample_rate = torchaudio.load(audio_path)
            except Exception as e1:
                try:
                    # Fallback to soundfile if installed
                    import soundfile as sf
                    torchaudio.set_audio_backend("soundfile")
                    waveform, sample_rate = torchaudio.load(audio_path)
                except Exception as e2:
                    raise RuntimeError(
                        f"Audio loading failed with both sox_io and soundfile. "
                        f"Sox_io error: {e1}, Soundfile error: {e2}"
                    )

            # --- Step 3: Resample if needed ---
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)

            # --- Step 4: Mel-spectrogram extraction ---
            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=64,
                n_fft=1024,
                hop_length=512
            )
            mel_spec = mel_spectrogram(waveform)

            # --- Step 5: Normalize and pad/truncate ---
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-9)
            if mel_spec.size(2) < 300:
                pad_amount = 300 - mel_spec.size(2)
                mel_spec = torch.nn.functional.pad(mel_spec, (0, pad_amount))
            else:
                mel_spec = mel_spec[:, :, :300]

            return mel_spec  # Shape: (1, 64, 300)

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg error while extracting audio: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Audio processing error: {str(e)}")
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()

        row = self.data.iloc[idx]
        try:
            # Construct video filename
            video_filename = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
            video_path = os.path.join(self.video_dir, video_filename)

            # Normalize slashes to avoid Windows path issues
            video_path = os.path.normpath(video_path)

            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")

            # Tokenize text
            text_inputs = self.tokenizer(
                row['Utterance'],
                padding='max_length',
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )

            # Load video and audio
            video_frames = self._load_video_frames(video_path)
            audio_features = self._extract_audio_features(video_path)

            # Map sentiment and emotion to numerical values
            emotion_label = self.emotion_map[row['Emotion'].lower()]
            sentiment_label = self.sentiment_map[row['Sentiment'].lower()]

            return {
                'text_inputs': {
                    'input_ids': text_inputs['input_ids'].squeeze(),
                    'attention_mask': text_inputs['attention_mask'].squeeze()
                },
                'video_frames': video_frames,      # Shape: (30, 3, 224, 224)
                'audio_features': audio_features,  # Shape: (1, 64, 300)
                'emotion_label': torch.tensor(emotion_label, dtype=torch.long),
                'sentiment_label': torch.tensor(sentiment_label)
            }

        except Exception as e:
            print(f"Error processing index {video_path}: {str(e)}")
            return None  # Skip problematic items

# Collate function to skip None entries
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None  # Empty batch
    return torch.utils.data.dataloader.default_collate(batch)

# Function to prepare dataloaders
def prepare_dataloader(train_csv, train_video_dir,
                       dev_csv, dev_video_dir,
                       test_csv, test_video_dir, batch_size=32):
    train_dataset = MELDDataset(train_csv, train_video_dir)
    dev_dataset = MELDDataset(dev_csv, dev_video_dir)
    test_dataset = MELDDataset(test_csv, test_video_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, dev_loader, test_loader

# Main execution
if __name__ == "__main__":
    train_loader, dev_loader, test_loader = prepare_dataloader(
        r'C:\MP\dataset\train\train_sent_emo.csv',
        r'C:\MP\dataset\train\train_splits',
        r'C:\MP\dataset\dev\dev_sent_emo.csv',
        r'C:\MP\dataset\dev\dev_splits_complete',
        r'C:\MP\dataset\test\test_sent_emo.csv',
        r'C:\MP\dataset\test\output_repeated_splits_test'
    )

    for batch in train_loader:
        if batch is None:
            continue  # skip empty batch
        print(batch['text_inputs'])
        print(batch['video_frames'].shape)
        print(batch['audio_features'].shape)
        print(batch['emotion_label'])
        print(batch['sentiment_label'])
       
        break
