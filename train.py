# import os
# from trainer import Trainer, TrainerArgs
# from TTS.tts.configs.glow_tts_config import GlowTTSConfig
# from TTS.tts.configs.shared_configs import BaseDatasetConfig
# from TTS.tts.datasets import load_tts_samples
# from TTS.tts.models.glow_tts import GlowTTS
# from TTS.tts.utils.text.tokenizer import TTSTokenizer
# from TTS.utils.audio import AudioProcessor

# # Define the output directory (e.g., current folder or a dedicated training folder)
# output_path = os.path.join(os.getcwd(), "output")

# # Configure your dataset
# dataset_config = BaseDatasetConfig(
#     formatter="ljspeech",  # Use the LJSpeech formatter (since your format is similar)
#     meta_file_train="metadata.txt",
#     path="C:/Users/Monika/Desktop/Saras AI/TTS/MyTTSDataset"  # Update this to your dataset folder path
# )

# # Initialize training configuration for the model (GlowTTS in this case)
# config = GlowTTSConfig(
#     batch_size=32,
#     eval_batch_size=16,
#     num_loader_workers=4,
#     num_eval_loader_workers=4,
#     run_eval=True,
#     test_delay_epochs=-1,
#     epochs=1000,
#     text_cleaner="phoneme_cleaners",
#     use_phonemes=True,
#     phoneme_language="en-us",
#     phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
#     print_step=25,
#     print_eval=True,
#     mixed_precision=True,
#     output_path=output_path,
#     datasets=[dataset_config],
# )

# # Initialize the audio processor (handles feature extraction)
# ap = AudioProcessor.init_from_config(config)

# # Initialize the tokenizer (converts text to token IDs)
# tokenizer, config = TTSTokenizer.init_from_config(config)

# # Load training and evaluation samples from your dataset
# train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True)

# # Initialize the model (GlowTTS) 
# model = GlowTTS(config, ap, tokenizer, speaker_manager=None)

# # Initialize the Trainer which manages the training process
# if __name__ == "__main__":
#     trainer = Trainer(
#         TrainerArgs(),
#         config,
#         output_path,
#         model=model,
#         train_samples=train_samples,
#         eval_samples=eval_samples
#     )
#     trainer.fit()


##################################



import os
import torch
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.vits_config import VitsConfig  # Note: class name is VitsConfig (lowercase 'i')
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits  # Vits model
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
import TTS.tts.datasets.formatters as formatters

# Custom formatter to modify the default LJSpeech behavior:
# It replaces "wavs" with "wavs_mono" in the audio file paths.
def custom_ljspeech_formatter(root_path, meta_file_train, **kwargs):
    # Use the default ljspeech formatter to parse metadata.txt
    items = formatters.ljspeech(root_path, meta_file_train, **kwargs)
    # Update the audio file paths: replace "wavs" with "wavs_mono"
    for item in items:
        item["audio_file"] = item["audio_file"].replace("wavs", "wavs_mono")
    return items

# Define the output directory (for example, a folder in the current working directory)
output_path = os.path.join(os.getcwd(), "output")

# Configure your dataset.
# The dataset root is "MyTTSDataset" where metadata.txt resides.
dataset_config = BaseDatasetConfig(
    formatter="ljspeech",        # We'll override with our custom formatter below.
    meta_file_train="metadata.txt",
    path="MyTTSDataset"           # metadata.txt is in MyTTSDataset; audio files are in MyTTSDataset/wavs_mono
)

# Initialize the training configuration for VITS.
# Adjust hyperparameters as needed; here we use a lower batch size for Colab.
config = VitsConfig(
    batch_size=8,                # Reduced batch size for Colab GPU
    eval_batch_size=4,
    num_loader_workers=2,        # Fewer workers to avoid instability on Colab
    num_eval_loader_workers=1,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,                 # Change if you wish to train fewer epochs initially
    text_cleaner="phoneme_cleaners",
    use_phonemes=True,
    phoneme_language="en-us",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    print_step=25,
    print_eval=True,
    mixed_precision=True,        # Enable FP16 mixed precision (Colab GPUs support this)
    output_path=output_path,
    datasets=[dataset_config],
)

# Ensure the model runs on GPU if available.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {device.upper()}")

# Initialize the audio processor (handles feature extraction)
ap = AudioProcessor.init_from_config(config)

# Initialize the tokenizer (converts text to token IDs)
tokenizer, config = TTSTokenizer.init_from_config(config)

# Load training and evaluation samples using the custom formatter
train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True, formatter=custom_ljspeech_formatter)

# Initialize the VITS model.
model = Vits(config, ap, tokenizer, speaker_manager=None)
model.to(device)

if __name__ == "__main__":
    trainer = Trainer(
        TrainerArgs(),
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples
    )
    trainer.fit()
