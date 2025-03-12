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
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

# Define the output directory (e.g., current folder or a dedicated training folder)
output_path = os.path.join(os.getcwd(), "output")

# Configure dataset path
dataset_config = BaseDatasetConfig(
    formatter="ljspeech",  
    meta_file_train="metadata.txt",
    path="C:/Users/Monika/Desktop/Saras AI/TTS/MyTTSDataset"
)

# Set up GlowTTS training configuration for CPU
config = GlowTTSConfig(
    batch_size=4,  # Reduced batch size for CPU training
    eval_batch_size=2,  # Smaller batch size for evaluation
    num_loader_workers=2,  # Reduce to prevent CPU overload
    num_eval_loader_workers=1,  
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="phoneme_cleaners",
    use_phonemes=True,
    phoneme_language="en-us",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    print_step=25,
    print_eval=True,
    mixed_precision=False,  # Disable mixed precision (CPU doesn't support FP16)
    output_path=output_path,
    datasets=[dataset_config],
)

# Ensure training runs on CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {device.upper()} (Expect slow training on CPU)")

# Initialize audio processor
ap = AudioProcessor.init_from_config(config)

# Initialize tokenizer
tokenizer, config = TTSTokenizer.init_from_config(config)

# Load training and evaluation samples
train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True)

# Initialize the GlowTTS model
model = GlowTTS(config, ap, tokenizer, speaker_manager=None)
model.to(device)  # Ensure the model runs on CPU

# Start training
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
