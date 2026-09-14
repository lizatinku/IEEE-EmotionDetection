from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import wavfile

VIBRATION_ROOT = Path("VibrationData")
OUTPUT_ROOT = Path("Samples")

ACTOR_IDS = ["1016", "1020", "1034", "1050", "1067", "1087"]

SAMPLE_RATE = 8000
SAMPLES_PER_ACTOR = 5

for actor_id in ACTOR_IDS:
    actor_dir = VIBRATION_ROOT / f"vib_{actor_id}"
    manifest_path = actor_dir / "manifest.csv"
    events_path = actor_dir / "events.csv"

    manifest = pd.read_csv(manifest_path).sort_values("chunk_index")
    events = pd.read_csv(events_path)

    boot_dirs = sorted(actor_dir.glob("node-*/boot-*"))
    if not boot_dirs:
        raise FileNotFoundError(f"No boot directory found for actor {actor_id}")

    boot_dir = boot_dirs[0]

    chunks = []
    for relative_path in manifest["relative_path"]:
        chunk_path = actor_dir / relative_path
        chunks.append(np.load(chunk_path).reshape(-1))

    signal = np.concatenate(chunks)

    valid_events = events[events["label"].isin(["calm", "distress"])].copy()

    if len(valid_events) < SAMPLES_PER_ACTOR:
        raise ValueError(
            f"Actor {actor_id} has only {len(valid_events)} "
            f"labeled events, fewer than {SAMPLES_PER_ACTOR}."
        )

    selected_events = valid_events.head(SAMPLES_PER_ACTOR)

    actor_output = OUTPUT_ROOT / f"actor_{actor_id}"
    actor_output.mkdir(parents=True, exist_ok=True)

    print(f"\nActor {actor_id}")

    for sample_num, (_, event) in enumerate(selected_events.iterrows(), start=1):
        start_s = float(event["start_s"])
        end_s = float(event["end_s"])

        start_sample = round(start_s * SAMPLE_RATE)
        end_sample = round(end_s * SAMPLE_RATE)

        clip = signal[start_sample:end_sample].astype(np.float32)

        if len(clip) == 0:
            print(f"  Skipping empty event: {event['source_clip_id']}")
            continue

        emotion = str(event["emotion"])
        label = str(event["label"])
        source_clip = Path(str(event["source_clip_id"])).stem

        filename = (
            f"actor_{actor_id}_"
            f"{emotion}_{label}_"
            f"sample_{sample_num}_"
            f"{source_clip}.wav"
        )

        wav_path = actor_output / filename

        wavfile.write(wav_path, SAMPLE_RATE, clip)

        print(
            f"  Sample {sample_num}: "
            f"{start_s:.2f}s–{end_s:.2f}s | "
            f"{emotion} → {label} | "
            f"{source_clip} → {wav_path.name}"
        )

print("\nDone! Created samples for all actors.")