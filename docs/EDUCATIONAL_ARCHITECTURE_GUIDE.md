# Educational architecture guide: Mapperatorinator and SnapBeat LoRA

This document is written for coursework and self-study in machine learning. It explains the **Mapperatorinator** stack from basic ideas through the **SnapBeat** LoRA fine-tuning setup used in this repository.

Related operational docs: [SNAPBEAT_FINETUNING.md](SNAPBEAT_FINETUNING.md), [SNAPBEAT_TRAINING_RESULTS.md](SNAPBEAT_TRAINING_RESULTS.md).

---

## 1. Problem statement and thesis hook

**Supervised sequence modeling.** The model learns a conditional distribution over **discrete chart tokens** given **audio**:

\[
P(\text{tokens} \mid \text{audio}) \approx \prod_t P(x_t \mid x_{<t}, \text{audio}).
\]

Training minimizes **cross-entropy** on next-token prediction. The chart is not predicted as raw JSON or pixels; it is predicted as a **sequence of event symbols** (timing, columns, note types, and so on) that downstream code can turn into `.osu` or SnapBeat JSON.

**Why this matters for a final project:** you can frame the work as **structured prediction from audio**—the same family as speech recognition (audio → text) or music transcription, but with a **domain-specific tokenizer** for rhythm games.

---

## 2. High-level system architecture

### 2.1 Data flow

```mermaid
flowchart LR
  subgraph inputs [Inputs]
    audio[RawAudio]
  end
  subgraph encoder_path [Encoder side]
    mel[MelSpectrogram]
    enc[TransformerEncoder]
    audio --> mel --> enc
  end
  subgraph decoder_path [Decoder side]
    ctx[DecoderContextTokens]
    dec[TransformerDecoder]
    lm[LMHead]
    ctx --> dec
    enc --> dec
    dec --> lm
  end
  subgraph output [Output]
    tok[EventTokenIDs]
    lm --> tok
  end
```

- **Encoder** consumes a time series of acoustic features (mel spectrogram frames).
- **Decoder** is **autoregressive**: at each step it sees previous tokens (and optional **context** tokens) and predicts the next token.
- **LM head** maps hidden states to logits over the **event vocabulary**.

### 2.2 What the code actually instantiates

The class **`Mapperatorinator`** ([`osuT5/osuT5/model/modeling_mapperatorinator.py`](../osuT5/osuT5/model/modeling_mapperatorinator.py)) combines:

1. A **`MelSpectrogram`** module (when not using raw wave mode) to turn waveform chunks into mel frames.
2. A **Hugging Face seq2seq backbone** chosen by `get_backbone_model`:

```16:34:osuT5/osuT5/model/modeling_mapperatorinator.py
def get_backbone_model(config: MapperatorinatorConfig):
    name = config.backbone_model_name
    b_config = config.backbone_config

    if name.startswith("google/t5"):
        from transformers import T5Config, T5ForConditionalGeneration
        if isinstance(b_config, dict):
            b_config = T5Config(**b_config)
        model_cls = T5ForConditionalGeneration
    elif name.startswith("OliBomby/nwhisper"):
        from .custom_transformers import NWhisperConfig, NWhisperForConditionalGeneration
        if isinstance(b_config, dict):
            b_config = NWhisperConfig(**b_config)
        model_cls = NWhisperForConditionalGeneration
    elif name.startswith("Tiger14n/ropewhisper"):
        from .custom_transformers import RoPEWhisperConfig, RoPEWhisperForConditionalGeneration
        if isinstance(b_config, dict):
            b_config = RoPEWhisperConfig(**b_config)
        model_cls = RoPEWhisperForConditionalGeneration
```

**Important for `OliBomby/Mapperatorinator-v31`:** training config [`configs/train/v31.yaml`](../configs/train/v31.yaml) pulls in [`configs/model/whisper_small_v2.yaml`](../configs/model/whisper_small_v2.yaml), which sets `name: 'Tiger14n/ropewhisper-small'`. So **v31 is a RoPE Whisper–family encoder–decoder**, not the legacy custom T5 implementation under [`osuT5/osuT5/model/custom_transformers/t5.py`](../osuT5/osuT5/model/custom_transformers/t5.py) (that path serves other backbone names).

The forward path applies the spectrogram, optionally concatenates **conditioning** embeddings along the feature axis, then passes features to the backbone:

```172:209:osuT5/osuT5/model/modeling_mapperatorinator.py
        if encoder_outputs is None and frames is not None:
            if not self.input_raw_wave:
                frames = self.spectrogram(frames)  # (N, L, M)
                frames = frames.to(dtype=self.transformer.dtype)  # Ensure correct dtype for the model
                conds = []

                if self.do_style_embed:
                    style_embedding = self.style_embedder(beatmap_idx)  # (N, D)
                    style_embedding = style_embedding.unsqueeze(1).repeat((1, frames.shape[1], 1))
                    conds.append(style_embedding)
                if self.do_difficulty_embed:
                    difficulty_embedding = self.difficulty_embedder(difficulty)
                    conds.append(difficulty_embedding)
                if self.do_mapper_embed:
                    mapper_embedding = self.mapper_embedder(mapper_idx)
                    conds.append(mapper_embedding)
                if self.do_song_position_embed:
                    song_position_embedding = self.song_pos_embedder(song_position)
                    conds.append(song_position_embedding)

                conds_expanded = [c.unsqueeze(1).expand((-1, frames.shape[1], -1)) for c in conds]
                inputs_embeds = torch.concatenate([frames] + conds_expanded, dim=-1)

            if self.project_encoder_input:
                inputs_embeds = self.encoder_embedder(inputs_embeds) if inputs_embeds is not None else None

            if self.input_raw_wave:
                inputs["input_values"] = frames.reshape(frames.shape[0], -1)
            elif self.input_features:
                inputs["input_features"] = torch.swapaxes(inputs_embeds, 1, 2) if inputs_embeds is not None else None
            else:
                inputs["inputs_embeds"] = inputs_embeds

        if self.embed_decoder_input:
            inputs["decoder_inputs_embeds"] = self.decoder_embedder(decoder_input_ids)
            del inputs["decoder_input_ids"]

        output = self.transformer.forward(**inputs)
```

For **`whisper_small_v2`**, `input_features: true` and `project_encoder_input: false`, so mel (+ optional conds) are passed as **`input_features`** after the channel/time swap.

---

## 3. From fundamentals to advanced concepts

### 3.1 Supervised learning and the generative view

You have pairs \((\text{audio}, \text{token sequence})\). The network parameters \(\theta\) are tuned so that the predicted token distribution matches the ground truth under **cross-entropy**. The implementation uses a **per-class weight vector** so **rhythm-related tokens** contribute more to the loss (configurable `rhythm_weight`), reflecting that timing mistakes are particularly harmful for charts:

```130:137:osuT5/osuT5/model/modeling_mapperatorinator.py
        class_weights = torch.ones(config.vocab_size)
        class_weights[config.rhythm_token_start:config.rhythm_token_end] = config.rhythm_weight
        self.loss_fn = nn.CrossEntropyLoss(
            weight=class_weights,
            reduction="none",
            ignore_index=LABEL_IGNORE_ID,
            label_smoothing=config.label_smoothing
        )
```

### 3.2 Discrete sequences and tokenization

Beatmaps are **event lists** (circle, hold, time shift, mania column, …). The **`Tokenizer`** maps each event type and value to an integer ID ([`osuT5/osuT5/tokenizer.py`](../osuT5/osuT5/tokenizer.py)). **Time** is quantized: `MILISECONDS_PER_STEP = 10`, so one token step corresponds to **10 ms** when encoding and decoding `TIME_SHIFT` ([`osuT5/osuT5/tokenizer.py`](../osuT5/osuT5/tokenizer.py), [`osuT5/osuT5/inference/processor.py`](../osuT5/osuT5/inference/processor.py)).

### 3.3 Autoregressive modeling

The decoder models \(P(x_t \mid x_{<t}, \text{encoder output})\). During **training**, **teacher forcing** exposes the decoder to the true prefix; the loss is applied on **shifted** labels. During **inference**, the model typically conditions on **its own** previous predictions, which can drift (exposure bias)—a standard topic in seq2seq generation.

### 3.4 Encoder–decoder structure and cross-attention

- **Encoder:** reads the full mel sequence and produces a sequence of hidden vectors.
- **Decoder:** uses **self-attention** over past tokens (with a causal mask) and **cross-attention** to the encoder states so each prediction can depend on **when** something happens in the audio.

### 3.5 Transformers (concise)

- **Self-attention** computes weighted combinations of values from all positions (masked in the decoder).
- **Multi-head attention** runs several attentions in parallel.
- **Feed-forward (MLP) blocks** apply position-wise nonlinearities after attention.

**Whisper-style inductive bias:** speech models are already trained for **acoustic input → discrete output**; this project reuses that family for **acoustic input → chart tokens**.

### 3.6 Spectrograms and mel scale (audio ML)

- **STFT:** splits the waveform into overlapping windows and Fourier-transforms each → a time–frequency representation.
- **Mel filterbank:** compresses frequency bins into **mel bands** aligned with human pitch perception.
- **Result:** each time frame is a vector of length **`n_mels`** (80 in [`configs/model/whisper_small_v2.yaml`](../configs/model/whisper_small_v2.yaml)).

### 3.7 Conditioning and SnapBeat-specific context

Many training configs inject **metadata** (difficulty, mapper, …) as extra embeddings broadcast across time. **SnapBeat** disables most osu!-specific prefix tokens and instead uses **`context_types`** so **timing context** (beat grid / timing events) can be placed in the **decoder input** before the map tokens. That is configured in [`configs/train/snapbeat_lora.yaml`](../configs/train/snapbeat_lora.yaml) and implemented in [`osuT5/osuT5/dataset/snapbeat_parser.py`](../osuT5/osuT5/dataset/snapbeat_parser.py) (`parse_timing`) and [`osuT5/osuT5/dataset/snapbeat_dataset.py`](../osuT5/osuT5/dataset/snapbeat_dataset.py). See [SNAPBEAT_FINETUNING.md](SNAPBEAT_FINETUNING.md) for the end-to-end pipeline table.

### 3.8 Transfer learning and LoRA

- **Transfer learning:** initialize from **`OliBomby/Mapperatorinator-v31`** (pretrained on large osu! chart data), then adapt weights toward SnapBeat charts.
- **LoRA (Low-Rank Adaptation):** instead of updating full weight matrices \(W\), train low-rank matrices \(A, B\) so the effective update is small-rank. Fewer trainable parameters → less GPU memory and often **stable** fine-tuning.

**Where LoRA is attached:** [`osuT5/train.py`](../osuT5/train.py) uses PEFT when `enable_lora` is true:

```74:81:osuT5/train.py
    if args.enable_lora:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(**args.lora)
        model = get_peft_model(model, lora_config)
        # lora_params = {n: p for n, p in model.named_parameters() if "lora" in n}
        # for n, p in lora_params.items():
        #     print(n, p.sum())
        model.print_trainable_parameters()
```

**SnapBeat LoRA targets** (attention projections **and** MLP `fc1`/`fc2`) and hyperparameters live in [`configs/train/snapbeat_lora.yaml`](../configs/train/snapbeat_lora.yaml): rank `r: 64`, `lora_alpha: 128`, `init_lora_weights: "pissa"`, dropout `0.05`. Training-only weights are the adapters; the base checkpoint can stay frozen (depending on optimizer grouping).

### 3.9 Training mechanics (terminology for your writeup)

- **Mixed precision (`bf16`):** faster matmuls on supported GPUs; often nicer than fp16 for training stability.
- **Gradient accumulation (`grad_acc`):** backward steps accumulate before an optimizer step → **effective batch size** = micro-batch × accumulation steps × number of devices.
- **Cosine LR schedule:** learning rate decays following a cosine curve (`final_cosine` in config).
- **IterableDataset / workers:** SnapBeat uses streaming-style loading; [SNAPBEAT_FINETUNING.md](SNAPBEAT_FINETUNING.md) documents **deadlock** risks with `persistent_workers` and multi-worker iterators—valid “systems + ML” content for a report.

### 3.10 Inference vs training

- **Training:** ground-truth prefixes, cross-entropy on next tokens.
- **Inference:** autoregressive decoding with optional **logit processors**—for example [`MonotonicTimeShiftLogitsProcessor`](../osuT5/osuT5/inference/logit_processors.py) restricts `TIME_SHIFT` tokens so time does not go backward in invalid ways.

**SnapBeat output path:** [`snapbeat_inference.py`](../snapbeat_inference.py) runs the usual mania inference to produce `.osu`, then [`snapbeat_converter.py`](../snapbeat_converter.py) converts to **SnapBeat JSON**. Separating **model output** from **format conversion** is a clean subsection in a thesis (serialization / schema).

---

## 4. Methods: SnapBeat data and config (implementation map)

| Role | Location |
|------|----------|
| Parse SnapBeat JSON → events | [`osuT5/osuT5/dataset/snapbeat_parser.py`](../osuT5/osuT5/dataset/snapbeat_parser.py) |
| Load audio + JSON pairs, build batches | [`osuT5/osuT5/dataset/snapbeat_dataset.py`](../osuT5/osuT5/dataset/snapbeat_dataset.py) |
| Route `dataset_type=snapbeat` | [`osuT5/osuT5/utils/model_utils.py`](../osuT5/osuT5/utils/model_utils.py) |
| LoRA + data hyperparameters | [`configs/train/snapbeat_lora.yaml`](../configs/train/snapbeat_lora.yaml) |
| Train entry | [`osuT5/train.py`](../osuT5/train.py) with `python osuT5/train.py --config-name snapbeat_lora` |

**Token mapping (SnapBeat notes → mania-compatible events)** is summarized in [SNAPBEAT_FINETUNING.md](SNAPBEAT_FINETUNING.md) (short/long notes → `CIRCLE` / `HOLD_NOTE` chains, lanes → `MANIA_COLUMN`, time → `TIME_SHIFT`).

### 4.1 SnapBeat training loop (conceptual)

```mermaid
flowchart TB
  json[SnapBeatJSON]
  aud[AudioFile]
  par[SnapBeatParser]
  ds[SnapBeatDataset]
  batch[Batch_mel_and_tokens]
  base[Mapperatorinator_v31]
  lora[LoRALayers]
  json --> par
  aud --> ds
  par --> ds
  ds --> batch
  batch --> base
  base --> lora
```

---

## 5. Evaluation narrative and limitations (SnapBeat LoRA runs)

Detailed tables live in [SNAPBEAT_TRAINING_RESULTS.md](SNAPBEAT_TRAINING_RESULTS.md). Summary points you can cite in a conclusion:

1. **Large timing jump when modeling timing explicitly:** enabling `add_timing` and `add_timing_points` raised timing accuracy from roughly **49%** (Run 2) to **62%+** (Run 3)—the strongest single configuration change in the logged experiments.
2. **LoRA breadth:** adding **`fc1` and `fc2`** (not only `q/k/v/out` projections) coincided with further practical gains in those runs.
3. **Run 5 (best reported exact timing):** about **64.6%** timing accuracy with `rhythm_weight=5.0`, `label_smoothing=0.05`, and `dt_augment_prob=0.0` (SnapBeat’s fixed BPM makes DT-style augmentation less appropriate).
4. **More data, same timing ceiling:** Run 6 added ~47% more training files; **column** and **other** accuracy improved (~+1.3 pp each) but **timing accuracy stayed flat** (~64.5% vs 64.6%)—evidence that **exact timing** may need **architecture / context** improvements, not only more examples.
5. **Run 7 lessons:** `timing_random_offset=1` **hurt** exact timing (~−9 pp) while fuzzy timing stayed similar—random jitter taught **imprecision**. Also, **`context_types: timing→map` had no effect** until the dataset actually supplied timing context (previously hardcoded `none`); post–Run 7, `SnapBeatParser.parse_timing()` and dataset wiring fix that—see [SNAPBEAT_FINETUNING.md](SNAPBEAT_FINETUNING.md) and [SNAPBEAT_TRAINING_RESULTS.md](SNAPBEAT_TRAINING_RESULTS.md).

**Fuzzy vs exact timing:** a large gap (e.g. fuzzy ~85% vs exact ~64%) suggests the model often lands **near** the correct beat but not on the exact 10 ms token—useful language for discussing **quantization** and **evaluation metrics** in your report.

---

## 6. Scope caveats (stay accurate in your final writeup)

- **`osu_diffusion`** refines **spatial** layouts for some osu! modes; do **not** present it as central to SnapBeat mania unless your inference pipeline actually enables it.
- **`get_backbone_model`** supports **multiple** architectures; for **v31 + SnapBeat**, state clearly that the backbone is **`Tiger14n/ropewhisper-small`**, not T5 or vanilla Whisper unless you change the model config.

---

## 7. Suggested report outline (checklist)

1. Problem: audio-conditioned **structured** sequence prediction.
2. Data: mel spectrogram + event tokenizer; SnapBeat JSON parsing.
3. Model: Mapperatorinator = mel front-end + RoPE Whisper seq2seq (for v31).
4. Objective: weighted cross-entropy + label smoothing (`snapbeat_lora.yaml`).
5. Fine-tuning: LoRA via PEFT (`train.py`, `snapbeat_lora.yaml`).
6. Results: cite [SNAPBEAT_TRAINING_RESULTS.md](SNAPBEAT_TRAINING_RESULTS.md); discuss timing plateau and timing context.
7. Ethics / data: licensing, dataset not in git ([SNAPBEAT_FINETUNING.md](SNAPBEAT_FINETUNING.md)).

This checklist aligns section-by-section with the learning material above so you can paste headings into a thesis or final report and expand each with your own figures and citations.
