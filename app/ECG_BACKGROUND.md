# ECG signals and the MIT-BIH dataset — a short primer

This is the domain context for the ECG Arrhythmia inference service.

## What is an ECG?

An electrocardiogram measures the small voltage changes on the skin produced by
the heart's electrical activity as it contracts. A "lead" is a pair of
electrodes; the recorded signal is the voltage difference between them.

Each heartbeat shows up as a stereotyped wave with three components:

```
       R
       /\
      /  \
     /    \
P   /      \    T
/\ /        \  /\
__/  \/          \/  \__   ← baseline
        \  /
         \/
         Q  S
```

- **P wave** — atrial depolarization (the top chambers contracting). Small bump
  before the main spike.
- **QRS complex** — ventricular depolarization (the bottom chambers, which do
  most of the pumping). The big spike. The peak inside QRS is the **R-peak**,
  used as the alignment point for almost all ECG analysis.
- **T wave** — ventricular repolarization (the ventricles recovering). Broader
  bump after the spike.

A normal heartbeat lasts roughly 0.6–1.0 seconds. The morphology of these
waves and their timing relative to each other is what a clinician (or a model)
reads to classify a beat.

## What is MIT-BIH?

The **MIT-BIH Arrhythmia Database** is the standard public benchmark for
heartbeat classification. It is hosted on PhysioNet and accessed via the
`wfdb` Python package.

Key facts:

- **48 records**, each ~30 minutes long, from 47 patients.
- **2-channel** recordings (usually MLII + V1 or V5, depending on the patient).
  MLII is the channel used by this project.
- **Sampling rate: 360 Hz** — every record at the same rate, which is why the
  inference code assumes 360 Hz and the model was trained at that rate.
- **Beat annotations:** every QRS complex is hand-annotated by cardiologists
  with a single-character symbol (`N`, `V`, `A`, `L`, etc.) marking what kind
  of beat it is. The annotations are the ground truth.
- **Rhythm annotations:** less critical for this project; we only use beat
  annotations.

The full database is ~100 MB. The project downloads it locally during training
via `scripts/download_data.py`. The container does not ship MIT-BIH; only a
few pre-rendered demo slices are baked into the Streamlit image.

## AAMI beat classes

MIT-BIH uses about 20 distinct beat symbols. The **ANSI/AAMI EC57:2012**
standard collapses these into 5 superclasses for the purposes of benchmarking
arrhythmia detectors. This project uses 4 of them (Q is dropped — see below).

| Class | Name                          | Description                                           | Examples of MIT-BIH symbols |
|-------|-------------------------------|-------------------------------------------------------|----------------------------|
| **N** | Normal                        | Regular sinus rhythm. The heart's pacemaker (SA node) is in charge and the beat looks textbook. Includes some bundle-branch blocks that look abnormal but originate normally. | `N`, `L`, `R`, `e`, `j` |
| **S** | Supraventricular ectopic      | A beat originating *above* the ventricles but not from the normal pacemaker. Often premature. Morphology can look very close to N. | `A`, `a`, `J`, `S` |
| **V** | Ventricular ectopic           | A beat originating in the ventricles themselves. Morphologically very different — wide, distorted QRS, no preceding P-wave. Visually distinctive. | `V`, `E` |
| **F** | Fusion                        | A normal beat and a ventricular beat fire simultaneously, producing a hybrid waveform. Rare and morphologically ambiguous. | `F` |
| **Q** | Unknown / paced               | Either an unclassifiable beat or one driven by a pacemaker. Most Q examples in MIT-BIH come from a few patients with implanted pacemakers. | `/`, `f`, `Q` |

### Why this project drops Q

This project follows the convention of excluding the four paced-only
records (102, 104, 107, 217) from training and evaluation (see other project using the same database). Once those are gone, there are fewer than 20 Q beats in the entire dataset — not enough to train or evaluate. So Q is dropped and we do 4-class classification.

### Class imbalance, in numbers

MIT-BIH is dominated by normal beats. After the DS1/DS2 split and dropping
paced records, the training distribution is roughly:

- **N: ~90%** (the easy class)
- **V: ~7%** (the moderate class, morphologically distinctive)
- **S: ~2%** (hard — looks like N)
- **F: ~1%** (very hard — rare and ambiguous)

A model that always predicts N gets 90% accuracy. This is why per-class F1
is the only metric worth looking at, and why we report it instead of accuracy
in the README.

## The DS1/DS2 inter-patient split

There are two ways to split MIT-BIH into train/test:

- **Intra-patient:** shuffle every beat from every record, then split 80/20.
  This is the *wrong* way because beats from the same patient end up in both
  sets, so the model effectively memorizes patient-specific features instead
  of generalizing. Reported accuracies are inflated by ~5-10%.
- **Inter-patient (de Chazal et al. 2004):** split *records*, not beats.
  DS1 = 22 specific records used for training. DS2 = the other 22 used as
  held-out test. No patient overlap. This is the *honest* split, and it's
  what the literature uses for any result that wants to be taken seriously.

This project uses inter-patient. The DS1/DS2 record lists are hardcoded in
`src/data/build_dataset.py`.

## What the model actually sees

The inference path is:

1. **Bandpass filter** the raw signal at 0.5–40 Hz. This removes:
   - Baseline wander (slow drift below 0.5 Hz, mostly from breathing and
     electrode movement).
   - High-frequency noise (above 40 Hz, mostly muscle artifact and 50/60 Hz
     power-line interference).
2. **Detect R-peaks** in the filtered signal. Training uses MIT-BIH's
   ground-truth peaks. Inference uses a simple amplitude-and-distance peak
   finder (`scipy.signal.find_peaks`). This is the weakest part of the
   pipeline and would be replaced with Pan-Tompkins in a production system.
3. **Cut a fixed window around each R-peak:** 250 samples before, 400 after.
   At 360 Hz that's about 0.7 s before + 1.1 s after = 1.8 s total. Big enough
   to capture the P, QRS, and T of the current beat with margin.
4. **Per-window z-score normalization.** Each window is standardized
   independently. This makes the model invariant to absolute amplitude, which
   varies a lot between patients and electrodes.
5. **1D CNN forward pass.** Output: 4 logits → softmax → class + confidence.

## Why a 1D CNN works here

ECG beats have a stereotyped local structure (P, QRS, T) that appears at
roughly the same relative position within each window but with variable
amplitude, width, and noise. Three properties of 1D convolutions match this:

- **Translation equivariance.** A filter that learns a QRS-like shape detects
  it wherever it appears in the window. R-peak alignment is imperfect (the
  peak detector adds jitter), and convolutions absorb that cheaply.
- **Local receptive fields.** Early layers pick up local features (slopes,
  narrow peaks); deeper layers reason about sub-wave timing. This mirrors
  how a clinician reads a beat: local morphology first, then rhythm context.
- **Parameter efficiency.** Weight sharing keeps the model small (~30k
  parameters here). With only a few hundred examples of S and F, a larger
  model would memorize N and overfit the minority classes.

You can think of a 1D CNN as a learned FIR-filter-bank with non-linearities —
the same intuition from classical signal processing, except the filter
coefficients are learned from labels rather than hand-designed.

## What this model does NOT do

- It does not detect rhythms (e.g. atrial fibrillation, ventricular
  tachycardia). It classifies individual beats only.
- It does not output uncertainty estimates beyond the softmax confidence —
  which is not a calibrated probability.
- It does not generalize to non-MIT-BIH recordings without significant
  domain-adaptation work. Different leads, different sampling rates,
  different patient populations all shift the input distribution.
- It is **not a medical device.** Do not use it to diagnose anything.

## References

- Moody, G. B., & Mark, R. G. (2001). The impact of the MIT-BIH Arrhythmia
  Database. *IEEE Engineering in Medicine and Biology Magazine*, 20(3), 45–50.
- de Chazal, P., O'Dwyer, M., & Reilly, R. B. (2004). Automatic classification
  of heartbeats using ECG morphology and heartbeat interval features.
  *IEEE Transactions on Biomedical Engineering*, 51(7), 1196–1206.
- AAMI (2012). *Testing and reporting performance results of cardiac rhythm
  and ST segment measurement algorithms* (ANSI/AAMI EC57:2012).
- Kiranyaz, S., Avci, O., Abdeljaber, O., Ince, T., Gabbouj, M., & Inman, D. J.
  (2021). 1D Convolutional Neural Networks and Applications: A Survey.
  *Mechanical Systems and Signal Processing*, 151, 107398.
