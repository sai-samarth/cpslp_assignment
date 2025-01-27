# CPSLP Speech Synthesiser Project

## Overview
This repository contains the implementation of the **Speech Synthesiser** assignment for the CPSLP Programming course. The program converts text input into intelligible speech by synthesizing waveforms from diphone recordings.

## Features
1. **Basic Synthesis**: Converts text phrases to audio using diphone concatenation.
2. **Volume Control**: Adjustable loudness through a command-line argument.
3. **Spelling Support**: Synthesizes text as spelled-out letters.
4. **Punctuation Handling**: Handles pauses based on punctuation for more natural synthesis.
5. **File Input Support**: Processes and synthesizes sentences from a text file.
6. **Smoother Concatenation**: Implements cross-fading for seamless diphone transitions.

## How to Run
### Prerequisites
- Python 3.x
- `nltk` library (Ensure `cmudict` and `punkt` are downloaded)
- Provided `simpleaudio.py` and `synth_args.py` modules.

### Basic Command
```bash
python synth.py -p "hello nice to meet you"
```

### Command-Line Options
- `-p/--phrase`: Specify the input phrase for synthesis.
- `--volume/-v`: Set volume (0-100).
- `--spell/-s`: Spell out the input phrase instead of reading normally.
- `--usepunc`: Handle punctuation in the input phrase.
- `--crossfade`: Enable cross-fading for smoother concatenation.
- `--fromfile`: Provide a text file for synthesis.
- `--outfile/-o`: Save the output to a `.wav` file.
- `--play`: Play the synthesised audio.

### Example Usage
#### Synthesizing a Phrase:
```bash
python synth.py -p "A rose by any other name would smell as sweet"
```

#### Synthesizing with Spelling:
```bash
python synth.py -p "hello" --spell
```

#### Synthesizing from a File:
```bash
python synth.py --fromfile input.txt --outfile output.wav
```

## Extensions Implemented
1. **Volume Control**: Allows dynamic amplitude adjustment.
2. **Spelling**: Converts input into a sequence of spelled-out phonemes.
3. **Punctuation Handling**: Adds appropriate pauses for commas, periods, etc.
4. **File Input**: Reads and synthesizes sentences from a text file.
5. **Smooth Concatenation**: Implements cross-fading using Hanning windows.

## Privacy and Access
This repository only contains:
- The implementation code.
- Related support files for execution.

**Important**: The provided diphone recordings and data files are not included in the repository and remain private.
