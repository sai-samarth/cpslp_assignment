import re
from pathlib import Path
from nltk.corpus import cmudict
import simpleaudio
from synth_args import process_commandline
from nltk.tokenize import sent_tokenize
import glob
import os
import numpy as np
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')

class Synth:
    def __init__(self, wav_folder):
        self.diphones = self.get_wavs(wav_folder)
        self.sample_rate = 16000

    def get_wavs(self, wav_folder):
        """Load all diphone .wav files from the given folder and return a dictionary of diphone names and audio data."""

        diphones = {}
        wav_files = Path(wav_folder)
        audio = simpleaudio.Audio(rate=16000)

        for wav_file in wav_files.glob("*.wav"):
            diphone_key = wav_file.stem
            audio.load(str(wav_file))
            diphones[diphone_key] = audio.data

        return diphones
    
    def phones_to_diphones(self, phones):
        """Convert a list of phones to diphones."""

        diphones = []
        for i in range(len(phones) - 1):
            phone1 = phones[i].lower()
            phone2 = phones[i + 1].lower()
            diphone_phone1 = 'pau' if phone1 in ['pauc', 'pauo'] else phone1 # pauc is for comma and pauo is for other punctuation
            diphone_phone2 = 'pau' if phone2 in ['pauc', 'pauo'] else phone2
            diphone_name = diphone_phone1 + '-' + diphone_phone2
            silence_duration = 0
            if phone2 in ['pauc', 'pauo']:
                silence_duration = 200 if phone2 == 'pauc' else 400
            diphones.append((diphone_name, silence_duration))

        return diphones

    def synthesise(self, phones, smooth_concat=False):
        """Synthesize audio from a list of phones. If smooth_concat is True, crossfade between diphones. Returns the final synthesised audio object."""
        diphones_list = self.phones_to_diphones(phones) 

        overlap_duration = 0.01 
        num_samples_overlap = int(overlap_duration * self.sample_rate) 
        hanning_window = np.hanning(num_samples_overlap * 2)

        diphone_arrays = []

        for diphone, silence_duration in diphones_list:
            if diphone in self.diphones:
                diphone_array = self.diphones[diphone]
                diphone_arrays.append((diphone_array, silence_duration))
            else:
                raise ValueError(f"Diphone {diphone} not found in the given wav folder.")

        # Smooth concatenation with crossfading
        if smooth_concat:
            final_audio_array = diphone_arrays[0][0]

            for i in range(1, len(diphone_arrays)): # Start from second diphone
                prev_diphone_array, _ = diphone_arrays[i - 1]
                curr_diphone_array, silence_duration = diphone_arrays[i]

                prev_diphone_tail = prev_diphone_array[-num_samples_overlap:]
                curr_diphone_head = curr_diphone_array[:num_samples_overlap]

                prev_diphone_tail_windowed = prev_diphone_tail * hanning_window[num_samples_overlap:]
                curr_diphone_head_windowed = curr_diphone_head * hanning_window[:num_samples_overlap]

                crossfade_array = prev_diphone_tail_windowed + curr_diphone_head_windowed

                # Remove the last num_samples_overlap samples from the final audio array and replace with crossfade and add the current diphone
                final_audio_array = final_audio_array[:-num_samples_overlap]
                final_audio_array = np.concatenate((final_audio_array, crossfade_array.astype(np.int16), curr_diphone_array[num_samples_overlap:].astype(np.int16))) 

                if silence_duration > 0:
                    num_samples_silence = int(silence_duration * 16000 / 1000)
                    silence_array = np.zeros(num_samples_silence, dtype=np.int16)
                    final_audio_array = np.concatenate((final_audio_array, silence_array))

        # Simple concatenation without crossfading            
        else:
            final_audio_array = []

            for diphone_array, silence_duration in diphone_arrays:
                final_audio_array.append(diphone_array)

                if silence_duration > 0:
                    silence_array = np.zeros(int(silence_duration * 16000 / 1000), dtype=np.int16)
                    final_audio_array.append(silence_array)

            final_audio_array = np.concatenate(final_audio_array)

        syn_audio = simpleaudio.Audio(rate=16000)
        syn_audio.data = final_audio_array
        return syn_audio


class Utterance:
    def __init__(self, phrase, ignore_punctuation=True):
        self.phrase = phrase
        self.ignore_punctuation = ignore_punctuation
        
    def preprocess(self, phrase):
        """Preprocess the input phrase by converting numbers to words and expanding few contractions."""
        
        contractions_dict = {"dr.": "doctor", "mr.": "mister", "mrs.": "missus", "ms.": "miss", "st.": "saint", "jr.": "junior", "sr.": "senior", "prof.": "professor"}
        
        proc_phrase = phrase.strip()
        for word in proc_phrase.split():
            if word.isdigit():
                word_as_words = number_to_words(int(word))
                proc_phrase = proc_phrase.replace(word, word_as_words)
            else:
                contractions_dict.get(word.lower(), word)

        return proc_phrase


    def get_phone_seq(self, spell=False):
        """Convert the input phrase to a list of phonemes using the CMU dictionary. Handles spelling and punctuation cases."""
        translater = cmudict.dict()
        proc_phrase = self.preprocess(self.phrase)

        # Spell the phrase letter by letter
        if spell:
            spell_phone_list = ["PAU"]  # Start with a pause
            for char in proc_phrase:
                if char.isalpha():
                    letter = char.lower()
                    if letter in translater:
                        # Take the second pronunication in the CMU dict for letter "A": 'AH0' & 'EY1'
                        phones = translater[letter][1] if letter == "a" else translater[letter][0] 
                        spell_phone_list.extend("".join(filter(lambda x: x.isalpha(), phone)) for phone in phones) # Remove numbers from phones
                    else:
                        raise ValueError(f"Character not found in cmudict: {char}")
            spell_phone_list.append("PAU") # End with a pause
            return spell_phone_list
            
        
        if self.ignore_punctuation:            
            tokens = re.sub(r'[^\w\s]', '', proc_phrase).split() # Remove all punctuation
        else:
            tokens = re.findall(r'\w+|[.,:?!]', proc_phrase) # Keep words and punctuation that are required (, . : ? !)

        proc_phone_list = ["PAU"] # Start with a pause
        for token in tokens:
            if token.isalpha(): 
                word = token.lower()
                if word in translater:
                    proc_phone_list.extend("".join(filter(lambda x: x.isalpha(), phone)) for phone in translater[word][0]) # Remove numbers from phones
                elif token.isupper():  # Consider tokens in all caps as acronyms
                    proc_phone_list.extend("".join(filter(lambda x: x.isalpha(), phone)) for char in token if char.lower() in translater for phone in translater[char.lower()][0])
                else:
                    raise ValueError(f"Word not found in cmudict: {token}")
            elif token == ',':
                proc_phone_list.append('PAUC')
            elif token in ['.', ':', '?', '!']:
                proc_phone_list.append('PAUO')

        return proc_phone_list + ["PAU"]  # End with a pause

def number_to_words(n):
    """Recurive function to convert a number to words handling special cases."""
    # Modified from https://stackoverflow.com/questions/19504350/how-to-convert-numbers-to-words-without-using-num2word-library

    ones = {0: '', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten', 11: 'eleven', 12: 'twelve',
        13: 'thirteen', 14: 'fourteen', 15: 'fifteen', 16: 'sixteen',17: 'seventeen', 18: 'eighteen', 19: 'nineteen'}
    
    tens = {2: 'twenty', 3: 'thirty', 4: 'forty', 5: 'fifty', 6: 'sixty', 7: 'seventy', 8: 'eighty', 9: 'ninety'}

    def join(*args):
        """Joins words with spaces and filters empty strings"""
        return ' '.join(filter(bool, args)) 

    def below_100(n):
        """Converts numbers below 100 to words."""
        if n < 20:
            return ones[n]
        return join(tens[n // 10], ones[n % 10])

    # Handle special cases
    if n < 0: 
        return 'negative ' + number_to_words(-n)
    if n == 0:
        return 'zero'
    if 1510 <= n <= 1999 and (n % 100 >= 10): # Special case for years between 1510 and 1999 excluding years ending in 00-09
        return join(below_100(n // 100), below_100(n % 100))
    if n < 100:
        return below_100(n)
    if n < 1000:
        return join(ones[n // 100], 'hundred', below_100(n % 100))
    if n < 10000:
        return join(ones[n // 1000], 'thousand', number_to_words(n % 1000))
    return ' '.join(ones[int(d)] for d in str(n) if d.isdigit()) # Convert each digit to words if n is greater than 10000

def volume_control(args, audio):
    """Adjust volume of audio based on user input. Raise ValueError if volume is not between 0 and 100."""
    if args.volume is not None:
        if not (0 <= args.volume <= 100):
            raise ValueError("Volume must be an integer between 0 and 100.")
        audio.rescale(args.volume / 100)

def play(args, audio):
    """Play audio based on user input."""
    if args.play:
        audio.play()

def save_file(args, audio_object):
    """Save the audio to wav file based on user input."""
    if args.outfile:
        audio_object.save(args.outfile)

def process_file(textfile, args):
    """Process a text file and synthesise all sentences in that file line by line."""

    with open(textfile, 'r') as f:
        text = f.read()
    sentences = sent_tokenize(text)

    synth_obj = Synth(wav_folder=args.diphones) 

    audio_objects = []
    for sentence in sentences:
        # Create Utterance and generate phone sequence
        utterance_obj = Utterance(phrase=sentence, ignore_punctuation=not(args.usepunc))
        phone_seq = utterance_obj.get_phone_seq(spell=args.spell)
        
        # Synthesize audio, apply volume control and play if required
        out = synth_obj.synthesise(phone_seq, smooth_concat=args.crossfade)
        volume_control(args, out) 
        play(args, out)
        audio_objects.append(out)
        
    # Save concatenated audio to a file if requiured
    if args.outfile:
        final_audio = simpleaudio.Audio(rate=16000)
        final_audio.data = np.concatenate([audio.data for audio in audio_objects])
        final_audio.save(args.outfile)

    return audio_objects

def main(args):
    
    if args.fromfile:
        # Process text from a file
        audio_objects = process_file(args.fromfile, args)
        
    else:
        utt = Utterance(phrase=args.phrase, ignore_punctuation=not(args.usepunc))
        phone_seq = utt.get_phone_seq(spell=args.spell)
        diphone_synth = Synth(wav_folder=args.diphones)

        # Synthesize audio, apply volume control, play and save file if required
        out = diphone_synth.synthesise(phone_seq, smooth_concat=args.crossfade)
        volume_control(args, out)
        play(args, out)
        save_file(args, out)


if __name__ == "__main__":
    main(process_commandline())