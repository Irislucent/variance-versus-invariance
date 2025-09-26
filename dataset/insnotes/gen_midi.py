import os
import argparse

import numpy as np
import pretty_midi


def gen_dataset_midi(
    save_dir, ins_list=[64, 19, 21, 41, 56, 59, 68, 71, 72, 75, 22, 52], n_per_ins=100
):
    """
    Generate midi files for training,
    Every file plays the 12 pitches cyclically, each for 1 second, consistently with one instrument, for 2 minutes.
    Every note has a velocity randomly sampled from [60, 120].
    Named like [ins_index]_[file_index].mid

    Instrument indices according to GeneraUser GS v1.43.sf2:
    0=Stereo Grand
    4=Tine Electric Piano
    19=Pipe Organ
    21=Accordian
    22=Harmonica
    40=Violin
    41=Viola
    42=Cello
    52=Choir Aahs
    56=Trumpet
    59=Muted Trumpet
    64=Soprano Sax
    65=Alto Sax
    68=Oboe
    71=Clarinet
    72=Piccolo
    73=Flute
    75=Pan Flute
    76=Blown Bottle
    78=Irish Tin Whistle

    n_per_ins: number of midi files per instrument

    ins_list for the training dataset: [64, 19, 21, 41, 56, 59, 68, 71, 72, 75, 22, 52]
    ins_list for the OOD dataset: [0, 4, 73, 78]

    """
    os.makedirs(save_dir, exist_ok=True)
    for i in range(len(ins_list)):
        for j in range(n_per_ins):
            midi_data = pretty_midi.PrettyMIDI(resolution=1000)
            instrument = pretty_midi.Instrument(program=ins_list[i])
            tone_row = np.arange(60, 72)
            for k in range(120):  # 2 minutes
                note = pretty_midi.Note(
                    velocity=np.random.randint(80, 121),
                    pitch=tone_row[k % 12],
                    start=k
                    * 1.056,  # start and end both in seconds. Every note has the duration of 1 second, but there should be a short gap (a hop_length) between notes
                    end=k * 1.056 + 1,  #
                )  # play 120 notes each for 1 second
                instrument.notes.append(note)

            midi_data.instruments.append(instrument)
            midi_name = "ins" + str(i).zfill(2) + "_" + str(j).zfill(2) + ".mid"
            midi_data.write(os.path.join(save_dir, midi_name))
            print(f"Generated {midi_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate midi files for training")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="../data/insnotes_midi_",
        help="Directory to save the generated midi files",
    )
    parser.add_argument(
        "--n_per_ins",
        type=int,
        default=100,
        help="Number of midi files per instrument",
    )
    args = parser.parse_args()
    gen_dataset_midi(args.save_dir, n_per_ins=args.n_per_ins)
