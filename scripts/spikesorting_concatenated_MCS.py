# Set logging before the rest as neo (and neo-based imports) needs to be imported after logging has been set
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('multirec_sorting')
logger.setLevel(logging.DEBUG)

import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import json
from jsmin import jsmin
import datetime
import re

import spikeinterface.core as sc
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.exporters as sexp

from spikeinterface import concatenate_recordings

from probeinterface import generate_multi_columns_probe

sys.path.append("/lustre/home/ucjuhae/spikesorting_scripts/")
from spikesorting_scripts.helpers import generate_warp_32ch_probeMCS
from spikesorting_scripts.preprocessing import remove_disconnection_events

def compute_rec_power(rec):
    subset_data = sc.get_random_data_chunks(rec, num_chunks_per_segment=100,
                                chunk_size=10000,
                                seed=0,
                                )
    power = np.mean(np.abs(subset_data))
    return power

def preprocess_rec(recording):
    probe = generate_warp_32ch_probeMCS()
    recording = recording.set_probe(probe)
    recording_pre = spre.common_reference(recording, reference='global', operator='median')
    recording_pre = remove_disconnection_events(recording_pre,
                            compute_medians="random",
                            chunk_size= int(recording_pre.get_sampling_frequency()*3),
                            n_median_threshold=3,
                            n_peaks=0,
                            )
    recording_pre = spre.bandpass_filter(recording_pre, freq_min=200, freq_max=4999)
    recording_pre = spre.whiten(recording_pre, dtype='float32')

    return recording_pre


def export_all(sortername, sortextract, recextract, rec_name, working_directory, output_folder, job_kwargs):

    outDir = output_folder / rec_name / sortername
    logger.info(f'saving {outDir} as phy')
    we = sc.extract_waveforms(recextract,
                            sortextract, outDir / 'waveforms', 
                            ms_before=2.5, ms_after=3, 
                            max_spikes_per_unit=500, #load_if_exists=True,
                            overwrite=True,
                            **job_kwargs
                            # n_jobs=10, 
                            # chunk_size=30000
                        )
    logger.info(f'WaveformExtractor: {we}')

    sexp.export_to_phy(we, outDir / 'phy', remove_if_exists=True,
            copy_binary=True,
            **job_kwargs
            )
    logger.info(f'saved {outDir} as phy')
    sexp.export_report(we, outDir / 'report', 
            format='png',
            force_computation=True,
            **job_kwargs)
            
    logger.info(f'saving report')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("params_file", help="path to the json file containing the parameters")
    args = parser.parse_args()
    # params_file = '/home/skgtjml/code/spikesorting_scripts/scripts/json_files/spikesorting_params_concatenated_WARP.json'
    with open(args.params_file) as json_file:
        minified = jsmin(json_file.read()) # Parses out comments.
        params = json.loads(minified)

    logpath = Path(params['logpath'])
    now = datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S')

    fh = logging.FileHandler(logpath / f'multirec_warp_sorting_logs_{now}.log')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    logger.info('Starting')

    sorter_list = params['sorter_list']
    logger.info(f'sorter list: {sorter_list}')

    if 'kilosort2' in sorter_list:
        ss.Kilosort2Sorter.set_kilosort2_path(params['sorter_paths']['kilosort2_path'])
    if 'waveclus' in sorter_list:
        ss.WaveClusSorter.set_waveclus_path(params['sorter_paths']['waveclus_path'])
    if 'kilosort3' in sorter_list:
        ss.Kilosort3Sorter.set_kilosort3_path(params['sorter_paths']['kilosort3_path'])

    datadir = Path(params['datadir']) / params['rec_name']
    print(datadir)
    streams = params['streams']

    output_folder = Path(params['output_folder']) / params['rec_name']
    output_folder.mkdir(parents=True, exist_ok=True)

    working_directory = Path(params['working_directory']) / params['rec_name']
    working_directory.mkdir(parents=True, exist_ok=True)

    blocks = [bl.name for bl in datadir.glob('FerretFace*')]
    print(blocks)
    blocks.sort(key=lambda f: int(re.sub('\D', '', f)))
    pbar = tqdm(blocks)

    recording_list = {stream: [] for stream in streams}

    for stream in streams:
        powers = []
        logger.info(f'Loading stream {stream}')
        for block in pbar:
            pbar.set_postfix_str(f'loading {block}')
            logger.info(f'Loading block {block}')
            try:
                h5_file = list((datadir / block).glob('*R.h5'))
                assert len(h5_file) == 1
                h5_file = h5_file[0]
                rec = se.read_mcsh5(h5_file, stream_id=2)
                powers.append(compute_rec_power(rec))
                rec= preprocess_rec(rec)
                recording_list[stream].append(rec)
            except Exception as e:
                logger.info(f'Could not load block {block}')
                logger.debug(f'Error: {e}')
                
        # only keep recordings with power below 2*median and above 0
        recording_list[stream] = [recording_list[stream][i] for i, power in enumerate(powers) if power < 2*np.median(powers) and power > 0]

    logger.info('Concatenating recordings')
    recordings = {f'{params["rec_name"]}_{stream}': concatenate_recordings(recording_list[stream]) for stream in streams}

    logger.info('Preprocessing recordings')
    recordings = {f'{params["rec_name"]}_{stream}': preprocess_rec(recordings[stream]) for stream in recordings}

    logger.info(f'{[recordings[stream] for stream in recordings]}')
    logger.info('Sorting')

    sortings = ss.run_sorter(sorter_list[2], rec, output_folder=working_directory,remove_existing_folder=True)

    logger.info('Finished sorting')

    export_all(sorter_list[2], sortings, rec, rec_name=params["rec_name"],working_directory=working_directory, 
            output_folder=output_folder,
            job_kwargs=params['job_kwargs']
            )

    # for stream in streams:
    #     logger.info(f'Starting sorting for stream {stream}')
    #     rec = recordings[stream]
    #     logger.info(rec)
    #     s = rec.get_num_samples(segment_index=0)
    #     logger.info(f'segment {0} num_samples {s}')
    #     sorting = ss.run_sorters(sorter_list, [recordings[stream]], working_folder=working_directory / stream,
    #             engine='loop', verbose=True,
    #             mode_if_folder_exists='keep',
    #             sorter_params=params['sorter_params']
    #             )
    #     logger.info(f'Finished sorting for stream {stream}')

    #     export_all(working_directory=working_directory / stream, 
    #             output_folder=output_folder / stream,
    #             job_kwargs=params['job_kwargs']
    #             )

if __name__ == '__main__':
    main()