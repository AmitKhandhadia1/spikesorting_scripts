

from pathlib import Path
import datetime
import argparse
import json
from jsmin import jsmin
import sys

import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.core as sc
import spikeinterface.curation as scu
import spikeinterface.qualitymetrics as sqm
import spikeinterface.exporters as sexp

sys.path.append("F:/Python/spikesorting_scripts/")
from spikesorting_scripts.npyx_metadata_fct import get_npix_sync
from spikesorting_scripts.helpers import get_channelmap_names, sort_np_sessions
from spikesorting_scripts.postprocessing import postprocessing_si
import torch

print(torch.cuda.is_available())

def spikeglx_preprocessing(recording):
    # Preprocessing steps
 #   logger.info(f'preprocessing recording')

    # equivalent to what catgt does
    recording = spre.phase_shift(recording)
    # bandpass filter and common reference can be skipped if using kilosort as it does it internally 
    # but doesn't change anything to keep it
    recording = spre.bandpass_filter(recording, freq_min=300, freq_max=6000)
    recording = spre.common_reference(recording, reference='global', operator='median')
    return recording

def spikesorting_pipeline(rec_name, params):
    # Spikesorting pipeline for a single recording
    working_directory = Path(params['working_directory']) / 'tempDir'

    recording = se.read_spikeglx(rec_name, stream_id = 'imec1.ap')
    recording = spikeglx_preprocessing(recording)

  #  logger.info(f'running spike sorting')
    sorting_output = ss.run_sorters(params['sorter_list'], [recording], working_folder=working_directory,
        mode_if_folder_exists='overwrite', 
        engine='loop', verbose=True)

def spikesorting_postprocessing(sorting,params):
    jobs_kwargs = params['jobs_kwargs']
    #sorting_output = ss.collect_sorting_outputs(Path(params['working_directory']))
   # for (rec_name, sorter_name), sorting in sorting_output.items():
       # logger.info(f'Postprocessing {rec_name} {sorter_name}')
    sorter_name=params['sorter_list']
    rec=sorting._recording
    if params['remove_dup_spikes']:
   #         logger.info(f'removing duplicate spikes')
        sorting = scu.remove_duplicated_spikes(sorting, censored_period_ms=params['remove_dup_spikes_params']['censored_period_ms'])
        
    sorting = scu.remove_excess_spikes(sorting, sorting._recording)

    #    logger.info('waveform extraction')
    outDir = Path(params['output_folder']) / sorter_name[0]/params['rec_name']
    if (outDir / 'waveforms_folder').exists():
        we = sc.load_waveforms(outDir / 'waveforms_folder', sorting=sorting)
    else:
        we = sc.create_sorting_analyzer(recording=rec, sorting=sorting, folder=outDir / 'sortings_folder',
                                        format="binary_folder",
                                        sparse=True,
                                        overwrite=True)
        we.compute('random_spikes',max_spikes_per_unit=300)
        we.compute('waveforms',ms_before=2,ms_after=3.)
        we.compute(['templates','spike_amplitudes','template_similarity','noise_levels'])
     #   we = sc.extract_waveforms(sorting._recording, sorting, outDir / 'waveforms_folder',
      #              # load_if_exists=True,
       #             overwrite=None,
        #            ms_before=2, 
         #           ms_after=3., 
          #          max_spikes_per_unit=300,
           #         sparse=True,
            #        num_spikes_for_sparsity=100,
             #       method="radius",
              #      radius_um=40,
               #     **jobs_kwargs)


    if not (outDir / 'report').exists():
        # logger.info(f'Computing quality netrics')
        
            
            #logger.info(f'Exporting to phy')
        sexp.export_to_phy(we, outDir / 'phy_folder', 
                            verbose=True, 
                            compute_pc_features=False,  copy_binary=False,
                            remove_if_exists=True,
                            **jobs_kwargs)
        
         

        try:
             #   logger.info('Export report')
            sexp.export_report(we, outDir / 'report',
                        format='png',
                        force_computation=True,
                        **jobs_kwargs)
            metrics = sqm.compute_quality_metrics(we, n_jobs = jobs_kwargs['n_jobs'], verbose=True)  
        except Exception as e:
            print(f'Export report failed: {e}')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("params_file", help="path to the json file containing the parameters")
    args = parser.parse_args()
    with open(args.params_file) as json_file:
        minified = jsmin(json_file.read()) # Parses out comments.
        params = json.loads(minified)

    logpath = Path(params['logpath'])
    now = datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S')

   # fh = logging.FileHandler(logpath / f'neuropixels_sorting_logs_{now}.log')
   # fh.setLevel(logging.DEBUG)
    #logger.addHandler(fh)

 #   logger.info('Starting')
#
    sorter_list = params['sorter_list'] #['klusta'] #'kilosort2']
  #  logger.info(f'sorter list: {sorter_list}')

    if 'kilosort2' in sorter_list:
        ss.Kilosort2Sorter.set_kilosort2_path(params['sorter_paths']['kilosort2_path'])
    if 'waveclus' in sorter_list:
        ss.WaveClusSorter.set_waveclus_path(params['sorter_paths']['waveclus_path'])
    if 'kilosort3' in sorter_list:
        ss.Kilosort3Sorter.set_kilosort3_path(params['sorter_paths']['kilosort3_path'])

    datadir = Path(params['datadir']) #/ params['rec_name']
    output_folder = Path(params['output_folder'])/params['rec_name']
    working_directory = Path(params['working_directory'])/params['rec_name']

   # logger.info('Start loading recordings')

    # Load recordings
    sessions = [sess for sess in datadir.glob(params['session_name'])]
    sessions = sort_np_sessions(sessions)
    stream=params['streams']
    recordings_dict = {}
    # /!\ This assumes that all the recordings must have same mapping
    # And assumes one probe per recording
    for session in sessions:
        # Extract sync onsets and save as catgt would
        # get_npix_sync(datadir / session, sync_trial_chan=[5])

        recording = se.read_spikeglx(datadir / session, stream_id=stream)
        recording = spikeglx_preprocessing(recording)
        chan_dict = get_channelmap_names(datadir/session)
        print(chan_dict)
        rec_names = [rec for rec in chan_dict]
        chan_map_name = chan_dict[rec_names[0]][:-5]

        if chan_map_name in recordings_dict:
            recordings_dict[chan_map_name].append(recording)
        else:
            recordings_dict[chan_map_name] = [recording]

        # recordings_list.append(recording)

    #logger.info('Concatenating recordings')
    multirecordings = {channel_map: sc.concatenate_recordings(recordings_dict[channel_map]) for channel_map in recordings_dict}
    multirecordings = {channel_map: multirecordings[channel_map].set_probe(recordings_dict[channel_map][0].get_probe()) for channel_map in multirecordings}

    #logger.info(f'{[multirecordings[ch_map] for ch_map in multirecordings]}')
   # multirecordings = sc.concatenate_recordings(recordings_list)
    #multirecordings = multirecordings.set_probe(recordings_list[0].get_probe())
   # sorting = ss.run_sorters(params['sorter_list'], multirecordings, working_folder=working_directory,
       # mode_if_folder_exists='keep', 
      # engine='loop', verbose=True,
      #  sorter_params=params['sorter_params'],
       # )
    channelschosen=['imec1.ap#AP0','imec1.ap#AP96','imec1.ap#AP192','imec1.ap#AP288']
    #test=recording.select_channels(channelschosen)
    for rec in multirecordings:
        sortings = ss.run_sorter(sorter_name=sorter_list[0], recording=recording, output_folder=working_directory,remove_existing_folder=True,**params['sorter_params'][sorter_list[0]])
        print(sortings)
    # # If recordings don't have same mapping, can do something like this:
    # # In this example, only 2 mappings are in the data, but it can be extended to more mappings
    # # To extract channel coordinates from a recording object, use recording.get_channel_locations()
    # # To extract channel coordinates from a probe object, use probe.get_channel_locations()
    # # And then group recordings based on this
    # # More information about probe object on https://probeinterface.readthedocs.io/en/main/
    # Not sure if it works with concatenated recordings
    # And might take a while to run extract waveforms
        spikesorting_postprocessing(sortings, params)

if __name__ == '__main__':
    main()