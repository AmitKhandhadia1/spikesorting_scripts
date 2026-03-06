from pathlib import Path
import spikeinterface.extractors as se
import numpy as np
from scipy.io import savemat
from kilosort.io import save_probe

savepath=Path('F:/Python/SpikeSortingResults/probeoutput/')
imec_path=Path('F:/F2406_Chevre/F2406_chevre_060525_AM_g0')
imec_data=se.read_spikeglx(imec_path,stream_id='imec0.ap')
chanMap = np.arange(384)
channel_locations=imec_data.get_channel_locations()
k_coords=imec_data.get_channel_groups()
x_c=channel_locations[:,0]
y_c=channel_locations[:,1]
n_chan=384
probe={"chanMap":chanMap,'xc': x_c, 'yc': y_c,'kcoords': k_coords,'n_chan': n_chan}
save_probe(probe,savepath/'060625_probe.json')