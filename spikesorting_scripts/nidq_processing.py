from pathlib import Path
import spikeinterface.extractors as se
import numpy as np
from scipy.io import savemat

savepath=Path('F:/Python/SpikeSortingResults/nidqoutput/')
nidq_path=Path('F:/F2301_Clove/F2406_chevre_200525_AM_g0')
nidq_data=se.read_spikeglx(nidq_path,stream_id='nidq')
imec_data=se.read_spikeglx(nidq_path,stream_id='imec0.ap')
channel_ids=nidq_data.get_channel_ids()
nidq_times=nidq_data.get_times()
imec_time=imec_data.get_times()
data_traces=nidq_data.get_traces()
nidq_dict={"data":data_traces,'label':'traces','nidq_times':nidq_times,'imec_times':imec_time}
savemat(savepath/'200525_AM_nidq.mat',nidq_dict)
np.save(savepath/'200525_AM_nidq.npy',data_traces)
np.save(savepath/'200525_AM_nidqtime.npy',nidq_times)
np.save(savepath/'200525_AM_imectime.npy',imec_time)