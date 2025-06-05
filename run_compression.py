import numpy as np
from pathlib import Path
from mtscomp import compress, decompress

# So the plan is to iterate through the folders and compress everything.
#

session_path = Path('F:/F2406_Chevre/20250506-20250509/') 
Sessions = list(session_path.glob('*')) # not in order. No need with this function.
stream_IDs = {'imec1.ap'}
skipIfAlready_cBin = True ### not active yet because atm i would rather overwrite
delete_bin_files_after = False
for session in Sessions:
    for stream_id in stream_IDs:
        print(f'Processing {session.name}')
        dp = session_path / session.name
        probeFolder = list(dp.glob('*' + stream_id[:-3]))
        if any(probeFolder): ### if this doesn't exist you probably don't have the relevant imec file.
            probeFolder = probeFolder[0]
            currentBinFile = list(probeFolder.glob('*.bin'))
            outFileName_cbin = probeFolder / (currentBinFile[0].name[:-3] + 'cbin')
            outFileName_ch = probeFolder / (currentBinFile[0].name[:-3] + 'ch')
            np_samplingRate = 30000
            n_channels_np = 385
            dtype=np.int16 # do not know if this is true. 80% sure it is though. I found a website that said this is the default format at least, but the meta file does not say the precision of individual files...
            if not (outFileName_cbin.exists()&skipIfAlready_cBin):
                compress(currentBinFile[0],outFileName_cbin,outFileName_ch,np_samplingRate,n_channels_np,dtype)

            if currentBinFile[0].exists()&delete_bin_files_after:
                currentBinFile[0].unlink()
                'stuff'

