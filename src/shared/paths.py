class PathPresets:
    def __init__(self,loc):
        match loc:
            case 'oscar':
                # Location of imaging parsed.xlsx files and preprocessed MNE "raw" files for SWR code
                self.imaging        = '/oscar/data/brainstorm-ws/seeg_data/Memory Task Data/Imaging/Epilepsy/'
                self.processed_raws = '/oscar/data/brainstorm-ws/megagroup_data/'
            
                # Preprocessing data operation paths
                self.figs          = '/oscar/data/brainstorm-ws/megagroup_data/figs'
                self.unproc_data   = '/oscar/data/brainstorm-ws/seeg_data/Memory Task Data/Epilepsy/Monitoring/'
                #self.save_preproc = '/oscar/data/brainstorm-ws/megagroup_data'
                self.save_preproc  = '/oscar/data/brainstorm-ws/tmp'
                self.chnl          = '/oscar/data/brainstorm-ws/seeg_data/Memory Task Data/Imaging/Epilepsy'

            case 'dan-xps-15':
                # Location of imaging parsed.xlsx files and preprocessed MNE "raw" files for SWR code
                self.imaging        = '/home/dan/projects/work/seeg_data/original/Imaging/Epilepsy/'
                self.processed_raws = '/home/dan/projects/work/seeg_data/processed/'
                self.processed_raws = '/home/dan/projects/work/seeg_data/megagroup/'

                # Preprocessing data operation paths
                self.figs         = '/home/dan/projects/work/seeg_data/figs'
                self.unproc_data  = '/home/dan/projects/work/seeg_data/original'
                self.save_preproc = '/home/dan/projects/work/seeg_data/processed'
                self.chnl         = '/home/dan/projects/work/seeg_data/original/Imaging/Epilepsy'

            case '_':
                raise Exception(f'No preset paths for {loc}')
        
        # Suffix to use for SWR detection
        #self.suffix = '_no60hz_bp_rmouts_raw.fif'
        #self.suffix = '_no60hz_bp_raw.fif'
        self.suffix = '_no60hz_ref_bp_raw.fif'

