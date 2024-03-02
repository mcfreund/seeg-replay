class PathPresets:
    def __init__(self,loc):
        match loc:
            case 'oscar':
                # Location of imaging parsed.xlsx files and preprocessed MNE "raw" files
                self.imaging   = '/oscar/data/brainstorm-ws/seeg_data/Memory Task Data/Imaging/Epilepsy/'
                self.raw_files = '/oscar/data/brainstorm-ws/megagroup_data/'
            
            case 'dan-xps-15':
                self.imaging   = '/home/dan/projects/work/seeg_data/original/Imaging/Epilepsy/'
                self.raw_files = '/home/dan/projects/work/megagroup_data/'

            case '_':
                raise Exception(f'No preset paths for {loc}')
