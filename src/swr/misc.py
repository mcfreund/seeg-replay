import mne
import numpy as np
import matplotlib.pyplot as plt
from src.shared.utils import *

def check_swr_times_against_matlab(swrs):

    matlab_swr_times = np.array(pd.read_csv('./data/e0010GP_encoding_swr_times_from_matlab.csv'))

    plt.plot(matlab_swr_times[:,0] - swrs['e0010GP']['Encoding']['CMHIP2'].tbeg[1:])

    import scipy as sp
    sp.io.load_mat('swr_times_from_matlab.mat')
    sp.io.loadmat('swr_times_from_matlab.mat')
    sp.io.loadmat('swr_times_from_matlab.mat')['times']
    matlab_swr_times = sp.io.loadmat('swr_times_from_matlab.mat')['times']

    durations = matlab_swr_times[:,1] - matlab_swr_times[:,0]
    devs_beg  = matlab_swr_times[:,0] - swrs['e0010GP']['Encoding']['CMHIP2'].tbeg
    devs_end  = matlab_swr_times[:,1] - swrs['e0010GP']['Encoding']['CMHIP2'].tend

    plt.figure()
    plt.plot(durations)
    plt.plot(devs_beg/durations)
    plt.plot(devs_end/durations)
    plt.legend({'Matlab SWR length','Python SWR beginning time error','Python SWR end time error'})
    plt.xlabel('SWR Number')


def print_trial_times(subjs, sessions):
    for subj in subjs:
        for sess in sessions:
            raw = mne.io.read_raw('/home/dan/projects/work/seeg_data/processed/'+subj+'/'+sess+'/'+subj+'_'+sess+'_raw.fif', verbose='Error')
            starts_msk  = raw._annotations.description == 'trial_start'
            start_times = raw._annotations[starts_msk].onset

            print(f'\nStart times for {subj}, {sess}')
            print(np.round(start_times,1))




def compare_files_at_swr():
    # Compare files at SWRs

    #swrs = dill_read('./e0010GP_Encoding_SameDayRecall_CMHIP2_swrs')
    swrs = dill_read('./data/ca1_swrs-03-04-verified-identical.pt')
    beg, end = swrs['e0010GP']['Encoding']['CMHIP2'].ctx_beg[0], swrs['e0010GP']['Encoding']['CMHIP2'].ctx_end[0]

    #beg = int(swrs['ctx_beg'][0])
    #end = int(swrs['ctx_end'][0])

    raw_1a = mne.io.read_raw('/home/dan/projects/work/seeg_data/megagroup/e0010GP/Encoding/e0010GP_Encoding_raw.fif')
    raw_1b = mne.io.read_raw('/home/dan/projects/work/seeg_data/processed/e0010GP/Encoding/e0010GP_Encoding_raw.fif')

    raw_2a = mne.io.read_raw('/home/dan/projects/work/seeg_data/megagroup/e0010GP/Encoding/e0010GP_Encoding_no60hz_raw.fif')
    raw_2b = mne.io.read_raw('/home/dan/projects/work/seeg_data/processed/e0010GP/Encoding/e0010GP_Encoding_no60hz_raw.fif')

    raw_3a = mne.io.read_raw('/home/dan/projects/work/seeg_data/megagroup/e0010GP/Encoding/e0010GP_Encoding_no60hz_ref_raw.fif')
    raw_3b = mne.io.read_raw('/home/dan/projects/work/seeg_data/processed/e0010GP/Encoding/e0010GP_Encoding_no60hz_ref_raw.fif')
    #raw_3c = mne.io.read_raw('/home/dan/projects/work/seeg_data/processed/e0010GP/Encoding/e0010GP_Encoding_no60hz_uni_raw.fif')
    raw_3d = mne.io.read_raw('/home/dan/projects/work/seeg_data/processed/e0010GP/Encoding/e0010GP_Encoding_no60hz_bip_raw.fif')
    raw_3e = mne.io.read_raw('/home/dan/projects/work/seeg_data/processed/e0010GP/Encoding/e0010GP_Encoding_no60hz_srf_raw.fif')

    raw_4a = mne.io.read_raw('/home/dan/projects/work/seeg_data/megagroup/e0010GP/Encoding/e0010GP_Encoding_no60hz_ref_bp_raw.fif')
    raw_4b = mne.io.read_raw('/home/dan/projects/work/seeg_data/processed/e0010GP/Encoding/e0010GP_Encoding_no60hz_ref_bp_raw.fif')

    plt.ion()
    plt.figure(figsize=[12,8])
    
    plt.subplot(4,1,1)
    plt.plot(raw_1a['C-MHIP2'][0][0,beg:end])
    plt.plot(raw_1b['C-MHIP2'][0][0,beg:end])
    plt.legend(['mega-raw', 'proc-raw'])
    
    plt.subplot(4,1,2)
    plt.plot(raw_2a['C-MHIP2'][0][0,beg:end])
    plt.plot(raw_2b['C-MHIP2'][0][0,beg:end])
    plt.legend(['mega-no60hz', 'proc-no60hz'])
    
    plt.subplot(4,1,3)
    plt.plot(raw_3a['C-MHIP2'][0][0,beg:end])
    plt.plot(raw_3b['C-MHIP2'][0][0,beg:end])
    #plt.plot(raw_3c['C-MHIP2'][0][0,beg:end])
    plt.plot(raw_3d['C-MHIP2'][0][0,beg:end])
    plt.plot(raw_3e['C-MHIP2'][0][0,beg:end])
    #plt.legend(['mega-no60hz-ref', 'proc-no60hz-ref', 'proc-no60hz-uni', 'proc-no60hz-bip', 'proc-no60hz-srf'])
    plt.legend(['mega-no60hz-ref', 'proc-no60hz-ref', 'proc-no60hz-bip', 'proc-no60hz-srf'])
    
    plt.subplot(4,1,4)
    plt.plot(raw_4a['C-MHIP2'][0][0,beg:end])
    plt.plot(raw_4b['C-MHIP2'][0][0,beg:end])
    plt.legend(['mega-no60hz-ref-bp', 'proc-no-60hz-ref-bp'])



    plt.figure()
    plt.plot(raw_3a['C-MHIP2'][0][0,beg:end])
    plt.plot(raw_3e['C-MHIP2'][0][0,beg:end])
    plt.plot(raw_3b['C-MHIP2'][0][0,beg:end])




def check_constructor_pipeline():

    raw_old = mne.io.read_raw('/home/dan/projects/work/seeg_data/megagroup/e0010GP/Encoding/e0010GP_Encoding_raw.fif')
    raw_new = mne.io.read_raw('/home/dan/projects/work/seeg_data/processed/e0010GP/Encoding/e0010GP_Encoding_raw.fif')
    raw_new2 = mne.io.read_raw('raw_from_save_3.fif')

    data1 = dill_read('data_file_make.pt')
    data3 = dill_read('data_file_construct_raw.pt')
    data4 = dill_read('data_file_construct_raw2.pt')

    data1a = dill_read('data_chkpt_load.pt')
    data3a = dill_read('data_chkpt_cnstrct_1.pt')
    data4a = dill_read('data_chkpt_cnstrct_2.pt')
    data5  = dill_read('data_chkpt_parallel.pt')

    plt.ion()
    plt.figure()
    plt.plot(data1[0]['signal'][0:100])
    plt.plot(raw_old['C-MHIP10'][0][0,0:100]*1e6)
    plt.plot(raw_new['C-MHIP10'][0][0,0:100]*1e6)
    plt.legend(['data from .pbz files', 'Jan 27th e0010GP_Encoding_raw', 'new'])
    plt.title('Comparison: Old, New, Data-loader')

    plt.figure()
    plt.plot(data3[0]['signal'][0:100])
    plt.plot(data4[0][0:100]*1e6)
    plt.legend(['chkpt 3', 'chkpt 4'])

    plt.figure()
    plt.plot(data4[0][0:100]*1e6)
    plt.plot(raw_old['C-MHIP10'][0][0,0:100]*1e6)
    plt.plot(raw_new['C-MHIP10'][0][0,0:100]*1e6)
    plt.legend(['chkpt 4', 'old', 'new'])

    plt.figure()
    plt.plot(data4[0][0:100]*1e6)
    plt.plot(raw_old['C-MHIP10'][0][0,0:100]*1e6)
    plt.plot(raw_new['C-MHIP10'][0][0,0:100]*1e6)
    plt.legend(['chkpt 4', 'old', 'new'])

    plt.figure()
    plt.plot(data1a[0]['signal'][0:100])
    plt.plot(data3a[0]['signal'][0:100])
    plt.legend(['internal', 'output'])
    plt.title('Load Check')

    plt.figure()
    plt.plot(data4a[0][0:100]*1e6)
    plt.plot(data5['C-MHIP10'][0][0,0:100]*1e6)
    plt.legend(['into constructor', 'out of constructor'])
    plt.title('Constructor Check')


    plt.plot(raw_new['C-MHIP10'][0][0,0:100]*1e6)
    plt.legend(['Jan 27th e0010GP_Encoding_raw.fif','data load from .pbz files','new raw file'])

##########################################
# Unfinished / snippets / in-progress
###########################################
def save_swr_data():
    df =  np.stack([swrs['e0010GP']['Encoding']['CMHIP2'].idx_beg, swrs['e0010GP']['Encoding']['CMHIP2'].idx_end,swrs['e0010GP']['Encoding']['CMHIP2'].ctx_beg, swrs['e0010GP']['Encoding']['CMHIP2'].ctx_end, swrs['e0010GP']['Encoding']['CMHIP2'].tbeg, swrs['e0010GP']['Encoding']['CMHIP2'].tend]).T
    pddf = pd.DataFrame(df, columns = ['idx_beg', 'idx_end', 'ctx_beg','ctx_end', 'tbeg', 'tend'])
    dill_save(pddf, 'e0010GP_Encoding_SameDayRecall_CMHIP2_swrs.pt')


def ___tmp__():

    x = df['swr_cnt_enc'].to_numpy()
    y = diff1.to_numpy()
    y
    y.to_float()
    y.asfloat()
    y.dtype
    y.astype(np.float))
    y.astype(np.float)
    y.astype(float)
    y = diff1.to_numpy().astype(float)
    sp.stats.linregress(x=df['swr_cnt_enc'].to_numpy().astype(float), y=diff1.to_numpy().astype(float))
    mask = ~np.isnan(x) & ~np.isnan(y)
    mask
    x
    y
    sp.stats.linregress(x=[mask], y=y[mask])
    mask
    sp.stats.linregress(x=x[mask], y=y[mask])
    y = diff2.to_numpy().astype(float)
    y
    mask = ~np.isnan(x) & ~np.isnan(y)
    sp.stats.linregress(x=x[mask], y=y[mask])