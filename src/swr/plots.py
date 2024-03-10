import matplotlib.pyplot as plt

def plot_swr(i, swrs, lfp_bp, pow_z, contact):

    plt.figure(figsize=[8,6])
    plt.suptitle(f'e0010GP Encoding and SameDayRecall\nContact {contact} SWR {i}')

    plt.subplot(3,1,1)
    plt.plot(swrs.ctime[i], swrs.ctxts[i])
    plt.plot(swrs.times[i], swrs.traces[i])
    plt.ylabel('LFP [V]')

    plt.subplot(3,1,2)
    plt.plot(swrs.ctime[i], lfp_bp[ swrs.ctx_beg[i]:swrs.ctx_end[i] ])
    plt.plot(swrs.times[i], lfp_bp[ swrs.idx_beg[i]:swrs.idx_end[i] ])
    plt.ylabel('Band-Passed LFP, 80-100 Hz')

    plt.subplot(3,1,3)
    plt.plot(swrs.ctime[i], pow_z[ swrs.ctx_beg[i]:swrs.ctx_end[i] ])
    plt.plot(swrs.times[i], pow_z[ swrs.idx_beg[i]:swrs.idx_end[i] ])
    plt.ylabel('Z-scored Power')
    plt.xlabel('Time [s]')

    plt.tight_layout()

def plot_swrs_and_behavior(df):
    # 
    plt.ion()

    figsize = [5,4]

    # Encoding counts vs performance
    plt.figure(figsize=figsize)
    plt.plot(df['swr_cnt_enc'], abs(df['err_pos_enc']),'o')
    plt.xlabel('Num. SWRs'); plt.ylabel('Position Error')
    plt.title('e0010GP Encoding SWR by Encoding Accuracy')
    plt.tight_layout()

    plt.figure(figsize=figsize)
    plt.plot(df['swr_cnt_enc'], abs(df['err_pos_same']),'o')
    plt.xlabel('Num. SWRs'); plt.ylabel('Position Error')
    plt.title('e0010GP Encoding SWR by SameDay Accuracy')
    plt.tight_layout()

    plt.figure(figsize=figsize)
    plt.plot(df['swr_cnt_enc'], abs(df['err_pos_next']),'o')
    plt.xlabel('Num. SWRs'); plt.ylabel('Position Error')
    plt.title('e0010GP Encoding SWR by NextDay Accuracy')
    plt.tight_layout()

    plt.figure(figsize=figsize)
    diff1 = abs(df['err_pos_enc'])-abs(df['err_pos_same'])
    diff2 = abs(df['err_pos_enc'])-abs(df['err_pos_next'])
    plt.plot(df['swr_cnt_enc']-0.05, diff1,'o')
    plt.plot(df['swr_cnt_enc']+0.05, diff2,'o')
    plt.tight_layout()

    plt.xlabel('Num. SWRs'); plt.ylabel('Position Error Change')
    plt.title('SWR Counts vs Improvement')
    plt.legend(['Encoding vs Same', 'Encoding vs Next'])
    plt.tight_layout()


    # Same-day counts vs performnace
    plt.figure(figsize=figsize)
    plt.plot(df['swr_cnt_same'], abs(df['err_pos_same']),'o')
    plt.xlabel('Num. SWRs'); plt.ylabel('Position Error')
    plt.title('e0010GP SameDay SWR by SameDay Accuracy')
    plt.tight_layout()

    plt.figure(figsize=figsize)
    plt.plot(df['swr_cnt_same'], abs(df['err_pos_next']),'o')
    plt.xlabel('Num. SWRs'); plt.ylabel('Position Error')
    plt.title('e0010GP SameDay SWR by NextDay Accuracy')
    plt.tight_layout()

    # Next-day counts vs performance
    plt.figure(figsize=figsize)
    plt.plot(df['swr_cnt_next'], abs(df['err_pos_next']),'o')
    plt.xlabel('Num. SWRs'); plt.ylabel('Position Error')
    plt.title('e0010GP NextDay SWR by NextDay Accuracy')
    plt.tight_layout()


    # SWR counts by trial and condition
    plt.figure(figsize=figsize)
    plt.plot(df['trial'], df['swr_cnt_enc'],'o')
    plt.xlabel('Trial Number')
    plt.ylabel('SWR Count')
    plt.title('e0010GP Encoding SWR Count by Trial')
    plt.tight_layout()

    plt.figure(figsize=figsize)
    plt.plot(df['condition'], df['swr_cnt_enc'],'o')
    plt.xlabel('Condition Number')
    plt.ylabel('SWR Count')
    plt.title('e0010GP Encoding SWR Count by Condition')
    plt.tight_layout()

