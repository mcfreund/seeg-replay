from matplotlib.pyplot import pyplot as plt

def plot_swr(i, swrs, lfp_bp, pow_z, contact):
    plt.figure(figsize=[8,6])
    plt.subplot(3,1,1)
    plt.plot(swrs.ctime[i], swrs.ctxts[i])
    plt.plot(swrs.times[i], swrs.traces[i])
    
    plt.subplot(3,1,2)
    plt.plot(swrs.ctime[i], lfp_bp[ swrs.ctx_beg[i]:swrs.ctx_end[i] ])
    plt.plot(swrs.times[i], lfp_bp[ swrs.idx_beg[i]:swrs.idx_end[i] ])

    plt.subplot(3,1,3)
    plt.plot(swrs.ctime[i], pow_z[ swrs.ctx_beg[i]:swrs.ctx_end[i] ])
    plt.plot(swrs.times[i], pow_z[ swrs.idx_beg[i]:swrs.idx_end[i] ])
    
    plt.title(f'Contact {contact} SWR {i} ')
    plt.tight_layout()

def plot_swrs_and_behavior(df):
    # Encoding counts vs performance
    plt.figure()
    plt.plot(df['swr_cnt_enc'], df['err_pos_enc'],'o')
    plt.xlabel('Num. SWRs'); plt.ylabel('Position Error')
    plt.title('e0010GP Encoding SWR by Encoding Accuracy')

    plt.figure()
    plt.plot(df['swr_cnt_enc'], df['err_pos_same'],'o')
    plt.xlabel('Num. SWRs'); plt.ylabel('Position Error')
    plt.title('e0010GP Encoding SWR by SameDay Accuracy')

    plt.figure()
    plt.plot(df['swr_cnt_enc'], df['err_pos_next'],'o')
    plt.xlabel('Num. SWRs'); plt.ylabel('Position Error')
    plt.title('e0010GP Encoding SWR by NextDay Accuracy')

    plt.figure()
    diff1 = abs(df['err_pos_enc'])-abs(df['err_pos_same'])
    diff2 = abs(df['err_pos_enc'])-abs(df['err_pos_next'])
    plt.plot(df['swr_cnt_enc']-0.05, diff1,'o')
    plt.plot(df['swr_cnt_enc']+0.05, diff2,'o')

    plt.xlabel('Num. SWRs'); plt.ylabel('Position Error Change')
    plt.title('SWR Counts vs Improvement')
    plt.legend(['Encoding vs Same', 'Encoding vs Next'])


    # Same-day counts vs performnace
    plt.figure()
    plt.plot(df['swr_cnt_same'], df['err_pos_same'],'o')
    plt.xlabel('Num. SWRs'); plt.ylabel('Position Error')
    plt.title('e0010GP SameDay SWR by SameDay Accuracy')

    plt.figure()
    plt.plot(df['swr_cnt_same'], df['err_pos_next'],'o')
    plt.xlabel('Num. SWRs'); plt.ylabel('Position Error')
    plt.title('e0010GP SameDay SWR by NextDay Accuracy')

    # Next-day counts vs performance
    plt.figure()
    plt.plot(df['swr_cnt_next'], df['err_pos_next'],'o')
    plt.xlabel('Num. SWRs'); plt.ylabel('Position Error')
    plt.title('e0010GP NextDay SWR by NextDay Accuracy')


    # SWR counts by trial and condition
    plt.figure()
    plt.plot(df['trial'], df['swr_cnt_enc'],'o')
    plt.xlabel('Trial Number')
    plt.ylabel('SWR Count')
    plt.title('e0010GP Encoding SWR Count by Trial')

    plt.figure()
    plt.plot(df['condition'], df['swr_cnt_enc'],'o')
    plt.xlabel('Condition Number')
    plt.ylabel('SWR Count')
    plt.title('e0010GP Encoding SWR Count by Condition')

