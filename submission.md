# Submission Responses & Report
## Responses
**Please provide a paragraph detailing your approach.**

We were (are) interested in understanding the relationship between replay and task performance within the broader context of network-level activity changes. In order to examine this, we characterized subjects’ behavior, neural data, relations between them, along with instances of putative replay (assessed via sharp-wave-ripple detection in CA1). To characterize behavior, we plotted each subject's encoding, same-day, and next-day task performance and analyzed distributions of responses. We attempted to link this with neural data in a number of ways: First, we associated putative SWRs, along with other oscillatory events, with trial information to discern whether event distributions correlated with performance or improvement over time. Next, we characterized the neural data according to time, frequency, and time-frequency (localized power) components, and we considered these in terms of task periods and SWRs, at individual trials and in aggregate (i.e. via peri-task-period-averages). Third, we ran traditional decoding analyses to assess how predictable task-periods and subject responses were from their neural data. And finally, we applied an auto-and-cross-history decoding pipeline to the neural data to determine whether (and to what extent) behavioral prediction was facilitated by neural data prediction.

**What was your rationale behind choosing that method?**

There are many specific methods we used. Broadly speaking, we wanted to characterize the data as well as we could, suggesting the various peri-event-time (task period and SWR) analyses and the decoding analyses. More specifically, we preprocessed the data using standard best practices (line-noise removal via spatial filter, bandpass filtering, Laplacian re-referencing), and we detected SWRs, along with other oscillatory coupling events (beta-gamma in particular) using an improved version of Matt Van der Meer’s toolbox we developed. We noticed that the Van der Meer toolbox primarily detects gamma-event outliers, whereas SWRs and other events of interest have multiple frequency components. We therefore modified this pipeline to enable conjunctive-event detection (e.g. joint beta, gamma deviations) and we ran this across all electrodes in an effort to understand oscillatory events relative to behavior. Complementing this, we chose to apply standard multivariate analysis methods used to characterize working memory representations in continuous report tasks such as inverted encoding models, representational similarity analysis, and time-resolved decoding. Lastly, because neural network methods are competitive with traditional filter-style and decoding analyses, we were interested in developing one, and our design choices prioritized simplicity and the prospective ability to make model comparisons.

## Report

### Behavior
Subjects performed the task well during the encoding period, and generally performed poorly in subsequent periods. This can be seen by inspecting plots of location and color accuracy for each session. An example participant can be shown below, and other participants behavioral data can be seen in ```./figs/behavior```.

(./figs/report/figure_1.png)

The approximate uniformity of the kernel density estimates for the subject's response distributions during same-day recall and next-day recall is both notable and typical.

Nonetheless, we considered subjects' relative accuracy across task periods and the possibility that they consistently remembered some movie-location associations but not others. This is summarized in the following plot:

(./figs/report/figure_2.png)

It was therefore possible, but not especially likely, that sharp-wave ripples would relate to behavior. 

### Sharp-wave Ripple Detection
To detect sharp-wave ripples, we adapted the Van der Meer lab toolbox to our use-case. We wrote analysis code in matlab to natively interface with the toolbox, generated a series of SWRs, and visually inspected them with the help of M.V.d.M. This established the parameters we used for SWR detection, including minimum durations, z-scored power thresholds, and false positive rejection methods. Specifically, our detection pipeline bandpass-filters the CA1 data between 80 and 100Hz, Hilbert-transforms the result, converts the complex result to power (via complex magnitude), z-scores the result, and marks sections of the LFP trace with sustained power > 3 standard deviations with a peak of at least > 5 standard deviations. Several sharp wave ripples from this pipeline are plotted below, and plots of all of them can be found in ```./figs/swrs/standard/```.

(./figs/report/figure_3.png)
(./figs/report/figure_4.png)

Observing the results of this pipeline, it was evident that some of the detected SWRs were more canonical than others, and the canonical SWRs tended to include a notable beta-frequency component (the sharp-wave preceding the ripple). Because the pipeline we just described from the Van der Meer lab only implements gamma detection (and because we were interested in oscillatory events more broadly), we decided to expand the functionality of our version to look for conjunctive oscillatory activity. This expanded pipeline operates in the same way (bandpass, hilbert, power, z-score, deviation detection) and allowed us to generate a second set of putative SWRs with more canonical properties. Two are shown here, whereas the full set is found in ```./figs/swrs/joint/```. A less-canonical SWR ruled out by this method is also shown.

(./figs/report/figure_5.png)
(./figs/report/figure_6.png)
(./figs/report/figure_7.png)

Next, we considered the relationships between these and behavior.

### Sharp-wave Ripple Associations with Behavior
In order to associate sharp-wave ripple occurance with behavior, we determined the timing of each sharp-wave ripple relative to the behavioral codes in our neural data. As a result, we obtained binned SWR counts for each session and each subject. We plotted these to inspect them, then ran regression analyses determining whether SWR counts appeared to impact either performance or improvement. Example results for these associations are included here.

(Some plots - not sure the exact number yet.)

The regression results plotted here were not significant, leading us to conclude that SWRs during the behavioral periods were not strongly, or not linearly, associated with subsequent behavioral performance. We intend to extend these analyses with further characterization of the SWRs themselves, as well as an analysis of the subjects' sleep data, however.

### Model-free and Model-light Neural and Neural-Behavioral Analyses
Even in the absence of SWR-behavioral associations, both SWRs and the other oscillatory events we can detect may have interesting correlates across other brain areas. To assess this, we first examined the neural data generally, characterizing different regions' activity in terms of time, frequency, and time-frequency analyses. Stereotyped responses to task periods were observed in a number of regions, especially those distributed across frontal cortex. These are plainly visible in the time-series data, especially for the clip-cue task periods, during which participants had to prepare their upcoming responses. Examples of these include the following:

(Some plots - not sure the exact number yet.)

If this data were externally recorded, we would regard some of these large deviations as likely artefacts. In this case, their provenance is not clear, especially since we lack observational data regarding the subjects' behaviors during the relevant time-framces. For example, it seems plausible that the subject depicted in (figure number) above consistently made gross movements during the clip-cue period. However, the lack of correlation between these gross patterns across electrodes, and their localization in specific areas of frontal cortex lends support to the idea that they may not be artefacts.

In the frequency domain, we observed widespread alpha oscillations, which is not surprising given the large fraction of the time subjects were likely resting, and time-frequency analyses indicated events across all the usual activity bands (delta, theta, alpha, beta, gamma). 

(... A number of these were task locked, say more about regions, etc. ...)

Areas with notable event detection included (...), which showed (...). All of our plots related to these events can be found in ```/figs/other-events/``` In order to understand the relations between these, task variables, and SWRs more systematically, we computed peri-event-time averages of distributed activity and power within task periods and proximal to SWRs. Several plots of these are shown here.

(Plots of peri-trial-period and peri-SWR averages)

These suggest to us that (...) but we will need to follow up on these analyses to explore the matter further.

Finally, the regression analyses we preformed with the SWRs were also possible with our expanded event detection pipeline. Plots of event counts relative to behavior are included below.

(Plots of other event counts vs behavior)

(Interpretation)

### Multivariate Working Memory Analyses
(Atsushi, can you write a basic blurb here please?)


### Network Decoding Analyses
To complement the standard set of decoding analyses, we were interested in developing predictions based on neural network models. We reasoned that the relative lack of behavioral data would make our current investigation difficult for traditional approaches, but that the relative abundance of neural data could potentially be used to enhance the already competitive performance of network models. To make use of this data, we therefore trained several simple neural networks to predict upcoming neural data, and subsequently added readouts for task period.

Some electrodes had more predictable data than others. We found that:

Furthermore, this approach allowed us to assess the shared information between electrodes, by considering how well groups of electrodes predicted one another.

Behaviorally ...


### Discussion
What have we learned? What will we do next?
