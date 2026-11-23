# Roadmap: beyond weather

The weather work (PRs #3-#6) is done. These are the next questions, ranked by how
likely they are to give a clear, honest answer with this data. The ranking comes
from a literature review (sources at the end) plus what the dataset can support:
every call has a timestamp to the second, an `Initial Type`, a priority, a beat,
and dispatch / arrival / cleared times. Until now only daily counts were used.

Status key: **[ ]** not started, **[~]** in progress, **[x]** done.

## Now

### 1. Darkness and daylight saving time [~]
- **Question:** does losing evening daylight raise robbery and other street crime
  in the hour after sunset? Does the spring clock change raise crash calls?
- **Prior:** Doleac & Sanders find robbery falls 7% over the day after the spring
  change and 27% in the hour after sunset. Fatal crashes rise about 6% in the week
  after the spring change, mostly in the morning.
- **Design:** about 18 clock changes, 2017-2026. Compare the same clock hours in
  the weeks just before and after each change, using hourly counts. Clock time
  stays fixed while daylight jumps an hour. Sunset times are computed (no new data).
- **Data needed:** none new.

### 2. Response times [~]
- **Question:** do snow, heavy rain, or a busy day slow police response,
  especially to top-priority calls?
- **Design:** dispatch-to-arrival time by priority, day-level medians against the
  weather categories and same-day call volume.
- **Data needed:** none new (the `Dispatch Arrive` and `CallRoute Dispatch` fields).

### 3. Full moon [~]
- **Question:** the myth. Studies consistently find no effect; officers widely
  believe there is one.
- **Design:** moon phase from an astronomical formula. Residuals of the existing
  out-of-sample model on full-moon days vs other days, with a bootstrap CI. Report
  what effect size the test could have detected.
- **Data needed:** none new.

## Next

### 4. School days [ ]
- **Prior:** Jacob & Lefgren find that on school days juvenile property crime is
  14% lower and violent crime 28% higher (identified from teacher training days).
- **Design:** non-holiday weekdays with no school (MCPS professional days, early
  dismissals) vs school days, by hour around 3pm dismissal; fights, disorder,
  theft. Exclude snow closures (snow already moves counts).
- **Data needed:** MCPS calendars for about 9 school years, entered by hand.

### 5. Policy changes [ ]
- July 1, 2023: Maryland bars stops and searches based only on cannabis odor
  (HB 1071). Watch TRAFFIC VIOLATION and drug-related calls; may bear on the
  unexplained 24% wind effect on traffic-violation calls.
- March 2020: COVID lockdowns. June 2022: Maryland juvenile justice reform.
- **Caveat:** many calls are officer-initiated, so a break can be a change in
  police behavior, not crime. The README must say so.

## Later / probably underpowered

### 6. Wildfire smoke and air quality [ ]
Studies find 2-4% more violent crime or assaults on smoky days, too small to
detect day to day here. The June 7-8, 2023 Canadian smoke days would work as a
single-event case study. Use EPA AirData daily PM2.5 (back past 2017); Open-Meteo
air-quality history only starts Aug 2022.

### 7. Pay and benefit cycles [ ]
Carr & Packham find grocery-store theft rises late in the SNAP benefit month.
Maryland spreads SNAP issuance over days 4-23, which blurs the cycle. Day-of-month
and federal biweekly paydays against theft calls cost nothing to check.

### 8. Sports [ ]
Card & Dahl: an upset home NFL loss raises at-home domestic violence about 10%,
right after the game. Test with Commanders games and betting lines. MoCo has few
domestic calls per evening, so count them before investing time.

### Checked: 2025 federal shutdown
Oct 1 - Nov 12, 2025: daily incidents within chance of the model's prediction
(-0.1%, +/-1.5%). Weak test: the model uses recent counts, so it absorbs slow,
sustained shifts. Not worth more work unless with a model that has no recent lags.

## Sources
- Doleac & Sanders, darkness and robbery: https://batten.virginia.edu/under-cover-darkness-how-ambient-light-influences-criminal-activity
- DST and fatal crashes (CU Boulder): https://www.colorado.edu/today/2020/01/30/spring-forward-daylight-saving-time-brings-surge-deadly-crashes
- Jacob & Lefgren, school days and juvenile crime: https://www.nber.org/papers/w9653
- Card & Dahl, NFL upsets and family violence: https://nber.org/papers/w15497
- Herrnstadt et al., air pollution and crime: https://www.aeaweb.org/doi/10.1257/app.20190091
- Wildfire smoke and crime, 21 US cities: https://link.springer.com/article/10.1007/s10640-025-01053-2
- Carr & Packham, SNAP timing and crime: https://business.purdue.edu/news/features/2022/snap.php
- Maryland HB 1071 fiscal note: https://mgaleg.maryland.gov/Pubs/BudgetFiscal/2023RS-HB1071-REIN.pdf
- Lunar cycles and crime (OJP): https://www.ojp.gov/ncjrs/virtual-library/abstracts/bad-moon-rise-lunar-cycles-and-incidents-crime
