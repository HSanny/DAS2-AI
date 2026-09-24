# Sources

Every published work this system leans on, what it is used for, and whether the
citation has been checked against the source.

This file exists because the honest answer to "is this method sound?" is
different for each layer. Some of what DAS2-AI does is a named standard test
with a manual you can look up; some is a published method applied to a
different network than it was validated on; and some is our own construction
with no external support at all. Those three deserve different amounts of
trust, and mixing them up is how a system talks its operators into believing
an invention.

**Status column**

| | |
|---|---|
| **verified** | citation checked against the source or publisher record; author list, journal, volume, pages and the specific figure quoted all confirmed |
| **partly verified** | the work exists and is correctly attributed, but a specific claim made about it could not be checked from here |
| **removed** | was cited in the code and could not be verified; the attribution has been taken out |

Nothing below is a validation of this system on PUB's network. No paper here
was written about this estate, and none of them has been re-run against it.
Where a method is borrowed the code says which part is borrowed and which part
is ours — see in particular `das2/detect/changepoint.py`, whose
INSTRUMENT_OFFSET discriminator is entirely our own and says so.

---

## The standard we follow

### IOOS QARTOD — *Manual for Real-Time Quality Control of Water Level Data*
**Status: verified.** U.S. Integrated Ocean Observing System.
<https://ioos.noaa.gov/ioos-in-action/manual-real-time-quality-control-water-level-data/>

Used for: the names and numbers of the L1 health tests, and the flag
vocabulary, in `das2/models.py` (`QARTOD_TEST`, `QartodFlag`,
`FLAG_PRECEDENCE`).

Confirmed: Test 1 Timing/Gap, Test 4 Gross Range, Test 6 Spike, Test 8 Flat
Line, Test 9 Multi-Variate, Test 10 Attenuated Signal, Test 11 Neighbor. Flags
1 Pass, 2 Not Evaluated, 3 Suspect, 4 Fail, 9 Missing.

Two notes on fidelity:

* Our enum spells flag 2 `UNKNOWN`, which is what the reference implementation
  [`ioos_qc.qartod.QartodFlags`](https://github.com/ioos/ioos_qc/blob/main/ioos_qc/qartod.py)
  calls it. The manual's prose calls it *Not Evaluated*. Same value, both names
  in use.
* Where we have no QARTOD equivalent the table says `None` explicitly rather
  than omitting the type, so an invention cannot be mistaken for something
  inherited.

---

## Methods we borrow

### Branisavljević, Kapelan & Prodanović (2011)
*Improved real-time data anomaly detection using context classification.*
**Journal of Hydroinformatics 13(3), 307–323.** doi:10.2166/hydro.2011.042
**Status: verified.**
<https://iwaponline.com/jh/article/13/3/307/3054/Improved-real-time-data-anomaly-detection-using>

Used for: the principle behind rain-conditioned detection — classify the
context first, then judge the reading — in `das2/weather/` and the
`STORMWATER_RESPONSE` signature.

### Rosin, Kapelan, Keedwell & Romano (2022)
*Near real-time detection of blockages in the proximity of combined sewer
overflows using evolutionary ANNs and statistical process control.*
**Journal of Hydroinformatics 24(2), 259–273.** doi:10.2166/hydro.2022.036
**Status: partly verified.**
<https://iwaponline.com/jh/article/24/2/259/87412/Near-real-time-detection-of-blockages-in-the>

Used for: precedent that unusual behaviour can be detected from level data
without a hydraulic model.

*What could not be checked:* an earlier version of `event_signatures.yaml`
stated that this method "degrades as rainfall intensity rises". The publisher
is unreachable from this environment and no open copy could be found, so that
claim has been removed rather than passed on to PUB. What the abstract does
support is the general result — blockages detected quickly and reliably with
few false alarms.

### Bakker, Jung, Vreeburg, van de Roer, Lansey & Rietveld (2014)
*Detecting pipe bursts using Heuristic and CUSUM methods.*
**Procedia Engineering 70, 85–92.**
**Status: verified.**
<https://repository.tudelft.nl/record/uuid:8204fa5b-9c7b-4caa-b590-74bb733362fc>

Used for: the reference field number in the `PRESSURE_LOSS` signature —
**44.4% of bursts detected at a 5.0% false-alarm rate**.

Read it with its sample size: that is 4 of 9 reported burst events in a winter
subset. It is the right order of magnitude for what burst detection achieves in
the field, not a precise expectation, and the config file now says so.

### Mandapaka & Qin (2013)
*Analysis and characterization of probability distribution and small-scale
spatial variability of rainfall in Singapore using a dense gauge network.*
**Journal of Applied Meteorology and Climatology 52, 2781–2796.**
doi:10.1175/JAMC-D-13-0115.1
**Status: verified.**

Used for: the distance over which one rain gauge is evidence about another
place — `DEFAULT_GAUGE_RADIUS_M` and `GAUGE_DECORRELATION_M` in
`das2/weather/provider.py`.

Confirmed: 49 gauges over Singapore; e-folding decorrelation distance ~10 km
at hourly aggregation, ~33 km daily. This is the rare case of a parameter
measured **for this city** rather than assumed, which is why it is used in
preference to any default.

### Talei & Chua (2012)
*Influence of lag time on event-based rainfall–runoff modeling using the data
driven approach.*
**Journal of Hydrology 438–439, 223–233.**
**Status: verified.**
<https://www.sciencedirect.com/science/article/abs/pii/S0022169412002363>

Used for: determining rainfall-to-response lag by cross-correlation, in
`das2/weather/lag.py`. The study is on a **Singapore** catchment and derives
lag the same way.

### Gericke & Smithers (2014)
*Review of methods used to estimate catchment response time for the purpose of
peak discharge estimation.*
**Hydrological Sciences Journal 59(11), 1935–1971.**
doi:10.1080/02626667.2013.866712
**Status: verified.**
<https://www.tandfonline.com/doi/full/10.1080/02626667.2013.866712>

Used for: the warning that catchment response time is the dominant error term,
and that a wrong lag is worse than no lag — which is why `lag.py` gates on
confidence and falls back to an unshifted window.

### Thornhill, Choudhury & Shah (2004)
*The impact of compression on data-driven process analyses.*
**Journal of Process Control 14(4), 389–398.** doi:10.1016/j.jprocont.2003.06.003
**Status: partly verified.**

Used as: the nearest published treatment of inferring compression and
quantisation from historian data, behind `QUANTISATION_COLLAPSE`.

*Be precise about what this supports:* the paper is about detecting
**compression** in archived process data. It is the same measurement problem
and the same research group as the quantisation work, but it is not a QC test
for our finding, and `QUANTISATION_COLLAPSE` is mapped to no QARTOD test for
that reason.

---

## Taxonomy: where "an offset is a sensor fault" comes from

These three establish that instrument offset belongs in a fault taxonomy at
all. They do **not** validate our method for telling an offset from a real
water-level change, which is ours.

### Ni, Ramanathan, Chehade, Balzano, Nair, Zahedi, Kohler, Pottie, Hansen & Srivastava (2009)
*Sensor Network Data Fault Types.*
**ACM Transactions on Sensor Networks 5(3).**
**Status: verified.**
<https://escholarship.org/content/qt1rb4285n/qt1rb4285n_noSplash_d3217554a7c821c8d7792a08ac67261e.pdf>

### Sharma, Golubchik & Govindan (2010)
*Sensor faults: detection methods and prevalence in real-world datasets.*
**ACM Transactions on Sensor Networks 6(3).**
**Status: verified.**
<https://dl.acm.org/doi/10.1145/1754414.1754419>

### Leigh, Alsibai, Hyndman, Kandanaarachchi, King, McGree et al. (2019)
*A framework for automated anomaly detection in high frequency water-quality
data from in situ sensors.*
**Science of the Total Environment 664, 885–898.**
**Status: verified.** Open preprint: <https://arxiv.org/abs/1810.13076>

---

## Statistics

### Rousseeuw & Leroy (1987)
*Robust Regression and Outlier Detection.* Wiley.
**Status: verified** (standard reference work).

### Hampel (1974)
*The influence curve and its role in robust estimation.*
**Journal of the American Statistical Association 69(346), 383–393.**
**Status: verified** (standard reference work).

Both are cited in `das2/detect/conventional.py` for the masking effect: a
sustained excursion inflates the σ computed from the window that contains it,
which is why that module scores each sensor twice — once against the whole
window and once against the hours before the event.

The multiple-comparison correction in the same module
(`1 - (1 - erfc(z/√2))^n`, giving P(noise touches 3σ in 900 samples) = 91%) is
elementary probability, not a borrowed result, and is reproduced by
`tests/test_conventional.py`.

---

## Removed

### Leow et al. (2017) — removed
An earlier version of `das2/detect/changepoint.py` cited *Leow et al. (2017),
Environ. Sci.: Water Res. Technol. 3(2)* for the figure "189 of 219 alarms
were maintenance". Searching for this paper and that figure produced nothing.
The attribution has been removed.

The argument it was supporting does not depend on it and still stands: a
recalibration is a scheduled human act, so the maintenance work-order log
would settle offset-versus-event without any algorithm at all. That remains
the single highest-value data request, and it is now made on its own merits.

---

## What has no citation, and says so

These are our constructions. Each is marked in the code, emitted as SUSPECT
rather than FAIL where it is an inference, and none is validated against
anything:

* **`INSTRUMENT_OFFSET`'s discriminator** — instantaneous transition plus
  sustained hold. `changepoint.py` states plainly that no paper validates the
  two tests in combination.
* **The three `ASSET_*` types** — cross-channel contradiction with electrical
  attribution. `asset.py` gives the hydraulic reasoning, including why the
  direction of the power change is deliberately not used (radial and axial
  pumps move opposite ways at shutoff).
* **`das2/data/event_signatures.yaml`** — hydraulic reasoning, not derived
  from data. Every entry carries a `status` and a `would_change_it`, nothing
  ships as `confirmed`, and a signature never changes what the system decides.
* **Every threshold in `config.yaml`** — engineering judgement, pending the
  commissioned HH/H/L/LL limits from the SCADA. That export would replace
  roughly ninety guessed numbers with the plant's own.

---

## Two things worth asking PUB for

Both would do more for accuracy than any further work on the algorithms:

1. **The maintenance / work-order log.** Turns "is this a recalibration or a
   real level change?" from an inference into a lookup.
2. **The commissioned per-point alarm limits (HH/H/L/LL) and engineering
   ranges** from the Fujitsu SCADA. Replaces the hand-set thresholds with the
   plant's own commissioned values.
